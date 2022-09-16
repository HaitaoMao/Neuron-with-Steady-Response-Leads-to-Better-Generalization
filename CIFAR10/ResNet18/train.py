import pdb
import argparse
import numpy as np
import os
from tqdm import tqdm

from util.misc import CSVLogger
from util.cutout import Cutout

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision.utils import make_grid
from torchvision import datasets, transforms

from model.train_stable import train_cuda


model_options = ['resnet18']
dataset_options = ['cifar10']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--num_of_batches_for_mean', type=int, default=1, help='the number of batches that used for calculating the mean of node output')
parser.add_argument('--lambda_balance', nargs = "*", type=float, default=[0.5], help='to balance the node stable loss and cross entropy loss')
parser.add_argument('--num_of_layers_with_nodeStableLoss', type=int, default=1, help='the number of layers for backward node stable loss, counting from the last one')
    
parser.add_argument('--dataset', '-d', default='cifar10',
                    choices=dataset_options)
parser.add_argument('--model', '-a', default='resnet18',
                    choices=model_options)
parser.add_argument('--batch_size', type=int, default=100,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='learning rate')
parser.add_argument('--data_augmentation', action='store_true', default=False,
                    help='augment data by flipping and cropping')
parser.add_argument('--cutout', action='store_true', default=False)
parser.add_argument('--n_holes', type=int, default=1,
                    help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16,
                    help='length of the holes')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 0)')
parser.add_argument('--gpu_id', type=int, default=0,
                    help='gpu_id (default 0)')
parser.add_argument('--wd', type=float,default=5e-4)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = True  # Should make training should go faster for large models

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

cuda_device = torch.device('cuda:' + str(args.gpu_id))

test_id = args.dataset + '_' + args.model

print(args)

# Image Preprocessing
normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std = [x / 255.0 for x in [63.0, 62.1, 66.7]])

train_transform = transforms.Compose([])
if args.data_augmentation:
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize)
if args.cutout:
    train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))


test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])

if args.dataset == 'cifar10':
    num_classes = 10
    train_dataset = datasets.CIFAR10(root='data/',
                                     train=True,
                                     transform=train_transform,
                                     download=True)

    test_dataset = datasets.CIFAR10(root='data/',
                                    train=False,
                                    transform=test_transform,
                                    download=True)

val_num = 10000
split_seed = 7
train_dataset_s, val_dataset_s = torch.utils.data.random_split(train_dataset,
        [len(train_dataset) - val_num, val_num], 
        generator=torch.Generator().manual_seed(split_seed))

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=2)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset_s,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=2)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)

if args.model == 'resnet18':
    cnn = ResNet18(num_classes=num_classes)

cnn = cnn.cuda(cuda_device)
criterion = nn.CrossEntropyLoss().cuda(cuda_device)
cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate,
                                momentum=0.9, nesterov=True, weight_decay=args.wd)

scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)

filename = 'logs/' + test_id + '.csv'
csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'test_acc'], filename=filename)


def test(loader):
    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    for images, labels in loader:
        images = images.cuda(cuda_device)
        labels = labels.cuda(cuda_device)

        with torch.no_grad():
            pred = cnn(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    cnn.train()
    return val_acc

y_log = [[] for i in range(args.num_of_layers_with_nodeStableLoss)]  # only record the hidden and the output layer
target_log = []
device = cuda_device

best_test = 0.0
best_val = 0.0
best_val_test = 0.0
for epoch in range(args.epochs):

    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    progress_bar = tqdm(train_loader)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        images = images.cuda(cuda_device)
        labels = labels.cuda(cuda_device)

        cnn.zero_grad()
        pred = cnn(images)

        xentropy_loss = criterion(pred, labels)
        xentropy_loss.backward()

        stable_loss = train_cuda(cnn, device, args, y_log, target_log, labels)


        cnn_optimizer.step()

        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

    test_acc = test(test_loader)
    val_acc = test(val_loader)
    best_test = max(test_acc, best_test)
    if val_acc > best_val:
        best_val = val_acc
        best_val_test = test_acc
    tqdm.write('epoch: %d\ntest_acc: %.3f; val_acc: %.3f\nbest_val_test: %.3f; best_test: %.3f; best_val: %.3f' % (epoch, test_acc, val_acc, best_val_test, best_test, best_val))

    #scheduler.step(epoch)  # Use this line for PyTorch <1.4
    scheduler.step()     # Use this line for PyTorch >=1.4

    row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)}
    csv_logger.writerow(row)

torch.save(cnn.state_dict(), 'checkpoints/' + test_id + '.pt')
csv_logger.close()
