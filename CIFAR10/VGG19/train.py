from __future__ import print_function
import math
from random import random
from random import seed
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as mplot
from train_stable import *
from VGG19 import Net
from CNN_new import SupCEResNet
from base_function import *
import json
import time



def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='input learning rate for training (default: 0.2)')
    parser.add_argument('--training_epochs', type=int, default=200,
                        help='input training epochs for training (default: 200)')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='input batch size for training (default: 100)')
    parser.add_argument("--lr_decay", type = int, default=0)

    parser.add_argument('--init_model_with_bias', type=int, default=0,
                        help='init model with bias as 0 or constant positive value')
    parser.add_argument('--random_seed', type=int, default=1,
                        help='input random seed for training (default: 1)')

    parser.add_argument('--num_valid', type=int, default=10000)
    parser.add_argument('--num_of_batches_for_mean', type=int, default=1, help='the number of batches that used for calculating the mean of node output')
    parser.add_argument('--lambda_balance', nargs = "*", type=float, default=[0], help='to balance the node stable loss and cross entropy loss')
    parser.add_argument('--num_of_layers_with_nodeStableLoss', type=int, default=1, help='the number of layers for backward node stable loss, counting from the last one')
    parser.add_argument('--model_name', type=str, default= "VGG")
    
    parser.add_argument('--is_schedule', type=bool, default= False)
    parser.add_argument('--factor', type = float, default = 0.05)
    parser.add_argument('--patience', type = int, default = 2)
    parser.add_argument('--threshold', type = float, default = 0.0001)
    parser.add_argument('--is_clip', type=bool, default= False)
    parser.add_argument('--clip_value', type = float, default = 5)


    # set common config
    args = parser.parse_args()
    assert(len(args.lambda_balance) == args.num_of_layers_with_nodeStableLoss)        

    train_loss_result, train_entropy_loss_result, node_stable_loss_result, \
        test_accuracy_result, test_loss_result = [], [], [], [], []
        
    model = Net()
    
    y_log = [[] for i in range(args.num_of_layers_with_nodeStableLoss)]  # only record the hidden and the output layer
    target_log = []

    random_seed = args.random_seed
    seed(random_seed)  # python random seed
    torch.manual_seed(random_seed)  # pytorch random seed

    # set training config
    batch_size = args.batch_size
    training_epochs = args.training_epochs
    learning_rate = args.learning_rate

    # set cpu or gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    init_model_with_bias = args.init_model_with_bias
    
    if init_model_with_bias > 0:
        model.apply(weights_init_apply_with_bias)
    else:
        model.apply(weights_init_apply_without_bias)
    model.to(device)
    print("learning_rate", learning_rate)
    print("parameter", model.parameters())

    optimizer = optim.Adam(model.parameters(), lr=learning_rate) 
    scheduler = None
    if args.is_schedule:  
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, patience=args.patience, \
            threshold=args.threshold, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
    criterion = nn.CrossEntropyLoss()

    transform_train_cifar10 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261)),
    ])

    transform_test_cifar10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261)),
    ])

    transform_train_cifar100 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test_cifar100 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    split_seed = 7
    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train_cifar10)
    testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test_cifar10)
    validset, trainset_s = torch.utils.data.random_split(trainset, [args.num_valid, 50000 - args.num_valid], generator=torch.Generator().manual_seed(split_seed))
        
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=2)


    record_dict = {}
    # train and test the created model
    max_valid_acc = -1
    max_test = 0.0
    max_val_test = 0.0
    is_test_list, valid_acc_list, valid_loss_list, train_acc_list = [], [], [], []

    for epoch in range(training_epochs):
        if epoch == 0:
            test_loss, test_correct, test_accuracy = test(model, device, test_loader, criterion)
            print('Testing set: Average loss: {:.4f}, Accuracy: ({:.4f}%)'.format(test_loss, test_accuracy))
            print('Test initialization end')
        if args.lr_decay == 1:
            if epoch == 100:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate * 0.1
            elif epoch == 120:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate * 0.01

        train_loss, node_stable_loss, cross_entropy_loss, train_acc = train_cuda(model, device, train_loader, optimizer, criterion, batch_size, args, y_log, target_log)
        train_acc_list.append(train_acc)
        train_entropy_loss_result.append(cross_entropy_loss)
        node_stable_loss_result.append(node_stable_loss)
        train_loss_result.append(train_loss)
        print('Train_epoch\tcross_loss\tstable_loss')
        print('{}\t{:.6f}\t{:.6f}'.format(epoch, cross_entropy_loss, node_stable_loss))

        valid_loss, valid_correct, valid_accuracy= test(model, device, valid_loader, criterion)
        print("valide_epoch\tvalid_loss\tvalid_accuracy")
        print("{}\t{:.6f}\t{:.4f}".format(epoch, valid_loss, valid_accuracy))
        is_test = max_valid_acc <= valid_accuracy
        max_valid_acc = max(max_valid_acc, valid_accuracy)
        
        is_test_list.append(is_test)
        valid_acc_list.append(valid_accuracy)
        test_loss, test_correct, test_accuracy = test(model, device, test_loader, criterion)
        if is_test:
            max_val_test = test_accuracy


        max_test = max(max_test, test_accuracy)
        print('Test_epoch\tTest_loss\tTest_accuracy\tmax_test\tmax_val_test')
        print('{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(epoch, test_loss, test_accuracy, max_test, max_val_test))
        test_accuracy_result.append(test_accuracy)
        test_loss_result.append(test_loss)
    record_dict["train_acc"] = train_acc_list
    record_dict["train_loss"] = train_loss_result
    record_dict["train_entropy_loss_result"] = train_entropy_loss_result
    record_dict["node_stable_loss_result"] = node_stable_loss_result
    record_dict["test_loss"] = test_loss_result
    record_dict["test_acc"] = test_accuracy_result
    record_dict["is_test"] = is_test_list
    record_dict["valid_acc"]  = valid_acc_list
    record_dict["valid_loss"]  = valid_loss_list

    record_dict["params"] = args.__dict__
    with open("./result/" + args.model_name + f"_{args.lambda_balance}_{args.random_seed}.json", 'w') as f:
        json.dump(record_dict, f, indent = 4)

if __name__ == '__main__':
    main()
