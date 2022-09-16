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
from src.train_stable import *
from src.DNN import Net
from src.base_function import *
import time



def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='input learning rate for training (default: 0.2)')
    parser.add_argument('--training_epochs', type=int, default=100,
                        help='input training epochs for training (default: 501)')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='input batch size for training (default: 100)')

    parser.add_argument('--init_model_with_bias', type=int, default=0,
                        help='init model with bias as 0 or constant positive value')
    parser.add_argument('--random_seed', type=int, default=1,
                        help='input random seed for training (default: 1)')

    parser.add_argument('--layer_unit_count_list', nargs="*", type=int, default=[784, 256, 100, 10])
    parser.add_argument('--num_valid', type=int, default=10000)
    parser.add_argument('--num_of_batches_for_mean', type=int, default=5, help='the number of batches that used for calculating the mean of node output')
    parser.add_argument('--lambda_balance', nargs = "*", type=float, default=[0], help='to balance the node stable loss and cross entropy loss')
    parser.add_argument('--num_of_layers_with_nodeStableLoss', type=int, default=1, help='the number of layers for backward node stable loss, counting from the last one')
    parser.add_argument('--model_name', type=str, default= "ResNet_valid")
    
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
    
    
    model = Net(args.layer_unit_count_list)
    num_layer = len(args.layer_unit_count_list) - 1
    model.start_count = num_layer - args.num_of_layers_with_nodeStableLoss
   
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

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)  
    scheduler = None
    if args.is_schedule:  
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, patience=args.patience, \
            threshold=args.threshold, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)

    validset, trainset = torch.utils.data.random_split(trainset, [args.num_valid, 60000 - args.num_valid])
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=2)

    record_dict = {}
    # train and test the created model
    max_valid_acc = -1

    for epoch in range(training_epochs):
        if epoch == 0:
            test_loss, test_correct, test_accuracy = test(model, device, test_loader, criterion)
            print('Testing set: Average loss: {:.4f}, Accuracy: ({:.4f}%)'.format(test_loss, test_accuracy))
            print('Test initialization end')
        train_loss, node_stable_loss, cross_entropy_loss, train_acc = train_cuda(model, device, train_loader, optimizer, criterion, batch_size, args, y_log, target_log)
        print('Train_epoch\tcross_loss\tstable_loss')
        print('{}\t{:.6f}\t{:.6f}'.format(epoch, cross_entropy_loss, node_stable_loss))

        valid_loss, valid_correct, valid_accuracy= test(model, device, valid_loader, criterion)
        print("Test_epoch\tvalid_loss\tvalid_accuracy")
        print("{:.4f}\t{:.4f}".format(valid_loss, valid_accuracy))
        is_test = max_valid_acc <= valid_accuracy
        max_valid_acc = max(max_valid_acc, valid_accuracy)
        
        if is_test:
            test_loss, test_correct, test_accuracy = test(model, device, test_loader, criterion)
            print('Test_epoch\tTest_loss\tTest_accuracy')
            print('{}\t{:.4f}\t{:.4f}'.format(epoch, test_loss, test_accuracy))


if __name__ == '__main__':
    main()



