#!/bin/bash

python train.py --dataset cifar10 --model resnet18 --data_augmentation --cutout --length 16 --lambda_balance 0.05
