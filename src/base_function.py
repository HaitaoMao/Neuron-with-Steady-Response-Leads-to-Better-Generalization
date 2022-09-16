import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
import numpy as np
import matplotlib.pyplot as plt

def weights_init_apply_with_bias(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)
        with torch.no_grad():
            std = m.weight.std(dim = 1).mean()
            print(std)
        torch.nn.init.constant_(m.bias, 0.01)  # 0.01

def weights_init_apply_without_bias(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight.data)
        # torch.nn.init.constant_(m.bias, 0.00)
        
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias, 0.00)

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            model = model.to(device)
            prediction = model(data)
            test_loss += criterion(prediction, target).item()
            pred = prediction.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, correct, accuracy
