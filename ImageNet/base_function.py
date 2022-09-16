import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
import numpy as np
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
def evaluate(model, device, loader, args, num_class = 10):
    model.eval()
    
    intermedia_range = range(args.num_of_layers_with_nodeStableLoss)
    
    with torch.no_grad():
        intermedia_y_mean, incorrect_data, incorrect_target, intermedia_loss = [], [], [], []
        intermedia_y_incorrect_distance,intermedia_y_correct_distance, intermedia_y_center, center_distance = [], [], [], []
        # num_layer, num_class
        target_count = torch.zeros(num_class).to(device)
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):        
                data, target = data.to(device), target.to(device)
                model = model.to(device)
                prediction = model(data)
                if batch_idx == 0:
                    # init record lists
                    for i in intermedia_range:
                        intermedia_y_mean.append(torch.zeros(num_class, model.intermedia_y[i].size(-1)).to(device))
                        intermedia_y_center.append(torch.zeros(model.intermedia_y[i].size(-1)).to(device))
                        intermedia_y_correct_distance.append([0.0 for _ in range(num_class)])
                        intermedia_y_incorrect_distance.append([0.0 for _ in range(num_class)])
                        center_distance.append(0.0)
                        intermedia_loss.append(0.0)
    
                        
                    intermedia_y_correct_distance = torch.tensor(intermedia_y_correct_distance).to(device)
                    intermedia_y_incorrect_distance = torch.tensor(intermedia_y_incorrect_distance).to(device)
                    # center_distance = torch.tensor(center_distance).to(device)
                    intermedia_loss = torch.tensor(intermedia_loss).to(device)

                pred = prediction.argmax(dim=1, keepdim=True) 
                #record for incorrect
                incorrect_idx = torch.squeeze(~pred.eq(target.view_as(pred)))
                target_count += torch.bincount(target,minlength=num_class)
                incorrect_data.append(data[incorrect_idx])
                incorrect_target.append(target[incorrect_idx])
                for i in intermedia_range:
                    intermedia_y_mean[i] += scatter_sum(model.intermedia_y[i], target, dim = 0)
                    intermedia_y_center[i] += torch.sum(model.intermedia_y[i], dim = 0)
            
            # print(torch.sum(target_count))
            num_dataset = torch.sum(target_count).item()
            for i in intermedia_range:
                intermedia_y_mean[i] /= target_count.view(num_class, 1)
                intermedia_y_center[i] /= num_dataset
            
            incorrect_data = torch.cat(incorrect_data, dim = 0)
            incorrect_target = torch.cat(incorrect_target, dim = 0)
            incorrect_target_count = torch.bincount(incorrect_target,minlength=num_class)
            # print("incorrect_data", incorrect_data.size(0))
            # print("incorrect_target", incorrect_target.size(0))
            # print(torch.sum(incorrect_target_count))

            incorrect_num = incorrect_data.size(0)    

            # calculate center
            for i in range(num_class):
                mask = (incorrect_target == i)
                incorrect_data_class = incorrect_data[mask]
                if incorrect_data_class.size(0) == 0:
                    continue
                model(incorrect_data_class)
                
                for j in intermedia_range:
                    intermedia_y_incorrect_distance[j][i] += torch.mean(torch.sqrt(torch.square(model.intermedia_y[j] - intermedia_y_mean[j][i]))).item()
            
            torch.cuda.empty_cache()
            # calculate distance on each batch
            for batch_idx, (data, target) in enumerate(loader):        
                data, target = data.to(device), target.to(device)
                model = model.to(device)
                prediction = model(data)
                
                for j in intermedia_range:
                    center_distance[j] += torch.mean(torch.sqrt(torch.square(model.intermedia_y[j] - intermedia_y_center[j]))).item()
                    # temp = torch.index_select(intermedia_y_mean[j], 0, target)
                    # intermedia_y_correct_distance[j] += torch.sum(torch.square(model.intermedia_y[j] - temp)).item()
                    for i in range(num_class):
                        mask = (target == i)
                        intermedia_y_correct_distance[j][i] += torch.sum(torch.mean(torch.sqrt(torch.square(model.intermedia_y[j][mask] - intermedia_y_mean[j][i]))), dim = -1).item()
                # print(intermedia_y_correct_distance[0][0])

            for i in intermedia_range:
                intermedia_loss[i] += torch.mean(intermedia_y_correct_distance[i])
                intermedia_y_correct_distance[i] -=  torch.multiply(intermedia_y_incorrect_distance[i], incorrect_target_count)
                # print(target_count - incorrect_target_count)
                intermedia_y_correct_distance[i] /= (target_count - incorrect_target_count)

                center_distance[j] /= (batch_idx + 1)
                
                
            # print(intermedia_y_correct_distance, intermedia_y_incorrect_distance, center_distance, intermedia_loss)
            return intermedia_y_correct_distance.cpu().numpy().tolist(), intermedia_y_incorrect_distance.cpu().numpy().tolist(), center_distance, intermedia_loss.cpu().numpy().tolist()

                        