import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum


def train_cuda(model, device, train_loader, optimizer, criterion, batch_size, args, y_log, target_log):
    model.train()
    node_stable_loss, cross_entropy_loss, total_loss, correct = 0.0, 0.0, 0.0, 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        model = model.to(device)
        optimizer.zero_grad()
        prediction = model(data)
        
        layer_list_for_backward = range(args.num_of_layers_with_nodeStableLoss)
        # Firstly record the output of the each node of each batch, maintain a fixed-length queue,list of [batch_size, num_of_node]
    
        if (len(y_log[0])<args.num_of_batches_for_mean):
            target_log.append(target.clone().detach())
            for layer_idx in layer_list_for_backward:
                y_log[layer_idx].append(model.intermedia_y[layer_idx].clone().detach())

        elif (len(y_log[0])==args.num_of_batches_for_mean):
            target_log.pop(0)
            target_log.append(target.clone().detach())
            for layer_idx in layer_list_for_backward:
                y_log[layer_idx].pop(0)
                y_log[layer_idx].append(model.intermedia_y[layer_idx].clone().detach())
        else:
            print('the length of queue is out of expectation')
        
        torch.cuda.empty_cache()
        node_stable_loss_sum = torch.zeros([1]).cuda()

        # concatenate the arrays in the list and change to tensor for scatter_mean function,[num of images used for mean, node]
        target_log_tensor = torch.stack(target_log).reshape(-1,1)
        for layer_idx in layer_list_for_backward:
            # print(layer_idx)
            y_log_tensor = torch.stack(y_log[layer_idx]).reshape((-1,y_log[layer_idx][0].shape[1]))
            # use scatter_mean to calculate the y_log_tensor_mean,[class,node]
            if layer_idx == args.num_of_layers_with_nodeStableLoss - 1:
                y_log_tensor_mean = scatter_mean(y_log_tensor, target_log_tensor, dim=0)
                # scatter the calculated mean value of 10 class to each image in the batch,[img in one batch,node]
                y_log_tensor_mean_scattered = torch.index_select(y_log_tensor_mean, 0, target)
                square_difference_the_class = (model.intermedia_y[layer_idx] - y_log_tensor_mean_scattered.to(device)) **2
                node_stable_loss_sum += (torch.mean(square_difference_the_class) * args.lambda_balance[layer_idx])
            else:
                count_scatter = scatter_sum((y_log_tensor != 0).int(), target_log_tensor, dim=0).detach()
                count_scatter = torch.where(count_scatter == 0, torch.full_like(count_scatter, 1), count_scatter)
                y_log_tensor_mean = scatter_sum(y_log_tensor, target_log_tensor, dim=0) / count_scatter
                y_log_tensor_mean_scattered = torch.index_select(y_log_tensor_mean, 0, target)
                mask = torch.abs(model.intermedia_y[layer_idx]) > 1e-5
                square_difference_the_class = (model.intermedia_y[layer_idx] - y_log_tensor_mean_scattered[layer_idx].to(device)) **2
                square_difference_the_class = torch.masked_select(square_difference_the_class, mask)
                node_stable_loss_sum += (torch.mean(square_difference_the_class) * args.lambda_balance[layer_idx])
            
        cross_entropy_loss_item = criterion(prediction, target)
        loss = cross_entropy_loss_item + node_stable_loss_sum
        loss.backward()        
        if args.is_clip:
            for p in model.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, -args.clip_value, args.clip_value))

        optimizer.step()
            
        node_stable_loss += node_stable_loss_sum.item()
        cross_entropy_loss += cross_entropy_loss_item.item()
        total_loss += loss.item()
        pred = prediction.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = 100. * correct / len(train_loader.dataset)                
    node_stable_loss /= len(train_loader.dataset)
    cross_entropy_loss /= len(train_loader.dataset)
    total_loss /= len(train_loader.dataset)

    return total_loss,node_stable_loss,cross_entropy_loss, accuracy
