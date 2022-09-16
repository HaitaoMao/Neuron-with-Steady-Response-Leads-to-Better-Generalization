import torch
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self, layer_unit_count_list):
        super(Net, self).__init__()

        self.fc = torch.nn.ModuleList()
        for i in range(len(layer_unit_count_list) - 2):
            self.fc.append(
                nn.Sequential(
                    nn.Linear(layer_unit_count_list[i], layer_unit_count_list[i + 1]),
                    nn.ReLU(inplace = True)
                )
            )
        
        self.fc.append(
            nn.Sequential(
                nn.Linear(layer_unit_count_list[i + 1], layer_unit_count_list[i + 2])
            )
        )

        self.intermedia_y = []
        self.start_count = 0 

    def forward(self, x):
        self.intermedia_y = []
        # reshape input
        x = x.view(x.size(0), -1)
        cnt = 0
        for i in range(len(self.fc)):
            x = self.fc[i](x)
            if cnt >= self.start_count:
                self.intermedia_y.append(x)
            
            cnt += 1
        
        return self.intermedia_y[-1]
