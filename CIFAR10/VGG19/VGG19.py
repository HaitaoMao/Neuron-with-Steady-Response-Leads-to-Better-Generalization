import torch
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        """CNN Builder."""
        super(Net, self).__init__()

        self.fc = torch.nn.ModuleList()
        self.cc = torch.nn.ModuleList()
        self.intermedia_y = []


        self.ac = nn.ReLU(inplace = True)
        self.start_count = 0 

        
        # VGG Net
        
        self.cc = nn.ModuleList( [
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace = True)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace = True),
                nn.MaxPool2d((2,2)),
                nn.Dropout(0.2)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias = False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace = True)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias = False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace = True),
                nn.MaxPool2d((2,2)), 
                nn.Dropout(0.3)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias = False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace = True)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias = False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace = True)
            ),   
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias = False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace = True)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias = False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace = True),
                nn.MaxPool2d((2,2)), 
                nn.Dropout(0.4)
            ),   
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias = False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace = True)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias = False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace = True)
            ),   
            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias = False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace = True)
            ),   
            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias = False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace = True),
                nn.MaxPool2d((2,2)), 
                nn.Dropout(0.4)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias = False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace = True)
            ),   
            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias = False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace = True)
            ),   
            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias = False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace = True)
            ),   
            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias = False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace = True)
            )   
        ])


        self.fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2048, 4096),
                nn.ReLU(inplace = True),
                nn.Dropout(0.5),
            ),
            nn.Sequential(
                nn.Linear(4096, 4096),
                nn.ReLU(inplace = True),
                nn.Dropout(0.5),
            ),
            nn.Sequential(
                nn.Linear(4096, 10)
            )
        ])
        
        

        
    def _make_res_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    
    def forward(self, x):
        self.intermedia_y = []
        cnt = 0
        for i in range(len(self.cc)):
            x = self.cc[i](x)
            if cnt >= self.start_count:
                self.intermedia_y.append(x.view(x.size(0), -1))

            cnt += 1
            
        x = x.view(x.size(0), -1)
        for i in range(len(self.fc)):
            x = self.fc[i](x)
            if cnt >= self.start_count:
                self.intermedia_y.append(x.view(x.size(0), -1))

            cnt += 1
            
        return self.intermedia_y[-1]

class Resnet_basic_block(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(Resnet_basic_block, self).__init__()
        DROPOUT = 0.1

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(DROPOUT)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(DROPOUT)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes),
                nn.Dropout(DROPOUT)
            )

    def forward(self, x):
        out = F.relu(self.dropout(self.bn1(self.conv1(x))))
        out = self.dropout(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out        

