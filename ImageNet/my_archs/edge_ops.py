import torch
import torch.nn as nn
import pdb
from torch.nn.functional import conv2d, pad


class SobelFilter(nn.Module):

    def __init__(self):
        super(SobelFilter, self).__init__()
        kernel_x = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        kernel_y = torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        self.register_buffer('kernel_x', kernel_x.view(1, 1, 3, 3))
        self.register_buffer('kernel_y', kernel_y.view(1, 1, 3, 3))

    def forward(self, x):
        n, c, h, w = x.shape
        x = x.view(n * c, 1, h, w)
        x = pad(x, [1, 1, 1, 1], mode='constant')
        out_x = conv2d(x, self.kernel_x, None, 1, 0, 1, 1)
        out_y = conv2d(x, self.kernel_y, None, 1, 0, 1, 1)
        out_x = out_x.view(n, c, h, w)
        out_y = out_y.view(n, c, h, w)
        return out_x, out_y


class GradFilter(nn.Module):

    def __init__(self):
        super(GradFilter, self).__init__()
        kernel_x = torch.FloatTensor([[0, 0, 0], [1, 0, -1], [0, 0, 0]])
        kernel_y = torch.FloatTensor([[0, 1, 0], [0, 0, 0], [0, -1, 0]])
        self.register_buffer('kernel_x', kernel_x.view(1, 1, 3, 3))
        self.register_buffer('kernel_y', kernel_y.view(1, 1, 3, 3))

    def forward(self, x):
        n, c, h, w = x.shape
        x = x.view(n * c, 1, h, w)
        x = pad(x, [1, 1, 1, 1], mode='constant')
        out_x = conv2d(x, self.kernel_x, None, 1, 0, 1, 1)
        out_y = conv2d(x, self.kernel_y, None, 1, 0, 1, 1)
        out_x = out_x.view(n, c, h, w)
        out_y = out_y.view(n, c, h, w)
        return out_x, out_y


class PrewittFilter(nn.Module):

    def __init__(self):
        super(PrewittFilter, self).__init__()
        kernel_x = torch.FloatTensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = torch.FloatTensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        self.register_buffer('kernel_x', kernel_x.view(1, 1, 3, 3))
        self.register_buffer('kernel_y', kernel_y.view(1, 1, 3, 3))

    def forward(self, x):
        n, c, h, w = x.shape
        x = x.view(n * c, 1, h, w)
        x = pad(x, [1, 1, 1, 1], mode='constant')
        out_x = conv2d(x, self.kernel_x, None, 1, 0, 1, 1)
        out_y = conv2d(x, self.kernel_y, None, 1, 0, 1, 1)
        out_x = out_x.view(n, c, h, w)
        out_y = out_y.view(n, c, h, w)
        return out_x, out_y


class AverageFilter(nn.Module):

    def __init__(self):
        super(AverageFilter, self).__init__()
        kernel_x = torch.FloatTensor([[1/9.0, 1/9.0, 1/9.0], [1/9.0, 1/9.0, 1/9.0], [1/9.0, 1/9.0, 1/9.0]])
        #kernel_y = torch.FloatTensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        self.register_buffer('kernel_x', kernel_x.view(1, 1, 3, 3))
        #self.register_buffer('kernel_y', kernel_y.view(1, 1, 3, 3))

    def forward(self, x):
        n, c, h, w = x.shape
        x = x.view(n * c, 1, h, w)
        x = pad(x, [1, 1, 1, 1], mode='constant')
        out_x = conv2d(x, self.kernel_x, None, 1, 0, 1, 1)
        #out_y = conv2d(x, self.kernel_y, None, 1, 0, 1, 1)
        out_x = out_x.view(n, c, h, w)
        #out_y = out_y.view(n, c, h, w)
        return out_x


class Average2Filter(nn.Module):#x,y

    def __init__(self):
        super(Average2Filter, self).__init__()
        kernel_x = torch.FloatTensor([[1/18.0, 1/18.0, 1/18.0], [1/18.0, 1/18.0, 1/18.0], [1/18.0, 1/18.0, 1/18.0]])
        kernel_y = torch.FloatTensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        self.register_buffer('kernel_x', kernel_x.view(1, 1, 3, 3))
        self.register_buffer('kernel_y', kernel_y.view(1, 1, 3, 3))

    def forward(self, x):
        n, c, h, w = x.shape
        x = x.view(n * c, 1, h, w)
        x = pad(x, [1, 1, 1, 1], mode='constant')
        out_x = conv2d(x, self.kernel_x, None, 1, 0, 1, 1)
        out_y = conv2d(x, self.kernel_y, None, 1, 0, 1, 1)
        out_x = out_x.view(n, c, h, w)
        out_y = out_y.view(n, c, h, w)
        return out_x,out_y


class SobelFilter_learnable(nn.Module):

    def __init__(self):
        super(SobelFilter_learnable, self).__init__()
        kernel_x = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        kernel_y = torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        #self.register_buffer('kernel_x', kernel_x.view(1, 1, 3, 3))
        #self.register_buffer('kernel_y', kernel_y.view(1, 1, 3, 3))
        self.param_x = nn.Parameter(kernel_x.unsqueeze(0).unsqueeze(0),requires_grad=True)
        self.param_y = nn.Parameter(kernel_y.unsqueeze(0).unsqueeze(0),requires_grad=True)

    def forward(self, x):
        n, c, h, w = x.shape
        x = x.view(n * c, 1, h, w)
        x = pad(x, [1, 1, 1, 1], mode='constant')
        #print('self.param_x',self.param_x)
        #print('self.param_y', self.param_y)
        out_x = conv2d(x, self.param_x, None, 1, 0, 1, 1)
        out_y = conv2d(x, self.param_y, None, 1, 0, 1, 1)
        out_x = out_x.view(n, c, h, w)
        out_y = out_y.view(n, c, h, w)
        return out_x, out_y


class RobertFilter(nn.Module):

    def __init__(self):
        super(RobertFilter, self).__init__()
        kernel_x = torch.FloatTensor([[-1, 0], [0, 1]])
        kernel_y = torch.FloatTensor([[0, -1], [1, 0]])
        self.register_buffer('kernel_x', kernel_x.view(1, 1, 2, 2))
        self.register_buffer('kernel_y', kernel_y.view(1, 1, 2, 2))

    def forward(self, x):
        n, c, h, w = x.shape
        x = x.view(n * c, 1, h, w)
        x1 = pad(x, [1, 0, 1, 0], mode='constant') #pad up and left
        x2 = pad(x, [0, 1, 1, 0], mode='constant')  # pad up and right
        out_x = conv2d(x1, self.kernel_x, None, 1, 0, 1, 1)
        out_y = conv2d(x2, self.kernel_y, None, 1, 0, 1, 1)
        out_x = out_x.view(n, c, h, w)
        out_y = out_y.view(n, c, h, w)
        return out_x, out_y