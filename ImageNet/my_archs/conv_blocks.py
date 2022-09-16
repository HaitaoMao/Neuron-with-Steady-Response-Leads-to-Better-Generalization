import torch
import torch.nn as nn
import numbers
from .edge_ops import SobelFilter, GradFilter,PrewittFilter,AverageFilter,Average2Filter,SobelFilter_learnable,RobertFilter
#from torch.nn.functional import conv2d, sigmoid

class VanillaConv(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = False,
            padding_mode: str = 'zeros'
    ):
        super(VanillaConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups,
                              bias, padding_mode)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.bn(x)
        return x


class DiffConvV0(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = False,
            padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        super(DiffConvV0, self).__init__()
        self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation, groups,
                                       bias, padding_mode)
        self.identity_bn = nn.BatchNorm2d(out_channels)
        self.diff_x_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                     stride, padding, dilation, groups,
                                     bias, padding_mode)
        self.diff_x_bn = nn.BatchNorm2d(out_channels)
        self.diff_y_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                     stride, padding, dilation, groups,
                                     bias, padding_mode)
        self.diff_y_bn = nn.BatchNorm2d(out_channels)

        self.sobel = SobelFilter()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.identity_conv(x)
        identity = self.identity_bn(identity)

        diff_x, diff_y = self.sobel(x)
        diff_x = self.diff_x_conv(diff_x)
        diff_x = self.diff_x_bn(diff_x)

        diff_y = self.diff_y_conv(diff_y)
        diff_y = self.diff_y_bn(diff_y)
        return identity + diff_x + diff_y


class DiffConvV1(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = False,
            padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        super(DiffConvV1, self).__init__()
        self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation, groups,
                                       bias, padding_mode)
        self.identity_bn = nn.BatchNorm2d(out_channels)
        self.diff_x_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                     stride, padding, dilation, groups,
                                     bias, padding_mode)
        self.diff_x_bn = nn.BatchNorm2d(out_channels)
        self.diff_y_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                     stride, padding, dilation, groups,
                                     bias, padding_mode)
        self.diff_y_bn = nn.BatchNorm2d(out_channels)

        self.sobel = GradFilter()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.identity_conv(x)
        identity = self.identity_bn(identity)

        diff_x, diff_y = self.sobel(x)
        diff_x = self.diff_x_conv(diff_x)
        diff_x = self.diff_x_bn(diff_x)

        diff_y = self.diff_y_conv(diff_y)
        diff_y = self.diff_y_bn(diff_y)
        return identity + diff_x + diff_y


class DiffConvV2(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = False,
            padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        super(DiffConvV2, self).__init__()
        self.is_1x1 = True
        if isinstance(kernel_size, numbers.Number):
            if kernel_size > 1:
                self.is_1x1 = False
        else:
            for _ in kernel_size:
                if _ > 1:
                    self.is_1x1 = False
        print("Check conv type: is1x1 {},{}".format(self.is_1x1, kernel_size))
        self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation, groups,
                                       bias, padding_mode)
        self.identity_bn = nn.BatchNorm2d(out_channels)

        if not self.is_1x1:
            self.diff_x_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                         stride, padding, dilation, groups,
                                         bias, padding_mode)
            self.diff_x_bn = nn.BatchNorm2d(out_channels)
            self.diff_y_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                         stride, padding, dilation, groups,
                                         bias, padding_mode)
            self.diff_y_bn = nn.BatchNorm2d(out_channels)

            self.sobel = SobelFilter()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.identity_conv(x)
        identity = self.identity_bn(identity)
        if not self.is_1x1:
            diff_x, diff_y = self.sobel(x)
            diff_x = self.diff_x_conv(diff_x)
            diff_x = self.diff_x_bn(diff_x)

            diff_y = self.diff_y_conv(diff_y)
            diff_y = self.diff_y_bn(diff_y)
            return identity + diff_x + diff_y
        else:
            return identity


class DiffConvV3(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = False,
            padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        super(DiffConvV3, self).__init__()
        self.is_1x1 = True
        if isinstance(kernel_size, numbers.Number):
            if kernel_size > 1:
                self.is_1x1 = False
        else:
            for _ in kernel_size:
                if _ > 1:
                    self.is_1x1 = False
        print("Check conv type: is1x1 {},{}".format(self.is_1x1, kernel_size))
        self.identity_conv = nn.Conv2d(in_channels, out_channels, 1,
                                       stride, 0, dilation, groups,
                                       bias, padding_mode)
        self.identity_bn = nn.BatchNorm2d(out_channels)

        if not self.is_1x1:
            self.diff_x_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                         stride, padding, dilation, groups,
                                         bias, padding_mode)
            self.diff_x_bn = nn.BatchNorm2d(out_channels)
            self.diff_y_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                         stride, padding, dilation, groups,
                                         bias, padding_mode)
            self.diff_y_bn = nn.BatchNorm2d(out_channels)

            self.sobel = SobelFilter()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.identity_conv(x)
        identity = self.identity_bn(identity)
        if not self.is_1x1:
            diff_x, diff_y = self.sobel(x)
            diff_x = self.diff_x_conv(diff_x)
            diff_x = self.diff_x_bn(diff_x)

            diff_y = self.diff_y_conv(diff_y)
            diff_y = self.diff_y_bn(diff_y)
            return identity + diff_x + diff_y
        else:
            return identity


class DiffConvV4(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = False,
            padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        super(DiffConvV4, self).__init__()
        self.is_1x1 = True
        if isinstance(kernel_size, numbers.Number):
            if kernel_size > 1:
                self.is_1x1 = False
        else:
            for _ in kernel_size:
                if _ > 1:
                    self.is_1x1 = False
        print("Check conv type: is1x1 {},{}".format(self.is_1x1, kernel_size))
        self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation, groups,
                                       bias, padding_mode)
        self.identity_bn = nn.BatchNorm2d(out_channels)

        if not self.is_1x1:
            self.diff_x_conv = nn.Conv2d(in_channels, out_channels, 1,
                                         stride, 0, dilation, groups,
                                         bias, padding_mode)
            self.diff_x_bn = nn.BatchNorm2d(out_channels)
            self.diff_y_conv = nn.Conv2d(in_channels, out_channels, 1,
                                         stride, 0, dilation, groups,
                                         bias, padding_mode)
            self.diff_y_bn = nn.BatchNorm2d(out_channels)

            self.sobel = SobelFilter()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.identity_conv(x)
        identity = self.identity_bn(identity)
        if not self.is_1x1:
            diff_x, diff_y = self.sobel(x)
            diff_x = self.diff_x_conv(diff_x)
            diff_x = self.diff_x_bn(diff_x)

            diff_y = self.diff_y_conv(diff_y)
            diff_y = self.diff_y_bn(diff_y)
            return identity + diff_x + diff_y
        else:
            return identity


class DiffConvV5(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = False,
            padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        super(DiffConvV5, self).__init__()
        self.is_1x1 = True
        if isinstance(kernel_size, numbers.Number):
            if kernel_size > 1:
                self.is_1x1 = False
        else:
            for _ in kernel_size:
                if _ > 1:
                    self.is_1x1 = False
        print("Check conv type: is1x1 {},{}".format(self.is_1x1, kernel_size))
        if self.is_1x1:
            self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                           stride, padding, dilation, groups,
                                           bias, padding_mode)
            self.identity_bn = nn.BatchNorm2d(out_channels)
        else:
            self.diff_x_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                         stride, padding, dilation, groups,
                                         bias, padding_mode)
            self.diff_x_bn = nn.BatchNorm2d(out_channels)
            self.diff_y_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                         stride, padding, dilation, groups,
                                         bias, padding_mode)
            self.diff_y_bn = nn.BatchNorm2d(out_channels)

            self.sobel = SobelFilter()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_1x1:
            identity = self.identity_conv(x)
            identity = self.identity_bn(identity)
            return identity
        else:
            diff_x, diff_y = self.sobel(x)
            diff_x = self.diff_x_conv(diff_x)
            diff_x = self.diff_x_bn(diff_x)

            diff_y = self.diff_y_conv(diff_y)
            diff_y = self.diff_y_bn(diff_y)
            return diff_x + diff_y


class DiffConvV6(DiffConvV2):

    def __init__(self, *args, **kwargs):
        super(DiffConvV6, self).__init__(*args, **kwargs)
        self.sobel = GradFilter()


class MultiV2(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = False,
            padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        super(MultiV2, self).__init__()
        self.is_1x1 = True
        if isinstance(kernel_size, numbers.Number):
            if kernel_size > 1:
                self.is_1x1 = False
        else:
            for _ in kernel_size:
                if _ > 1:
                    self.is_1x1 = False
        print("Check conv type: is1x1 {},{}".format(self.is_1x1, kernel_size))
        self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation, groups,
                                       bias, padding_mode)
        self.identity_bn = nn.BatchNorm2d(out_channels)

        if not self.is_1x1:
            self.diff_x_conv = nn.Conv2d(in_channels, out_channels, 1,
                                         stride, 0, dilation, groups,
                                         bias, padding_mode)
            self.diff_x_bn = nn.BatchNorm2d(out_channels)
            self.diff_y_conv = nn.Conv2d(in_channels, out_channels, 1,
                                         stride, 0, dilation, groups,
                                         bias, padding_mode)
            self.diff_y_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.identity_conv(x)
        identity = self.identity_bn(identity)
        if not self.is_1x1:
            diff_x = self.diff_x_conv(x)
            diff_x = self.diff_x_bn(diff_x)

            diff_y = self.diff_y_conv(x)
            diff_y = self.diff_y_bn(diff_y)
            return identity + diff_x + diff_y
        else:
            return identity


class DiffConvV7(nn.Module):  #Prewitt

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = False,
            padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        super(DiffConvV7, self).__init__()
        self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation, groups,
                                       bias, padding_mode)
        self.identity_bn = nn.BatchNorm2d(out_channels)
        self.diff_x_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                     stride, padding, dilation, groups,
                                     bias, padding_mode)
        self.diff_x_bn = nn.BatchNorm2d(out_channels)
        self.diff_y_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                     stride, padding, dilation, groups,
                                     bias, padding_mode)
        self.diff_y_bn = nn.BatchNorm2d(out_channels)

        self.sobel = PrewittFilter()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.identity_conv(x)
        identity = self.identity_bn(identity)

        diff_x, diff_y = self.sobel(x)
        diff_x = self.diff_x_conv(diff_x)
        diff_x = self.diff_x_bn(diff_x)

        diff_y = self.diff_y_conv(diff_y)
        diff_y = self.diff_y_bn(diff_y)
        return identity + diff_x + diff_y


class DiffConvV8(nn.Module):  #Average single

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = False,
            padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        super(DiffConvV8, self).__init__()
        self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation, groups,
                                       bias, padding_mode)
        self.identity_bn = nn.BatchNorm2d(out_channels)
        self.diff_x_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                     stride, padding, dilation, groups,
                                     bias, padding_mode)
        self.diff_x_bn = nn.BatchNorm2d(out_channels)
        #self.diff_y_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
        #                             stride, padding, dilation, groups,
        #                             bias, padding_mode)
        #self.diff_y_bn = nn.BatchNorm2d(out_channels)

        self.sobel = AverageFilter()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.identity_conv(x)
        identity = self.identity_bn(identity)

        diff_x = self.sobel(x)
        diff_x = self.diff_x_conv(diff_x)
        diff_x = self.diff_x_bn(diff_x)

        #diff_y = self.diff_y_conv(diff_y)
        #diff_y = self.diff_y_bn(diff_y)
        return identity + diff_x


class DiffConvV9(nn.Module):  #Average 2

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = False,
            padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        super(DiffConvV9, self).__init__()
        self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation, groups,
                                       bias, padding_mode)
        self.identity_bn = nn.BatchNorm2d(out_channels)
        self.diff_x_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                     stride, padding, dilation, groups,
                                     bias, padding_mode)
        self.diff_x_bn = nn.BatchNorm2d(out_channels)
        self.diff_y_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups,
                                    bias, padding_mode)
        self.diff_y_bn = nn.BatchNorm2d(out_channels)

        self.sobel = Average2Filter()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.identity_conv(x)
        identity = self.identity_bn(identity)

        diff_x, diff_y = self.sobel(x)
        diff_x = self.diff_x_conv(diff_x)
        diff_x = self.diff_x_bn(diff_x)

        diff_y = self.diff_y_conv(diff_y)
        diff_y = self.diff_y_bn(diff_y)
        return identity + diff_x + diff_y


class DiffConvV10(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = False,
            padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        super(DiffConvV10, self).__init__()
        self.is_1x1 = True
        if isinstance(kernel_size, numbers.Number):
            if kernel_size > 1:
                self.is_1x1 = False
        else:
            for _ in kernel_size:
                if _ > 1:
                    self.is_1x1 = False
        print("Check conv type: is1x1 {},{}".format(self.is_1x1, kernel_size))
        self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation, groups,
                                       bias, padding_mode)
        self.identity_bn = nn.BatchNorm2d(out_channels)

        if not self.is_1x1:
            self.diff_x_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                         stride, padding, dilation, groups,
                                         bias, padding_mode)
            self.diff_x_bn = nn.BatchNorm2d(out_channels)
            #self.diff_y_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            #                             stride, padding, dilation, groups,
            #                             bias, padding_mode)
            #self.diff_y_bn = nn.BatchNorm2d(out_channels)

            self.sobel = AverageFilter()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.identity_conv(x)
        identity = self.identity_bn(identity)
        if not self.is_1x1:
            diff_x = self.sobel(x)
            diff_x = self.diff_x_conv(diff_x)
            diff_x = self.diff_x_bn(diff_x)

            #diff_y = self.diff_y_conv(diff_y)
            #diff_y = self.diff_y_bn(diff_y)
            return identity + diff_x
        else:
            return identity


class DiffConvV11(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = False,
            padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        super(DiffConvV11, self).__init__()
        self.is_1x1 = True
        if isinstance(kernel_size, numbers.Number):
            if kernel_size > 1:
                self.is_1x1 = False
        else:
            for _ in kernel_size:
                if _ > 1:
                    self.is_1x1 = False
        print("Check conv type: is1x1 {},{}".format(self.is_1x1, kernel_size))
        self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation, groups,
                                       bias, padding_mode)
        self.identity_bn = nn.BatchNorm2d(out_channels)

        if not self.is_1x1:
            self.diff_x_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                         stride, padding, dilation, groups,
                                         bias, padding_mode)
            self.diff_x_bn = nn.BatchNorm2d(out_channels)
            self.diff_y_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                         stride, padding, dilation, groups,
                                         bias, padding_mode)
            self.diff_y_bn = nn.BatchNorm2d(out_channels)

            self.sobel = Average2Filter()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.identity_conv(x)
        identity = self.identity_bn(identity)
        if not self.is_1x1:
            diff_x, diff_y = self.sobel(x)
            diff_x = self.diff_x_conv(diff_x)
            diff_x = self.diff_x_bn(diff_x)

            diff_y = self.diff_y_conv(diff_y)
            diff_y = self.diff_y_bn(diff_y)
            return identity + diff_x + diff_y
        else:
            return identity


class SobelInitial(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = False,
            padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        super(SobelInitial, self).__init__()
        self.is_3x3 = False
        if isinstance(kernel_size, numbers.Number):
            if kernel_size == 3:
                self.is_3x3 = True
        else:
            for _ in kernel_size:
                if _ == 3:
                    self.is_3x3 = True
        print("Check conv type: is3x3 {},{}".format(self.is_3x3, kernel_size))
        self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation, groups,
                                       bias, padding_mode)
        self.identity_bn = nn.BatchNorm2d(out_channels)

        if self.is_3x3:
            kernel_x = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
            kernel_x=kernel_x.unsqueeze(0).unsqueeze(0).repeat(in_channels, out_channels, 1, 1)
            #print('kernel_x.size',kernel_x.size())
            kernel_y = torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            kernel_y = kernel_y.unsqueeze(0).unsqueeze(0).repeat(in_channels, out_channels, 1, 1)
            self.diff_x_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                         stride, padding, dilation, groups,
                                         bias, padding_mode)
            #print('shape of conv kernel',self.diff_x_conv.weight.data.size())
            #print("before sobel initialize:",self.diff_x_conv.weight)
            self.diff_x_conv.weight=torch.nn.Parameter(kernel_x)
            #print("After sobel initialize:", self.diff_x_conv.weight)
            self.diff_x_bn = nn.BatchNorm2d(out_channels)
            self.diff_y_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                         stride, padding, dilation, groups,
                                         bias, padding_mode)
            self.diff_y_conv.weight = torch.nn.Parameter(kernel_y)
            self.diff_y_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.identity_conv(x)
        identity = self.identity_bn(identity)
        if self.is_3x3:
            diff_x = self.diff_x_conv(x)
            diff_x = self.diff_x_bn(diff_x)

            diff_y = self.diff_y_conv(x)
            diff_y = self.diff_y_bn(diff_y)
            return identity + diff_x + diff_y
        else:
            return identity

class DiffConvV12(DiffConvV2):

    def __init__(self, *args, **kwargs):
        super(DiffConvV12, self).__init__(*args, **kwargs)
        self.sobel = SobelFilter_learnable()

class DiffConvV13(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = False,
            padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        super(DiffConvV13, self).__init__()
        self.is_3x3 = False
        if isinstance(kernel_size, numbers.Number):
            if kernel_size == 3:
                self.is_3x3 = True
        else:
            for _ in kernel_size:
                if _ == 3:
                    self.is_3x3 = True
        print("Check conv type: is3x3 {},{}".format(self.is_3x3, kernel_size))
        self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation, groups,
                                       bias, padding_mode)
        self.identity_bn = nn.BatchNorm2d(out_channels)

        if self.is_3x3:
            self.diff_x_conv = nn.Conv2d(in_channels, out_channels, 2,
                                         stride, 0, dilation, groups,
                                         bias, padding_mode)
            self.diff_x_bn = nn.BatchNorm2d(out_channels)
            self.diff_y_conv = nn.Conv2d(in_channels, out_channels, 2,
                                         stride, 0, dilation, groups,
                                         bias, padding_mode)
            #self.diff_y_conv.weight = torch.nn.Parameter(kernel_y)
            self.diff_y_bn = nn.BatchNorm2d(out_channels)
            self.sobel = RobertFilter()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.identity_conv(x)
        identity = self.identity_bn(identity)
        if self.is_3x3:
            diff_x, diff_y = self.sobel(x)
            diff_x= nn.functional.pad(diff_x, [1, 0, 1, 0], mode='constant')
            diff_x = self.diff_x_conv(diff_x)
            diff_x = self.diff_x_bn(diff_x)

            diff_y= nn.functional.pad(diff_y, [0, 1, 1, 0], mode='constant')
            diff_y = self.diff_y_conv(diff_y)
            diff_y = self.diff_y_bn(diff_y)
            return identity + diff_x + diff_y
        else:
            return identity

class DiffConvV13_(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = False,
            padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        super(DiffConvV13_, self).__init__()
        self.is_3x3 = False
        if isinstance(kernel_size, numbers.Number):
            if kernel_size == 3:
                self.is_3x3 = True
        else:
            for _ in kernel_size:
                if _ == 3:
                    self.is_3x3 = True
        print("Check conv type: is3x3 {},{}".format(self.is_3x3, kernel_size))
        self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation, groups,
                                       bias, padding_mode)
        self.identity_bn = nn.BatchNorm2d(out_channels)

        if self.is_3x3:
            self.diff_x_conv = nn.Conv2d(in_channels, out_channels, 2,
                                         stride, 0, dilation, groups,
                                         bias, padding_mode)
            self.diff_x_bn = nn.BatchNorm2d(out_channels)
            self.diff_y_conv = nn.Conv2d(in_channels, out_channels, 2,
                                         stride, 0, dilation, groups,
                                         bias, padding_mode)
            #self.diff_y_conv.weight = torch.nn.Parameter(kernel_y)
            self.diff_y_bn = nn.BatchNorm2d(out_channels)
            self.sobel = RobertFilter()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.identity_conv(x)
        identity = self.identity_bn(identity)
        if self.is_3x3:
            diff_x, diff_y = self.sobel(x)
            diff_x= nn.functional.pad(diff_x, [0, 1, 0, 1], mode='constant')
            diff_x = self.diff_x_conv(diff_x)
            diff_x = self.diff_x_bn(diff_x)

            diff_y= nn.functional.pad(diff_y, [1, 0, 0, 1], mode='constant')
            diff_y = self.diff_y_conv(diff_y)
            diff_y = self.diff_y_bn(diff_y)
            return identity + diff_x + diff_y
        else:
            return identity


class DiffConvV14(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = False,
            padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        super(DiffConvV14, self).__init__()
        self.is_3x3 = False
        self.is_7x7 = False
        if isinstance(kernel_size, numbers.Number):
            if kernel_size == 3:
                self.is_3x3 = True
            elif kernel_size == 7:
                self.is_7x7 = True
        else:
            for _ in kernel_size:
                if _ == 3:
                    self.is_3x3 = True
                elif _ == 7:
                    self.is_7x7 = True
        print("Check conv type: is3x3 {},{}".format(self.is_3x3, kernel_size))
        print("Check conv type: is7x7 {},{}".format(self.is_7x7, kernel_size))
        self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation, groups,
                                       bias, padding_mode)
        self.identity_bn = nn.BatchNorm2d(out_channels)

        if self.is_7x7:
            self.diff_x_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                         stride, padding, dilation, groups,
                                         bias, padding_mode)
            self.diff_x_bn = nn.BatchNorm2d(out_channels)
            self.diff_y_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                         stride, padding, dilation, groups,
                                         bias, padding_mode)
            self.diff_y_bn = nn.BatchNorm2d(out_channels)

            self.sobel = SobelFilter()
        elif self.is_3x3:
            self.diff_x_conv = nn.Conv2d(in_channels, out_channels, 2,
                                         stride, 0, dilation, groups,
                                         bias, padding_mode)
            self.diff_x_bn = nn.BatchNorm2d(out_channels)
            self.diff_y_conv = nn.Conv2d(in_channels, out_channels, 2,
                                         stride, 0, dilation, groups,
                                         bias, padding_mode)
            #self.diff_y_conv.weight = torch.nn.Parameter(kernel_y)
            self.diff_y_bn = nn.BatchNorm2d(out_channels)
            self.sobel = RobertFilter()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.identity_conv(x)
        identity = self.identity_bn(identity)
        if self.is_3x3:
            diff_x, diff_y = self.sobel(x)
            diff_x= nn.functional.pad(diff_x, [1, 0, 1, 0], mode='constant')
            diff_x = self.diff_x_conv(diff_x)
            diff_x = self.diff_x_bn(diff_x)

            diff_y= nn.functional.pad(diff_y, [0, 1, 1, 0], mode='constant')
            diff_y = self.diff_y_conv(diff_y)
            diff_y = self.diff_y_bn(diff_y)
            return identity + diff_x + diff_y
        elif self.is_7x7:
            diff_x, diff_y = self.sobel(x)
            diff_x = self.diff_x_conv(diff_x)
            diff_x = self.diff_x_bn(diff_x)

            diff_y = self.diff_y_conv(diff_y)
            diff_y = self.diff_y_bn(diff_y)
            return identity + diff_x + diff_y
        else:
            return identity


class DiffConvV14_(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = False,
            padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        super(DiffConvV14_, self).__init__()
        self.is_3x3 = False
        self.is_7x7 = False
        if isinstance(kernel_size, numbers.Number):
            if kernel_size == 3:
                self.is_3x3 = True
            elif kernel_size == 7:
                self.is_7x7 = True
        else:
            for _ in kernel_size:
                if _ == 3:
                    self.is_3x3 = True
                elif _ == 7:
                    self.is_7x7 = True
        print("Check conv type: is3x3 {},{}".format(self.is_3x3, kernel_size))
        print("Check conv type: is7x7 {},{}".format(self.is_7x7, kernel_size))
        self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation, groups,
                                       bias, padding_mode)
        self.identity_bn = nn.BatchNorm2d(out_channels)

        if self.is_7x7:
            self.diff_x_conv = nn.Conv2d(in_channels, out_channels, 5,
                                         stride, 2, dilation, groups,
                                         bias, padding_mode)
            #print("--------------------",stride, padding, dilation)
            self.diff_x_bn = nn.BatchNorm2d(out_channels)
            self.diff_y_conv = nn.Conv2d(in_channels, out_channels, 5,
                                         stride, 2, dilation, groups,
                                         bias, padding_mode)
            self.diff_y_bn = nn.BatchNorm2d(out_channels)

            self.sobel = SobelFilter()
        elif self.is_3x3:
            self.diff_x_conv = nn.Conv2d(in_channels, out_channels, 2,
                                         stride, 0, dilation, groups,
                                         bias, padding_mode)
            self.diff_x_bn = nn.BatchNorm2d(out_channels)
            self.diff_y_conv = nn.Conv2d(in_channels, out_channels, 2,
                                         stride, 0, dilation, groups,
                                         bias, padding_mode)
            #self.diff_y_conv.weight = torch.nn.Parameter(kernel_y)
            self.diff_y_bn = nn.BatchNorm2d(out_channels)
            self.sobel = RobertFilter()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.identity_conv(x)
        identity = self.identity_bn(identity)
        if self.is_3x3:
            diff_x, diff_y = self.sobel(x)
            diff_x= nn.functional.pad(diff_x, [0, 1, 0, 1], mode='constant')
            diff_x = self.diff_x_conv(diff_x)
            diff_x = self.diff_x_bn(diff_x)

            diff_y= nn.functional.pad(diff_y, [1, 0, 0, 1], mode='constant')
            diff_y = self.diff_y_conv(diff_y)
            diff_y = self.diff_y_bn(diff_y)
            return identity + diff_x + diff_y
        elif self.is_7x7:
            diff_x, diff_y = self.sobel(x)
            diff_x = self.diff_x_conv(diff_x)
            diff_x = self.diff_x_bn(diff_x)

            diff_y = self.diff_y_conv(diff_y)
            diff_y = self.diff_y_bn(diff_y)
            return identity + diff_x + diff_y
        else:
            return identity


class AdaptV14_(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = False,
            padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        super(AdaptV14_, self).__init__()
        self.is_3x3 = False
        self.is_7x7 = False
        if isinstance(kernel_size, numbers.Number):
            if kernel_size == 3:
                self.is_3x3 = True
            elif kernel_size == 7:
                self.is_7x7 = True
        else:
            for _ in kernel_size:
                if _ == 3:
                    self.is_3x3 = True
                elif _ == 7:
                    self.is_7x7 = True
        print("Check conv type: is3x3 {},{}".format(self.is_3x3, kernel_size))
        print("Check conv type: is7x7 {},{}".format(self.is_7x7, kernel_size))
        self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation, groups,
                                       bias, padding_mode)
        self.identity_bn = nn.BatchNorm2d(out_channels)
        self.att_weights=nn.Parameter(torch.ones(3,out_channels),requires_grad=True)

        if self.is_7x7:
            self.diff_x_conv = nn.Conv2d(in_channels, out_channels, 5,
                                         stride, 2, dilation, groups,
                                         bias, padding_mode)
            self.diff_x_bn = nn.BatchNorm2d(out_channels)
            self.diff_y_conv = nn.Conv2d(in_channels, out_channels, 5,
                                         stride, 2, dilation, groups,
                                         bias, padding_mode)
            self.diff_y_bn = nn.BatchNorm2d(out_channels)

            self.sobel = SobelFilter()
        elif self.is_3x3:
            self.diff_x_conv = nn.Conv2d(in_channels, out_channels, 2,
                                         stride, 0, dilation, groups,
                                         bias, padding_mode)
            self.diff_x_bn = nn.BatchNorm2d(out_channels)
            self.diff_y_conv = nn.Conv2d(in_channels, out_channels, 2,
                                         stride, 0, dilation, groups,
                                         bias, padding_mode)
            #self.diff_y_conv.weight = torch.nn.Parameter(kernel_y)
            self.diff_y_bn = nn.BatchNorm2d(out_channels)
            self.sobel = RobertFilter()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.identity_conv(x)
        identity = self.identity_bn(identity)
        if self.is_3x3:
            diff_x, diff_y = self.sobel(x)
            diff_x= nn.functional.pad(diff_x, [0, 1, 0, 1], mode='constant')
            diff_x = self.diff_x_conv(diff_x)
            diff_x = self.diff_x_bn(diff_x)

            diff_y= nn.functional.pad(diff_y, [1, 0, 0, 1], mode='constant')
            diff_y = self.diff_y_conv(diff_y)
            diff_y = self.diff_y_bn(diff_y)
            return identity * self.att_weights[0].view(1, -1, 1, 1) + \
                   diff_x * self.att_weights[1].view(1, -1, 1, 1) + \
                   diff_y * self.att_weights[2].view(1, -1, 1, 1)
        elif self.is_7x7:
            diff_x, diff_y = self.sobel(x)
            diff_x = self.diff_x_conv(diff_x)
            diff_x = self.diff_x_bn(diff_x)

            diff_y = self.diff_y_conv(diff_y)
            diff_y = self.diff_y_bn(diff_y)
            #rint(".............",type(torch.nn.functional.softmax(self.att_weights,dim=0)))
            #self.att_weights=torch.nn.functional.softmax(self.att_weights,dim=0)
            #print("identity  ----",identity.size)
            #print(self.att_weights)
            return identity * self.att_weights[0].view(1,-1,1,1) + \
                   diff_x * self.att_weights[1].view(1,-1,1,1) + \
                   diff_y * self.att_weights[2].view(1,-1,1,1)
        else:
            return identity

conv_types = {
    'vanilla': VanillaConv,
    'diffv0': DiffConvV0,
    'diffv1': DiffConvV1,
    'diffv2': DiffConvV2,
    'diffv3': DiffConvV3,
    'diffv4': DiffConvV4,
    'diffv5': DiffConvV5,
    'diffv6': DiffConvV6,
    'diffv7': DiffConvV7,
    'diffv8': DiffConvV8,
    'diffv9': DiffConvV9,
    'diffv10': DiffConvV10,
    'diffv11': DiffConvV11,
    'sobelinitial': SobelInitial,
    'diffv12': DiffConvV12,
    'diffv13': DiffConvV13,
    'diffv13_': DiffConvV13_,
    'diffv14': DiffConvV14,
    'diffv14_': DiffConvV14_,
    'adaptv14_': AdaptV14_,
    'multiv2': MultiV2,
}
