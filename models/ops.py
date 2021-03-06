# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

OPS_ORDER = ['bn','act','weight']
sepconv_stack = False

def configure_ops(config):
    global OPS_ORDER
    OPS_ORDER = config.ops_order.split('_')
    print('ops order set to: {}'.format(OPS_ORDER))
    
    global sepconv_stack
    sepconv_stack = config.sepconv_stack
    print('SepConv stack: {}'.format(sepconv_stack))

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: PoolBN('avg', C, 3, stride, 1, affine=affine),
    'max_pool_3x3': lambda C, stride, affine: PoolBN('max', C, 3, stride, 1, affine=affine),
    'skip_connect': lambda C, stride, affine: \
        Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine), # 5x5
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine), # 9x9
    'conv_7x1_1x7': lambda C, stride, affine: FacConv(C, C, 7, stride, 3, affine=affine),
    'conv_1x1': lambda C, stride, affine: StdConv(C, C, 1, stride, 0, affine=affine)
}


def drop_path_(x, drop_prob, training):
    if training and drop_prob > 0.:
        keep_prob = 1. - drop_prob
        # per data point mask; assuming x in cuda.
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob).mul_(mask)

    return x


class DropPath_(nn.Module):
    def __init__(self, p=0.):
        """ [!] DropPath is inplace module
        Args:
            p: probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def extra_repr(self):
        return 'p={}, inplace'.format(self.p)

    def forward(self, x):
        drop_path_(x, self.p, self.training)

        return x


class MBConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, expansion, affine=True):
        super().__init__()
        C_t = C_in * expansion
        nets = [] if expansion == 1 else [
            nn.Conv2d(C_in, C_t, 1, 1, 0, bias=False),
            nn.BatchNorm2d(C_t, affine),
            nn.ReLU6(),
        ]
        nets.extend([
            nn.Conv2d(C_t, C_t, kernel_size, stride, padding, groups=C_t, bias=False),
            nn.BatchNorm2d(C_t, affine),
            nn.ReLU6(),
            nn.Conv2d(C_t, C_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(C_out, affine)
        ])
        self.net=nn.Sequential(*nets)

    def forward(self, x):
        return self.net(x)


class PoolBN(nn.Module):
    """
    AvgPool or MaxPool - BN
    """
    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True):
        """
        Args:
            pool_type: 'max' or 'avg'
        """
        super().__init__()
        if pool_type.lower() == 'max':
            pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            pool = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)
        else:
            raise ValueError()

        nets = []
        for i in OPS_ORDER:
            if i=='bn':
                nets.append(nn.BatchNorm2d(C, affine=affine))
            elif i=='weight':
                nets.append(pool)
            elif i=='act':
                pass

        self.net = nn.Sequential(*nets)

    def forward(self, x):
        return self.net(x)


class StdConv(nn.Module):
    """ Standard conv
    ReLU - Conv - BN
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        C = C_in
        nets = []
        for i in OPS_ORDER:
            if i=='bn':
                nets.append(nn.BatchNorm2d(C, affine=affine))
            elif i=='weight':
                bias = False if OPS_ORDER[-1] == 'bn' else True
                nets.append(nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=bias))
                C = C_out
            elif i=='act':
                nets.append(nn.ReLU(inplace=False if OPS_ORDER[0]=='act' else True))
        self.net = nn.Sequential(*nets)

    def forward(self, x):
        return self.net(x)


class FacConv(nn.Module):
    """ Factorized conv
    ReLU - Conv(Kx1) - Conv(1xK) - BN
    """
    def __init__(self, C_in, C_out, kernel_length, stride, padding, affine=True):
        super().__init__()
        C = C_in
        nets = []
        for i in OPS_ORDER:
            if i=='bn':
                nets.append(nn.BatchNorm2d(C, affine=affine))
            elif i=='weight':
                bias = False if OPS_ORDER[-1] == 'bn' else True
                nets.append(nn.Conv2d(C_in, C_in, (kernel_length, 1), stride, padding, bias=bias))
                nets.append(nn.Conv2d(C_in, C_out, (1, kernel_length), stride, padding, bias=bias))
                C = C_out
            elif i=='act':
                nets.append(nn.ReLU(inplace=False if OPS_ORDER[0]=='act' else True))

        self.net = nn.Sequential(*nets)

    def forward(self, x):
        return self.net(x)


class DilConv(nn.Module):
    """ (Dilated) depthwise separable conv
    ReLU - (Dilated) depthwise separable - Pointwise - BN

    If dilation == 2, 3x3 conv => 5x5 receptive field
                      5x5 conv => 9x9 receptive field
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        C = C_in
        nets = []
        for i in OPS_ORDER:
            if i=='bn':
                nets.append(nn.BatchNorm2d(C, affine=affine))
            elif i=='weight':
                bias = False if OPS_ORDER[-1] == 'bn' else True
                nets.append(nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in, bias=bias))
                nets.append(nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=bias))
                C = C_out
            elif i=='act':
                nets.append(nn.ReLU(inplace=False if OPS_ORDER[0]=='act' else True))
        self.net = nn.Sequential(*nets)

    def forward(self, x):
        return self.net(x)


class SepConv(nn.Module):
    """ Depthwise separable conv
    DilConv(dilation=1) * 2
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        if sepconv_stack:
            self.net = nn.Sequential(
                DilConv(C_in, C_in, kernel_size, stride, padding,   dilation=1, affine=affine),
                DilConv(C_in, C_out, kernel_size, 1, padding, dilation=1, affine=affine)
            )
            return
        C = C_in
        nets = []
        for i in OPS_ORDER:
            if i=='bn':
                nets.append(nn.BatchNorm2d(C, affine=affine))
            elif i=='weight':
                bias = False if OPS_ORDER[-1] == 'bn' else True
                nets.append(nn.Conv2d(C_in, C_in, kernel_size, stride, padding, groups=C_in, bias=bias))
                nets.append(nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=bias))
                C = C_out
            elif i=='act':
                nets.append(nn.ReLU(inplace=False if OPS_ORDER[0]=='act' else True))
        self.net = nn.Sequential(*nets)

    def forward(self, x):
        return self.net(x)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.

        # re-sizing by stride
        return x[:, :, ::self.stride, ::self.stride] * 0.


class FactorizedReduce(nn.Module):
    """
    Reduce feature map size by factorized pointwise(stride=2).
    """
    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

