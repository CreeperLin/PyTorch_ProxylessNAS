import torch
import torch.nn as nn
import math

class GroupConv(nn.Module):
    """
    BN - ReLU - Conv
    """
    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0, groups=1, relu=True, affine=True):
        super().__init__()
        self.bn = nn.BatchNorm2d(C_in, affine=affine)
        self.act = nn.ReLU(inplace=True) if relu else None
        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride, padding, groups=groups, bias=False)

    def forward(self, x):
        x = self.bn(x)
        x = x if self.act is None else self.act(x)
        x = self.conv(x)
        return x

class BottleneckBlock(nn.Module):
    def __init__(self, C_in, C, stride=1, groups=1, bneck_ratio=4,
                 downsample=None, cell_cls=None, cell_kwargs={}):
        super(BottleneckBlock, self).__init__()
        self.bottle_in = GroupConv(C_in, C, 1, 1, 0, groups, relu=False)
        self.cell = cell_cls(**cell_kwargs)
        self.bottle_out = GroupConv(C, C * bneck_ratio, 1, 1, 0, groups)
        self.bn = nn.BatchNorm2d(C * bneck_ratio)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.bottle_in(out)
        out = self.cell(out)
        out = self.bottle_out(out)
        out = self.bn(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.zeros(batch_size, residual_channel - shortcut_channel,
                                 featuremap_size[0], featuremap_size[1])
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut 

        return out


class PyramidNet(nn.Module):
        
    def __init__(self, config, cell_cls, cell_kwargs):
        super(PyramidNet, self).__init__()
        self.chn_in = config.channel_in
        self.chn = config.channel_init
        self.n_classes = config.classes
        self.n_groups = config.groups
        self.n_blocks = config.blocks
        self.n_inputs_model = config.inputs_model
        self.n_inputs_layer = config.inputs_layer
        self.n_inputs_node = config.inputs_node
        self.conv_groups = config.conv_groups
        self.bneck_ratio = config.bottleneck_ratio
        self.addrate = config.alpha / (self.n_groups*self.n_blocks*1.0)
        
        block = BottleneckBlock
        self.chn_cur = self.chn
        self.conv1 = nn.Conv2d(self.chn_in, self.chn_cur, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(self.chn_cur)
        
        self.groups = nn.ModuleList((self.pyramidal_make_layer(block, self.n_blocks, 1, cell_cls, cell_kwargs)))
        for i in range(1, self.n_groups):
            self.groups.append(self.pyramidal_make_layer(block, self.n_blocks, 2, cell_cls, cell_kwargs))
        
        self.bn_final= nn.BatchNorm2d(self.chn_cur)
        self.relu_final = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.chn_cur, self.n_classes)

    def pyramidal_make_layer(self, block, n_blocks, stride, cell_cls, cell_kwargs):
        downsample = None
        if stride != 1: # or self.chn_cur != int(round(featuremap_dim_1st)) * block.outchannel_ratio:
            downsample = nn.AvgPool2d((2,2), stride = (2, 2), ceil_mode=True)

        layers = []
        chn_prev = self.chn_cur
        self.chn_cur += self.addrate
        layers.append(block(chn_prev, int(round(self.chn_cur)), stride, downsample))
        for i in range(1, n_blocks):
            chn_next = int(round(self.chn_cur + self.addrate))
            chn_next -= chn_next % groups
            blk = block(int(round(self.chn_cur)) * self.bneck_ratio,
                        chn_next, 1,
                        cell_cls, cell_kwargs)
            layers.append(blk)
            self.chn_cur += self.addrate
        self.chn_cur = int(round(self.chn_cur)) * self.bneck_ratio
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        
        for g in self.groups:
            x = g(x)

        x = self.bn_final(x)
        x = self.relu_final(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
    
        return x
