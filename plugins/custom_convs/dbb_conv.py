import torch.nn as nn
from mmcv.cnn import fuse_conv_bn
from mmcv.cnn.bricks.registry import CONV_LAYERS
from mmcv.cnn import ConvModule
import torch.nn.functional as F
import torch
from collections import OrderedDict


@CONV_LAYERS.register_module()
class DBBConv(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 bias=False):
        super(DBBConv, self).__init__(in_channels, out_channels,
                 kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, groups=1, bias=bias)
        self.weight = None
        self.bias = None

        self.group = groups
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_kxk = ConvModule(in_channels, out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=kernel_size//2,
                                   norm_cfg=norm_cfg,
                                   groups=groups,
                                   act_cfg=None)
        kxk_1x1 = OrderedDict()

        kxk_1x1["conv_1x1"] = ConvModule(in_channels,  out_channels, kernel_size=1, stride=1, norm_cfg=norm_cfg, groups=groups, act_cfg=None)
        kxk_1x1["conv_kxk"] = ConvModule(out_channels, out_channels, kernel_size=kernel_size,
                                                        stride=stride,
                                                        padding=kernel_size//2,
                                                        norm_cfg=norm_cfg,
                                                        groups=groups,
                                                        act_cfg=None)
        self.kxk_1x1 = nn.Sequential(kxk_1x1)

        conv_1x1_avg = OrderedDict()
        conv_1x1_avg["conv_1x1"] = ConvModule(in_channels, out_channels, kernel_size=1,
                                                              stride=1,
                                                              norm_cfg=norm_cfg,
                                                              groups=groups,
                                                              act_cfg=None)
        conv_1x1_avg["avg"] = nn.AvgPool2d(kernel_size=kernel_size, padding=1, stride=stride)
        self.conv_1x1_avg = nn.Sequential(conv_1x1_avg)


        self.conv_1x1 = ConvModule(in_channels, out_channels, kernel_size=(1, 1), stride=stride, padding=(0, 0),
                                       norm_cfg=norm_cfg, act_cfg=None, groups=groups)
        self.stride = stride

        self.conv = nn.ModuleList([self.conv_kxk, self.kxk_1x1, self.conv_1x1_avg, self.conv_1x1])

        self.init_weight()

    def init_weight(self):
        def ini(module):
            for name, child in module.named_children():
                if isinstance(child, (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
                    nn.init.constant_(child.running_var, 0.3)
                else:
                    if isinstance(child, nn.Module):
                        ini(child)
            return
        ini(self)

    def fuse_kxk_1x1(self, k1, k2, b1, b2):
        if self.group == 1:
            conv_1x1_kxk_weight = F.conv2d(k2, k1.permute(1, 0, 2, 3))  #
            conv_1x1_kxk_bias = (k2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3))
        else:
            k_slices = []
            b_slices = []
            k1_T = k1.permute(1, 0, 2, 3)
            k1_group_width = k1.size(0) // self.group
            k2_group_width = k2.size(0) // self.group
            for g in range(self.group):
                k1_T_slice = k1_T[:, g * k1_group_width:(g + 1) * k1_group_width, :, :]
                k2_slice = k2[g * k2_group_width:(g + 1) * k2_group_width, :, :, :]
                k_slices.append(F.conv2d(k2_slice, k1_T_slice))
                b_slices.append(
                    (k2_slice * b1[g * k1_group_width:(g + 1) * k1_group_width].reshape(1, -1, 1, 1)).sum((1, 2, 3)))
            conv_1x1_kxk_weight, conv_1x1_kxk_bias = torch.cat(k_slices), torch.cat(b_slices)
        return conv_1x1_kxk_weight, conv_1x1_kxk_bias + b2

    def fuse_conv(self):
        fuse_conv_bn(self)

        conv_1x1_weight = nn.Parameter(torch.zeros_like(self.conv_kxk.conv.weight))

        conv_1x1_weight[:, :, self.kernel_size // 2:self.kernel_size // 2 + 1,
                              self.kernel_size // 2:self.kernel_size // 2 + 1] = self.conv_1x1.conv.weight
        conv_1x1_bias = self.conv_1x1.conv.bias
        conv_1x1_kxk_weight, conv_1x1_kxk_bias = self.fuse_kxk_1x1(self.kxk_1x1.conv_1x1.conv.weight, self.kxk_1x1.conv_kxk.conv.weight, self.kxk_1x1.conv_1x1.conv.bias, self.kxk_1x1.conv_kxk.conv.bias)


        avg_pooling_weight = nn.Parameter(torch.ones_like(self.conv_kxk.conv.weight) / self.kernel_size**2)
        avg_pooling_mask_ = torch.eye(self.in_channels //self.group).reshape(self.in_channels // self.group, self.in_channels // self.group, 1, 1).to(self.conv_kxk.conv.weight.device)
        avg_pooling_mask = torch.cat([avg_pooling_mask_ for _ in range(self.group)])

        avg_pooling_weight = avg_pooling_mask * avg_pooling_weight
        conv_1x1_avg_weight, conv_1x1_avg_bias = self.fuse_kxk_1x1(self.conv_1x1_avg.conv_1x1.conv.weight, avg_pooling_weight, self.conv_1x1_avg.conv_1x1.conv.bias, 0)

        self.conv_kxk.conv.weight = nn.Parameter(conv_1x1_weight + conv_1x1_kxk_weight + conv_1x1_avg_weight + self.conv_kxk.conv.weight)
        self.conv_kxk.conv.bias = nn.Parameter(conv_1x1_bias + conv_1x1_kxk_bias + conv_1x1_avg_bias + self.conv_kxk.conv.bias)

        self.conv = nn.ModuleList([self.conv_kxk])

    def forward(self, x):
        out = []
        for module in self.conv:
            out.append(module(x))
        res = out[0]
        for i in range(1, len(out)):
            res += out[i]
        return res
