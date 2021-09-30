import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.cnn import constant_init, kaiming_init
from builder import BACKBONES
import os
from collections import OrderedDict


class VGGStage(nn.Module):
    def __init__(
        self, in_ch, stage_ch, num_block, kernel_size=3, groups=1,
            conv_cfg=dict(type="Conv2d"),
            norm_cfg=dict(type="BN"),
            act_cfg=dict(type="ReLU")):
        super(VGGStage, self).__init__()
        LayerDict = OrderedDict()
        padding = kernel_size // 2
        if conv_cfg["type"] in ["RepVGGConv", "DBBConv"]:
            norm_cfg = None

        for num in range(num_block):
            if num == 0:
                LayerDict["Block{}".format(num)] = \
                    ConvModule(in_ch, stage_ch, kernel_size,
                               groups=groups, stride=2, padding=padding,
                               norm_cfg=dict(type="BN"),
                               act_cfg=act_cfg)
                continue
            LayerDict["Block{}".format(num)] = ConvModule(stage_ch,
                                                          stage_ch,
                                                          kernel_size,
                                                          groups=groups,
                                                          stride=1,
                                                          padding=padding,
                                                          conv_cfg=conv_cfg,
                                                          act_cfg=act_cfg,
                                                          norm_cfg=norm_cfg)
        self.Block = nn.Sequential(LayerDict)

    def forward(self, x):
        return self.Block(x)


@BACKBONES.register_module()
class VGGNet(nn.Module):
    def __init__(self,
                 stem_channels,
                 stage_channels,
                 block_per_stage,
                 kernel_size=3,
                 num_out=5,
                 conv_cfg=dict(type="Conv2d"),
                 act_cfg=dict(type="ReLU")
                 ):
        super(VGGNet, self).__init__()
        if isinstance(kernel_size, int):
            kernel_sizes = [kernel_size for _ in range(len(stage_channels))]
        if isinstance(kernel_size, list):
            assert len(kernel_size) == len(stage_channels), \
                "if kernel_size is list, len(kernel_size) should == len(stage_channels)"
            kernel_sizes = kernel_size

        assert num_out <= len(
            stage_channels), 'num output should be less than stage channels!'

        self.stage_nums = len(stage_channels)
        self.stem = ConvModule(
            3,
            stem_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=dict(type='BN', requires_grad=True))
        '''defult end_stage is the last stage'''
        self.start_stage = len(stage_channels) - num_out + 1
        self.stages = nn.ModuleList()
        self.last_stage = len(stage_channels)
        in_channel = stem_channels
        for num_stages in range(self.stage_nums):
            stage = VGGStage(
                in_channel,
                stage_channels[num_stages],
                block_per_stage[num_stages],
                kernel_size=kernel_sizes[num_stages],
                groups=1,
                conv_cfg=conv_cfg,
                act_cfg=act_cfg)
            in_channel = stage_channels[num_stages]
            self.stages.append(stage)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            import torch
            assert os.path.isfile(
                pretrained), "file {} not found.".format(pretrained)
            self.load_state_dict(torch.load(pretrained), strict=False)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.stem(x)
        for i in range(self.start_stage):
            x = self.stages[i](x)
        out = []
        for i in range(self.start_stage, len(self.stages)):
            out.append(x)
            x = self.stages[i](x)
        out.append(x)
        return out
