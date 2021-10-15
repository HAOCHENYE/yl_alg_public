import torch.nn as nn
from mmcv.cnn import ConvModule
import torch.nn.functional as F
from builder import NECK


@NECK.register_module()
class RetinaFaceNeck(nn.Module):
    def __init__(
        self, in_channels, out_channels, norm_cfg=dict(
            type="BN"), conv_cfg=dict(
            type="Conv2d"), act_cfg=dict(
                type="LeakyReLU")):
        super(RetinaFaceNeck, self).__init__()
        negative_slope = 0
        if (out_channels <= 64):
            negative_slope = 0.1
        act_cfg["negative_slope"] = negative_slope
        self.output1 = ConvModule(
            in_channels[0],
            out_channels,
            kernel_size=1,
            stride=1,
            norm_cfg=norm_cfg,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg)
        self.output2 = ConvModule(
            in_channels[1],
            out_channels,
            kernel_size=1,
            stride=1,
            norm_cfg=norm_cfg,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg)
        self.output3 = ConvModule(
            in_channels[2],
            out_channels,
            kernel_size=1,
            stride=1,
            norm_cfg=norm_cfg,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg)

        self.merge1 = ConvModule(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            norm_cfg=norm_cfg,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg)
        self.merge2 = ConvModule(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            norm_cfg=norm_cfg,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg)

    def forward(self, input):
        # names = list(input.keys())
        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(
            output3,
            size=[
                output2.size(2),
                output2.size(3)],
            mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(
            output2,
            size=[
                output1.size(2),
                output1.size(3)],
            mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out
