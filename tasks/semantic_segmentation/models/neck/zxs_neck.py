import torch.nn as nn
from mmcv.cnn import ConvModule
import torch.nn.functional as F
from builder import NECK

class HPM(nn.Module):
    def __init__(self,
                 in_channel_left,
                 in_channel_down,
                 out_channels=64):
        super(HPM, self).__init__()
        self.conv0 = ConvModule(in_channel_left, out_channels, 1, norm_cfg=dict(type="BN"))
        self.conv1 = ConvModule(in_channel_down, out_channels, 1)
        self.conv2 = ConvModule(out_channels, out_channels, 1)

        # self.global_pool = nn.AvgPool2d((8, 14), count_include_pad=False)

    def forward(self, left, down):
        left = self.conv0(left)
        down = F.adaptive_avg_pool2d(down, (1, 1))
        # down = down.mean(dim=(2, 3), keepdim=True)  # down = self.global_pool(down)

        down = self.conv1(down)
        down = self.conv2(down)
        return left + down


class SPM(nn.Module):
    def __init__(self, in_channel, conv_cfg=dict(type='Conv2d')):
        super(SPM, self).__init__()

        self.conv1 = ConvModule(in_channel,
                                in_channel,
                                kernel_size=3,
                                padding=1,
                                norm_cfg=dict(type='BN'),
                                conv_cfg=conv_cfg)

        self.conv2m = ConvModule(in_channel, in_channel, 1, act_cfg=None)
        self.conv2a = ConvModule(in_channel, in_channel, 1, act_cfg=None)

    def forward(self, x):
        out1 = self.conv1(x)

        w = self.conv2m(out1)
        b = self.conv2a(out1)

        out = F.relu(w * x + b, inplace=True)
        return out


class MPM(nn.Module):
    _up_kwargs = dict(mode='bilinear', align_corners=True)

    def __init__(self, in_channel_left, in_channel_down, mid_channel,
                 conv_cfg=dict(type='Conv2d')):
        super(MPM, self).__init__()

        self.conv0 = ConvModule(in_channel_left, mid_channel, 1, norm_cfg=dict(type='BN'))
        self.conv1 = ConvModule(in_channel_down, in_channel_down, 3, padding=1, norm_cfg=dict(type='BN'), conv_cfg=conv_cfg)

        self.conv2m = ConvModule(in_channel_down, mid_channel, 1, act_cfg=None)
        self.conv2a = ConvModule(in_channel_down, mid_channel, 1, act_cfg=None)

    def forward(self, left, down):
        left = self.conv0(left)
        down = self.conv1(down)

        w = self.conv2m(down)
        b = self.conv2a(down)
        if down.size()[2:] != left.size()[2:]:
            w = F.interpolate(w, size=left.size()[2:], **self._up_kwargs)
            b = F.interpolate(b, size=left.size()[2:], **self._up_kwargs)

        out = F.relu(w * left + b, inplace=True)
        return out


@NECK.register_module()
class ZxsNeck(nn.Module):
    _up_kwargs = dict(mode='bilinear', align_corners=True)

    def __init__(self,
                 in_channels=[32, 96, 192, 256],
                 mid_channels=[32, 48, 64],
                 out_channels=[32, 32, 48]):
        super().__init__()
        self.hpm = HPM(in_channels[-1], in_channels[-1])

        self.mpm23 = MPM(in_channels[0], mid_channels[0], out_channels[0])
        self.mpm34 = MPM(in_channels[1], mid_channels[1], out_channels[1])
        self.mpm45 = MPM(in_channels[2], mid_channels[2], out_channels[2])

        self.spm2 = SPM(out_channels[0])
        self.spm3 = SPM(out_channels[1])
        self.spm4 = SPM(out_channels[2])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out2, out3, out4, out5 = x

        out5 = self.hpm(out5, out5)
        out4 = self.spm4(self.mpm45(out4, out5))
        out3 = self.spm3(self.mpm34(out3, out4))
        out2 = self.spm2(self.mpm23(out2, out3))

        return out2, out3, out4, out5



