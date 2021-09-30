import torch.nn as nn
from mmcv.cnn import fuse_conv_bn
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.registry import CONV_LAYERS
import torch

@CONV_LAYERS.register_module()
class RepVGGConv(nn.Conv2d):
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
        super(RepVGGConv, self).__init__(in_channels, out_channels,
                 kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, groups=1, bias=bias)
        assert groups == 1 or groups == in_channels, "current only support groups=1 or in_channel"
        self.depth_wise = groups == in_channels
        self.weight = None
        self.bias = None

        self.kernel_size = kernel_size
        self.norm_cfg = norm_cfg
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_3x3 = ConvModule(in_channels, out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=kernel_size//2,
                                   norm_cfg=norm_cfg,
                                   groups=groups,
                                   act_cfg=None)

        self.conv_1x1 = ConvModule(in_channels, out_channels, kernel_size=(1, 1), stride=stride, padding=(0, 0),
                                   groups=groups, norm_cfg=norm_cfg, act_cfg=None)
        self.stride = stride
        if self.stride == 1:
            self.ShortCut = nn.Identity()
            self.conv = nn.ModuleList([self.conv_3x3, self.conv_1x1, self.ShortCut])
        else:
            self.conv = nn.ModuleList([self.conv_3x3, self.conv_1x1])


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
    def fuse_conv(self):
        self.conv_3x3 = fuse_conv_bn(self.conv_3x3)
        self.conv_1x1 = fuse_conv_bn(self.conv_1x1)

        # self.conv = nn.ModuleList([self.conv_kxk_fuse, self.conv_kx1_fuse, self.conv_1xk_fuse])
        self.conv_3x3.conv.weight[:,
                                  :,
                                  self.kernel_size // 2:self.kernel_size // 2 + 1,
                                  self.kernel_size // 2:self.kernel_size // 2 + 1] += self.conv_1x1.conv.weight

        self.conv_3x3.conv.bias = torch.nn.Parameter(self.conv_1x1.conv.bias + self.conv_3x3.conv.bias)

        if self.stride == 1 and self.in_channels == self.out_channels:
            if not self.depth_wise:
                short_cut_weight = torch.nn.Parameter(torch.eye(self.in_channels) \
                                                  .reshape(self.in_channels, self.in_channels, 1, 1)).to(self.conv_3x3.conv.weight.device)
            else:
                short_cut_weight = torch.nn.Parameter(torch.ones_like(self.conv_1x1.conv.weight))
            self.conv_3x3.conv.weight[:,
                                      :,
                                      self.kernel_size // 2:self.kernel_size // 2 + 1,
                                      self.kernel_size // 2:self.kernel_size // 2 + 1] += short_cut_weight

        self.conv = nn.ModuleList([self.conv_3x3])

    def forward(self, x):
        out = []
        for module in self.conv:
            out.append(module(x))
        res = out[0]
        for i in range(1, len(out)):
            res += out[i]
        return res



if __name__ == "__main__":
    x = torch.randn(1, 32, 224, 224)
    conv = RepVGGConv(32, 32, 3, groups=32, stride=1, padding=1)
    conv.eval()
    res_before_fuse = conv(x)
    conv.fuse_conv()
    res_after_fuse = conv(x)
    print(f"error of fuse conv is {torch.abs(res_before_fuse - res_after_fuse).mean()}")