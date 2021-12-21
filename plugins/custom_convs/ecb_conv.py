import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer, build_activation_layer
from mmcv.cnn.bricks.registry import CONV_LAYERS
from typing import Optional
from onnxsim import simplify
import onnx


def _fuse_bn(conv_w, conv_b, bn):
    conv_b = conv_b if conv_b is not None else torch.zeros_like(
        bn.running_mean)

    factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    fuse_weight = nn.Parameter(conv_w *
                               factor.reshape([conv_w.shape[0], 1, 1, 1]))
    fuse_bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)
    return fuse_weight, fuse_bias


class ECBSepConv(nn.Module):
    """
        if bn_cfg is none ,bias will enabled
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion: int = 1,
                 stride: int = 1,
                 seq_type: str = 'conv1x1-conv3x3',
                 norm_cfg: Optional[dict] = dict(type="BN"),
                 depth_wise: bool = False,
                 ):
        super(ECBSepConv, self).__init__()
        self.seq_type = seq_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.mid_channels = int(out_channels * expansion)
        self.stride = stride
        self.depth_wise = depth_wise
        if depth_wise:
            assert in_channels == out_channels
            assert expansion == 1
            self.infer_groups = in_channels
            self.groups = in_channels
            con1x1_channel = in_channels
        else:
            if seq_type == 'conv1x1-conv3x3':
                con1x1_channel = self.mid_channels
                self.infer_groups = 1
                self.groups = 1
            else:
                con1x1_channel = self.out_channels
                self.infer_groups = 1
                self.groups = self.out_channels
        self.conv1x1 = ConvModule(
            in_channels,
            con1x1_channel,
            kernel_size=1,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=None,
            groups=self.infer_groups)

        self._init_conv3x3()

    def _init_conv3x3(self):
        self.mask, self.scale = self._init_target_kernel()
        if self.norm_cfg:
            self.bias = None
            _, self.norm = build_norm_layer(self.norm_cfg, self.out_channels)
        else:
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            self.norm = nn.Identity()

    def _init_target_kernel(self):
        if self.seq_type == 'conv1x1-conv3x3':
            mask = torch.zeros(
                (self.out_channels, self.mid_channels // self.groups, 3, 3), dtype=torch.float32)
        else:
            mask = torch.zeros(
                (self.out_channels, 1, 3, 3), dtype=torch.float32)
        if self.seq_type == 'conv1x1-sobelx':
            mask[..., 0, 0] = 1.0
            mask[..., 1, 0] = 2.0
            mask[..., 2, 0] = 1.0
            mask[..., 0, 2] = -1.0
            mask[..., 1, 2] = -2.0
            mask[..., 2, 2] = -1.0
        elif self.seq_type == 'conv1x1-sobely':
            mask[..., 0, 0] = 1.0
            mask[..., 0, 1] = 2.0
            mask[..., 0, 2] = 1.0
            mask[..., 2, 0] = -1.0
            mask[..., 2, 1] = -2.0
            mask[..., 2, 2] = -1.0
        elif self.seq_type == 'conv1x1-laplacian':
            mask[..., 0, 1] = 1.0
            mask[..., 1, 0] = 1.0
            mask[..., 1, 2] = 1.0
            mask[..., 2, 1] = 1.0
            mask[..., 1, 1] = -4.0
        mask = nn.Parameter(data=mask, requires_grad=False)
        scale = nn.Parameter(torch.rand(mask.shape))
        return mask, scale

    def reparam(self):
        device = self.mask.device
        if self.norm_cfg:
            tmp_weight_1x1, tmp_bias_1x1 = _fuse_bn(
                self.conv1x1.conv.weight, self.conv1x1.conv.bias, self.conv1x1.norm)
            tmp_weight_3x3, tmp_bias_3x3 = _fuse_bn(
                self.mask * self.scale, self.bias, self.norm)
        else:
            tmp_weight_1x1 = self.conv1x1.conv.weight
            tmp_bias_1x1 = self.conv1x1.conv.bias
            tmp_weight_3x3 = self.mask * self.scale
            tmp_bias_3x3 = self.bias

        bias_1x1 = tmp_bias_1x1.view(1, -1, 1, 1)
        if self.depth_wise:
            weight_3x3 = tmp_weight_3x3
            bias_channels = self.out_channels
        else:
            if self.seq_type == 'conv1x1-conv3x3':
                weight_3x3 = tmp_weight_3x3
                bias_channels = self.mid_channels
            else:
                weight_3x3 = torch.zeros(
                    self.out_channels, self.out_channels, 3, 3, device=device)
                for i in range(self.out_channels):
                    weight_3x3[i, i, :, :] = tmp_weight_3x3[i, 0, :, :]

                bias_channels = self.out_channels

        if self.depth_wise:
            weight = F.conv2d(input=weight_3x3.permute(1, 0, 2, 3),
                              weight=tmp_weight_1x1,
                              stride=1,
                              padding=0,
                              dilation=1,
                              groups=self.groups
                              ).permute(1, 0, 2, 3)
            bias = torch.ones(1, bias_channels, 3, 3, device=device) * bias_1x1
            bias = F.conv2d(input=bias, weight=weight_3x3, groups=self.out_channels).view(-1,) + tmp_bias_3x3
        else:
            weight = F.conv2d(
                input=weight_3x3,
                weight=tmp_weight_1x1.permute(1, 0, 2, 3),
                stride=1,
                padding=0,
                dilation=1)
        # re-param conv bias
            bias = torch.ones(1, bias_channels, 3, 3, device=device) * bias_1x1
            bias = F.conv2d(input=bias, weight=weight_3x3).view(-1,) + tmp_bias_3x3

        return weight, bias

    def forward(self, x):
        if not self.training:
            weight, bias = self.reparam()
            return F.conv2d(x, weight, bias, self.stride, 1, 1, self.infer_groups)
        x = self.conv1x1(x)
        x = F.conv2d(x,
                     self.mask * self.scale,
                     self.bias,
                     self.stride, 0, 1,
                     self.groups)
        x = self.norm(x)
        return x


@CONV_LAYERS.register_module()
class ECBConv(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            norm_cfg: Optional[dict] = dict(
                type='BN',
                requires_grad=True),
            bias=False,
            act_cfg: Optional[dict] = dict(
                type='ReLU'),
            expansion: int = 1,
            ):
        super(ECBConv, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        del self._parameters['weight']
        del self._parameters['bias']
        self.register_buffer('bias', None)
        self.register_buffer('weight', None)
        depth_wise = groups == in_channels
        if depth_wise:
            assert self.out_channels == self.in_channels
            assert expansion == 1
            self.groups = self.out_channels
        self.depth_multiplier = expansion
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_cfg = act_cfg
        self.stride = stride
        self.channel_identity = (in_channels == out_channels)
        self.depth_wise = depth_wise

        self.conv3x3 = ConvModule(
            in_channels,
            out_channels,
            stride=stride,
            kernel_size=3,
            padding=1,
            act_cfg=None,
            norm_cfg=norm_cfg,
            groups=self.groups)
        self.conv1x1_3x3 = ECBSepConv(
            in_channels=in_channels,
            out_channels=out_channels,
            expansion=expansion,
            seq_type='conv1x1-conv3x3',
            stride=stride,
            norm_cfg=norm_cfg,
            depth_wise=depth_wise)
        self.conv1x1_sbx = ECBSepConv(
            in_channels=in_channels,
            out_channels=out_channels,
            expansion=expansion,
            seq_type='conv1x1-sobelx',
            stride=stride,
            norm_cfg=norm_cfg,
            depth_wise=depth_wise)
        self.conv1x1_sby = ECBSepConv(
            in_channels=in_channels,
            out_channels=out_channels,
            expansion=expansion,
            seq_type='conv1x1-sobely',
            stride=stride,
            norm_cfg=norm_cfg,
            depth_wise=depth_wise)
        self.conv1x1_lpl = ECBSepConv(
            in_channels=in_channels,
            out_channels=out_channels,
            expansion=expansion,
            seq_type='conv1x1-laplacian',
            norm_cfg=norm_cfg,
            stride=stride,
            depth_wise=depth_wise)

        self.act = build_activation_layer(act_cfg)
        self.fuse_conv()

    def reparam(self):
        K0, B0 = _fuse_bn(self.conv3x3.conv.weight, self.conv3x3.conv.bias, self.conv3x3.norm)
        K1, B1 = self.conv1x1_3x3.reparam()
        K2, B2 = self.conv1x1_sbx.reparam()
        K3, B3 = self.conv1x1_sby.reparam()
        K4, B4 = self.conv1x1_lpl.reparam()
        RK, RB = (K0 + K1 + K2 + K3 + K4), (B0 + B1 + B2 + B3 + B4)

        device = RB.device
        if self.channel_identity and self.stride == 1:
            if self.depth_wise:
                K_idt = torch.zeros(
                    self.out_channels,
                    1,
                    3,
                    3,
                    device=device)
                K_idt[..., 1, 1] = 1.0
                B_idt = 0.0
                RK, RB = RK + K_idt, RB + B_idt
            else:
                K_idt = torch.zeros(
                    self.out_channels,
                    self.out_channels,
                    3,
                    3,
                    device=device)
                for i in range(self.out_channels):
                    K_idt[i, i, 1, 1] = 1.0
                B_idt = 0.0
                RK, RB = RK + K_idt, RB + B_idt
        return RK, RB

    def fuse_conv(self):
        self.weight, self.bias = self.reparam()

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key == 'training' and value is False:
            self.fuse_conv()

    def _train_forward(self, x):
        y = self.conv3x3(x)
        y += self.conv1x1_3x3(x)
        y += self.conv1x1_sbx(x)
        y += self.conv1x1_sby(x)
        y += self.conv1x1_lpl(x)
        if self.channel_identity and self.stride == 1:
            y += x
        return y

    def forward(self, x):
        if self.training:
            y = self._train_forward(x)
        else:
            y = F.conv2d(input=x, weight=self.weight, bias=self.bias, stride=self.stride, padding=1, groups=self.groups)
            y_train = self._train_forward(x)
            if (y_train - y).mean().abs() > 1e3:
                print(f'fuse conv mean error is: {(y_train - y).mean().abs()}, please check!')
        y = self.act(y)
        return y


if __name__ == '__main__':
    # # test seq-conv
    channel = 196
    x = torch.randn(1, channel, 224, 224)
    stride = 1
    groups = channel
    my_ecb = ECBConv(channel, channel, expansion=1, stride=stride, groups=groups)
    my_ecb.eval()
    res_train = my_ecb.act(my_ecb._train_forward(x))
    w, b = my_ecb.reparam()
    res_val = my_ecb.act(F.conv2d(x, weight=w, bias=b, stride=stride, padding=1, groups=groups))
    print((res_train - res_val).sum())
    export_name = 'test.onnx'
    torch.onnx.export(my_ecb, x, export_name, opset_version=11)
    model_opt, check_ok = simplify(export_name,
                                   3,
                                   True,
                                   False,
                                   dict(),
                                   None,
                                   False,
                                   )
    onnx.save(model_opt, export_name)

