import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import copy


def fuse_kxk_1x1(conv_kxk: torch.Tensor,
                 conv_1x1: torch.Tensor):
    weight_fuse = F.conv2d(conv_kxk,
                           conv_1x1.permute(1, 0, 2, 3),
                           bias=None)
    return weight_fuse


def fuse_parallel_conv(*buffer_list: torch.Tensor):
    return torch.stack(buffer_list).sum(axis=0)


def fuse_kxk_1xk_matmul(weight_conv_kxk: torch.Tensor,
                        bias_conv_kxk: torch.Tensor,
                        weight_conv_1x1: torch.Tensor,
                        bias_conv_1x1: torch.Tensor,
                        groups):
    out_channels, groups_in_channels, _, _ = weight_conv_1x1.shape
    _, groups_out_channels, kernel_size, _ = weight_conv_kxk.shape
    out_groups = out_channels // groups
    weight3x3_ = weight_conv_kxk.reshape(
        groups, out_groups, groups_out_channels, -1).permute(0, 1, 3, 2)
    weight1x1_ = weight_conv_1x1.reshape(
        groups, 1, out_groups, groups_in_channels).repeat(
        1, out_groups, 1, 1)
    weight_fused = torch.matmul(
        weight3x3_,
        weight1x1_).permute(
        0,
        1,
        3,
        2).reshape(
            groups *
            out_groups,
            groups_in_channels,
            kernel_size,
        kernel_size)
    if bias_conv_1x1 is not None:
        bias1x1_ = bias_conv_1x1.reshape(groups, 1, 1, groups_out_channels)
        fused_bias = (bias1x1_ * weight3x3_).sum((2, 3)
                                                 ).view(-1) + bias_conv_kxk
    else:
        fused_bias = None
    return weight_fused, fused_bias


class DBBConv(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 debug=False):
        super(
            DBBConv,
            self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        assert self.kernel_size[0] == self.kernel_size[1], "DBB reparam only support square convolution!"
        self.groups_channels_in = self.in_channels // self.groups
        self.groups_channels_out = self.out_channels // self.groups

        del self.weight
        del self.bias
        self.bias = bias
        self.debug_enabled = debug
        self._init_layers()
        self.init_weights()

    def debug(self, feat):
        res_11 = F.conv2d(
            feat,
            self.conv_11_weight_buffer,
            self.conv_11_bias_buffer,
            groups=self.groups)
        res_kk = F.conv2d(
            feat,
            self.conv_kk_weight_buffer,
            self.conv_kk_bias_buffer,
            padding=1,
            groups=self.groups)

        res_11kk = F.conv2d(
            feat,
            self.conv_11kk_11_weight_buffer,
            self.conv_11kk_11_bias_buffer,
            padding=1,
            groups=self.groups)
        res_11kk = F.conv2d(
            res_11kk,
            self.conv_11kk_kk_weight_buffer,
            self.conv_11kk_kk_bias_buffer,
            groups=self.groups)

        res_avg = F.conv2d(
            feat,
            self.conv_avg_11_weight_buffer,
            self.conv_avg_11_bias_buffer,
            padding=1,
            groups=self.groups)
        res_avg = F.conv2d(
            res_avg,
            self.conv_avg_kk_weight_buffer,
            self.conv_avg_kk_bias_buffer,
            groups=self.groups)
        return res_11kk + res_11 + res_kk + res_avg

    def _init_layers(self):
        self.register_parameter(
            "conv_kk_weight_param",
            nn.Parameter(
                torch.Tensor(
                    self.out_channels,
                    self.in_channels //
                    self.groups,
                    *
                    self.kernel_size)))
        self.register_parameter(
            "conv_11_weight_param",
            nn.Parameter(
                torch.Tensor(
                    self.out_channels,
                    self.in_channels //
                    self.groups,
                    1,
                    1)))
        self.register_parameter(
            "conv_11kk_kk_weight_param",
            nn.Parameter(
                torch.Tensor(
                    self.out_channels,
                    self.out_channels //
                    self.groups,
                    *
                    self.kernel_size)))
        self.register_parameter(
            "conv_11kk_11_weight_param",
            nn.Parameter(
                torch.Tensor(
                    self.out_channels,
                    self.in_channels //
                    self.groups,
                    1,
                    1)))
        self.register_parameter(
            "conv_avg_kk_weight_param",
            nn.Parameter(
                torch.Tensor(
                    self.out_channels,
                    self.out_channels //
                    self.groups,
                    *
                    self.kernel_size)))
        self.register_parameter(
            "conv_avg_11_weight_param",
            nn.Parameter(
                torch.Tensor(
                    self.out_channels,
                    self.in_channels //
                    self.groups,
                    1,
                    1)))

        if self.bias:
            self.register_parameter(
                "conv_kk_bias_param", nn.Parameter(
                    torch.Tensor(
                        self.out_channels)))
            self.register_parameter(
                "conv_11_bias_param", nn.Parameter(
                    torch.Tensor(
                        self.out_channels)))
            self.register_parameter(
                "conv_11kk_kk_bias_param",
                nn.Parameter(
                    torch.Tensor(
                        self.out_channels)))
            self.register_parameter(
                "conv_11kk_11_bias_param",
                nn.Parameter(
                    torch.Tensor(
                        self.out_channels)))
            self.register_parameter(
                "conv_avg_11_bias_param",
                nn.Parameter(
                    torch.Tensor(
                        self.out_channels)))
            self.register_parameter(
                "conv_avg_kk_bias_param",
                nn.Parameter(
                    torch.Tensor(
                        self.out_channels)))
            self.register_buffer(
                "fused_bias", torch.Tensor(
                    torch.Tensor(
                        self.out_channels)))

        self.register_buffer(
            "fused_weight",
            torch.Tensor(
                self.out_channels,
                self.in_channels //
                self.groups,
                *
                self.kernel_size))

    def init_weights(self):
        for name, params in self.named_parameters():
            if len(params.shape) == 4:
                if name.startswith('conv_avg_kk'):
                    with torch.no_grad():
                        params.zero_()
                        value = 1.0 / params.shape[2] * params.shape[3]
                        params[torch.arange(self.out_channels), torch.tile(torch.arange(
                            self.groups_channels_out), [self.groups]), ...] = 1.0 / value
                    params.requires_grad = False
                else:
                    nn.init.kaiming_uniform_(params, a=math.sqrt(5))
            else:
                weight_name = name.replace("bias", "weight")
                weight = getattr(self, weight_name)
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(params, -bound, bound)
        if self.debug_enabled:
            for name, param in self.named_parameters():
                buffer_name = name.replace("param", "buffer")
                self.register_buffer(buffer_name, param.data.requires_grad_())

    def fuse_conv(self):
        self.fused_weight.zero_()
        mid_idx = self.kernel_size[0] // 2
        fuse_kk_11 = fuse_kxk_1x1(
            self.conv_11kk_kk_weight_param,
            self.conv_11kk_11_weight_param)
        fuse_avg = fuse_kxk_1x1(
            self.conv_avg_kk_weight_param,
            self.conv_avg_11_weight_param)
        self.fused_weight[..., mid_idx,
                          mid_idx] = self.conv_11_weight_param[..., 0, 0]
        self.fused_weight.add_(
            fuse_parallel_conv(
                fuse_kk_11,
                fuse_avg,
                self.conv_kk_weight_param))

    def fuse_conv_matmul(self):
        self.fused_weight.zero_()
        mid_idx = self.kernel_size[0] // 2
        fuse_kk_11_weight, fuse_kk_11_bias = fuse_kxk_1xk_matmul(
            self.conv_11kk_kk_weight_param, getattr(
                self, "conv_11kk_kk_bias_param", None), self.conv_11kk_11_weight_param, getattr(
                self, "conv_11kk_11_bias_param", None), self.groups)
        fuse_avg_weight, fuse_avg_bias = fuse_kxk_1xk_matmul(
            self.conv_avg_kk_weight_param, getattr(
                self, "conv_avg_kk_bias_param", None), self.conv_avg_11_weight_param, getattr(
                self, "conv_avg_11_bias_param", None), self.groups)
        self.fused_weight[..., mid_idx,
                          mid_idx] = self.conv_11_weight_param[..., 0, 0]
        self.fused_weight.add_(
            fuse_parallel_conv(
                fuse_kk_11_weight,
                fuse_avg_weight,
                self.conv_kk_weight_param))
        if self.bias:
            self.fused_bias.zero_()
            self.fused_bias.add_(
                fuse_parallel_conv(
                    fuse_kk_11_bias,
                    fuse_avg_bias,
                    self.conv_kk_bias_param,
                    self.conv_11_bias_param))

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key == 'training':
            if not value:
                with torch.no_grad():
                    self.fuse_conv_matmul()
                    self.weight = nn.Parameter(self.fused_weight)
                    self.bias = nn.Parameter(self.fused_bias)
            else:
                if hasattr(self, "conv"):
                    del self.weight
                    del self.bias

    def forward(self, feat: torch.Tensor):
        # self.fuse_conv()
        if self.training:
            self.fuse_conv_matmul()
            feat = F.conv2d(feat,
                            self.fused_weight,
                            self.fused_bias,
                            stride=self.stride,
                            padding=self.padding,
                            dilation=self.dilation,
                            groups=self.groups)
            return feat
        else:
            return self.conv(feat)


if __name__ == "__main__":
    conv = DBBConv(16, 32, 3, bias=True, groups=4, padding=1, debug=True)
    conv.init_weights()
    conv_debug = copy.deepcopy(conv)
    feat = torch.randn(1, 16, 64, 64)
    res = conv(feat)
    debug = conv_debug.debug(feat)
    res.sum().backward()
    debug.sum().backward()
    print(f"forward error is: {(res - debug).mean()}")
    print(f"backward error is: {(conv.conv_11_weight_param.grad - conv_debug.conv_11_weight_buffer.grad).mean()}")
    # x_ = nn.Parameter(torch.tensor(1.0), requires_grad=False)
    # w_ = nn.Parameter(torch.tensor(2.0))
    # x = x_.data.requires_grad_()
    # w = w_.data.requires_grad_()
    # y = w * x
    # optimizer = SGD([x_, w_], lr=0.01)
    # y.backward()
    # x_.grad = x.grad
    # w_.grad = w.grad
    # optimizer.step()
    # print(x_.grad)
