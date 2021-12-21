import torch.nn.functional as F
import torch
import torch.nn as nn


def test_normal_conv2mm():
    batch = 5
    in_channel = 64
    out_channel = 64
    width = 224
    height = 224
    feat = torch.randn(batch, in_channel, height, width)
    weight = torch.randn(in_channel, in_channel, 1, 1)
    res_conv = F.conv2d(feat, weight)
    feat_ = feat.permute(0, 2, 3, 1).view(batch, -1, in_channel)
    weight_ = weight.permute(1, 0, 2, 3).view(1, -1, out_channel).expand(batch, in_channel, out_channel)
    res_bmm = torch.bmm(feat_, weight_)
    res_bmm = res_bmm.view(batch, height, width, out_channel).permute(0, 3, 1, 2)
    print((res_bmm - res_conv).abs().sum())


def test_group_conv2mm():
    batch = 5
    in_channel = 64
    out_channel = 64
    width = 3
    height = 3
    groups = 8
    group_channels = 64 // groups
    feat = torch.randn(batch, in_channel, height, width)
    weight = torch.randn(out_channel, in_channel // groups, 1, 1)
    res_conv = F.conv2d(feat, weight, groups=groups)

    feat = feat.permute(0, 2, 3, 1).reshape(batch, width * height, groups, -1).permute(0, 2, 1, 3).view(-1, width*height, group_channels)
    weight_ = weight.view(groups, group_channels, group_channels).permute(0, 2, 1).repeat(batch, 1, 1)
    res_bmm = torch.bmm(feat, weight_)
    res_bmm = res_bmm.view(batch, groups, height, width, group_channels).permute(0, 1, 4, 2, 3).reshape(batch, out_channel, height, width)
    # feat = feat.permute(0, 2, 3, 1).reshape(batch, width*height, groups, -1)
    # res_bmm = []
    # for i in range(groups):
    #     weight_ = weight[i*group_channels:(i+1)*group_channels, ...].permute(1, 0, 2, 3).view(1, -1, group_channels).expand(batch, group_channels, group_channels)
    #     feat_ = feat[..., i:i+1, :].view(batch, -1, group_channels)
    #     res_bmm.append(torch.bmm(feat_, weight_))
    # res_bmm = torch.cat(res_bmm, dim=2)
    # res_bmm = res_bmm.view(batch, height, width, out_channel).permute(0, 3, 1, 2)
    print((res_bmm - res_conv).abs().sum())


def test_group_conv_fuse():
    batch = 5
    in_channel = 64
    out_channel = 64
    width = 3
    height = 3
    groups = 8
    groups_channels = in_channel // groups
    out_groups = out_channel // groups
    kernel_size = 3
    feat = torch.rand(batch, in_channel, height, width)
    conv1x1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=None, groups=groups, padding=1)
    conv3x3 = nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, bias=None, groups=groups)
    conv_seq = nn.Sequential()
    conv_seq.add_module("conv1x1", conv1x1)
    conv_seq.add_module("conv3x3", conv3x3)
    conv_seq.eval()
    weight3x3 = conv3x3.weight.data.requires_grad_()
    weight1x1 = conv1x1.weight.data.requires_grad_()
    weight3x3_ = weight3x3.reshape(groups, out_groups, groups_channels, -1).permute(0, 1, 3, 2) #groups, out_groups, hw, groups_channels
    weight1x1_ = weight1x1.reshape(groups, 1, out_groups, groups_channels).repeat(1, out_groups, 1, 1)
    weight_fused = torch.matmul(weight3x3_, weight1x1_).permute(0, 1, 3, 2).reshape(groups*out_groups, groups_channels, kernel_size, kernel_size)
    res_seq = conv_seq(feat)
    res_fused = F.conv2d(feat, weight_fused, groups=groups, padding=1)
    print(f"convfused error: {(res_seq - res_fused).mean().item()}")

def test_group_conv_bias_fuse():
    batch = 5
    in_channel = 128
    out_channel = 128
    width = 3
    height = 3
    groups = 128
    groups_channels_1x1 = in_channel // groups
    groups_channels = out_channel // groups
    out_groups = out_channel // groups
    kernel_size = 3
    feat = torch.rand(batch, in_channel, height, width)
    conv1x1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=groups, padding=1)
    conv3x3 = nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, groups=groups)
    conv_seq = nn.Sequential()
    conv_seq.add_module("conv1x1", conv1x1)
    conv_seq.add_module("conv3x3", conv3x3)
    conv_seq.eval()
    weight3x3 = conv3x3.weight.data.requires_grad_()
    weight1x1 = conv1x1.weight.data.requires_grad_()
    bias3x3 = conv3x3.bias.data.requires_grad_()
    bias1x1 = conv1x1.bias.data.requires_grad_()
    weight3x3_ = weight3x3.reshape(groups, out_groups, groups_channels, -1).permute(0, 1, 3, 2) #groups, out_groups, hw, groups_channels
    weight1x1_ = weight1x1.reshape(groups, 1, out_groups, groups_channels_1x1).repeat(1, out_groups, 1, 1)
    bias1x1_ = bias1x1.reshape(groups, 1, 1, groups_channels)
    fused_bias = (bias1x1_ * weight3x3_).sum((2, 3)).view(-1) + bias3x3

    weight_fused = torch.matmul(weight3x3_, weight1x1_).permute(0, 1, 3, 2).reshape(groups*out_groups, groups_channels_1x1, kernel_size, kernel_size)
    res_seq = conv_seq(feat)
    res_fused = F.conv2d(feat, weight_fused, bias=fused_bias, groups=groups, padding=1)
    print(f"convfused error: {(res_seq - res_fused).mean().item()}")





if __name__ == '__main__':
    x = torch.randn(5, 4, 8, 8).requires_grad_()
    w = x.permute(0, 2, 3, 1).reshape(5, 128, 2)
    y = 10*w
    y.sum().backward()
    test_normal_conv2mm()
    test_group_conv2mm()
    test_group_conv_fuse()
    test_group_conv_bias_fuse()