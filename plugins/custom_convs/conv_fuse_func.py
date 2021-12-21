import torch
import torch.nn.functional as F
import torch.nn as nn
nn.Parameter

def fuse_kxk_1x1(conv_kxk:nn.Conv2d,
                 conv_1x1:nn.Conv2d):
    weight_kxk = torch.Tensor(conv_kxk.weight)
    weight_1x1 = torch.Tensor(conv_1x1.weight)
    weight_fuse = F.conv2d(weight_kxk.to(torch.float64),
                           weight_1x1.permute(1, 0, 2, 3).to(torch.float64),
                           bias=None,
                           stride=1,
                           padding=0,
                           dilation=1,
                           groups=1)
    return weight_fuse


def inference_kx1_1x1(input: torch.Tensor,
                      conv_kxk: nn.Conv2d,
                      conv_1x1: nn.Conv2d):
    kernel_size = conv_kxk.weight.shape[2]
    padding = 0
    conv_kxk.eval()
    conv_1x1.eval()
    weight_fuse = fuse_kxk_1x1(conv_kxk, conv_1x1)

    test_res = F.conv2d(input, weight_fuse.to(torch.float32), bias=None, stride=1, padding=padding)
    train_res = conv_kxk(conv_1x1(input))
    error = train_res - test_res
    print(f"error is : {error.abs().sum().item()}")
    return


if __name__ == "__main__":
    channel_in = 64
    channel_out = 64
    feat_width = 224
    feat_height = 224
    kernel_size = 3
    conv_3x3 = nn.Conv2d(channel_in,
                         channel_out,
                         kernel_size=kernel_size,
                         padding=0,
                         bias=None)
    conv_1x1 = nn.Conv2d(channel_in,
                         channel_out,
                         kernel_size=1,
                         padding=0,
                         bias=None)
    x = torch.randn(1, channel_in, feat_height, feat_height)
    inference_kx1_1x1(x, conv_3x3, conv_1x1)