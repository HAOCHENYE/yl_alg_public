import torch
import torch.nn as nn


class MyConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.conv.weight[:, :, 0, 0]
        self.conv1 = nn.Conv2d(3, 16, 3)

    def forward(self, x):
        self.conv.weight = nn.Parameter(self.conv1.weight + torch.ones(self.conv1.weight.shape))
        return self.conv(x)


if __name__ == '__main__':
    model = MyConv()
    model.eval()
    x = torch.rand(1, 3, 224, 224)
    torch.onnx.export(model, x, 'test.onnx', opset_version=11)