import torch.nn as nn
from abc import abstractmethod
from mmcv.cnn import (constant_init, is_norm, normal_init)

@HEAD.register_module()
class BaseHead(nn.Module):
    @abstractmethod
    def loss(self):
        return NotImplemented

    @abstractmethod
    def get_bboxes(self):
        return NotImplemented

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)
