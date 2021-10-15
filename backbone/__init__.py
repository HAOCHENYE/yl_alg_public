from .general_sand_net import *
from .vggnet import *
from .mobilenetv1 import *


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)
