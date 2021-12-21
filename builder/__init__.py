from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry
from .base_models import BaseModels

TRAINER = Registry("trainer")
TESTER = Registry('Tester')

MODELS = Registry('models', parent=MMCV_MODELS)
BACKBONES = MODELS
CUMSTOM_MODELS = MODELS
NECK = MODELS
HEAD = MODELS


def build_head(cfg):
    return HEAD.build(cfg)


def build_neck(cfg):
    return NECK.build(cfg)


def build_models(cfg):
    return CUMSTOM_MODELS.build(cfg)


__all__ = ["BaseModels", "MODELS", "TRAINER", "TESTER",
           "BACKBONES", "NECK", "HEAD", "CUMSTOM_MODELS",
           "build_head", "build_neck", "CUMSTOM_MODELS"]


