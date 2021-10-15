from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry
from .base_models import BaseModels

TRAINER = Registry("trainer")
TESTER = Registry('Tester')

MODELS = Registry('models', parent=MMCV_MODELS)
BACKBONES = MODELS
DETECTOR = MODELS
NECK = MODELS
HEAD = MODELS


def build_head(cfg):
    return HEAD.build(cfg)


def build_neck(cfg):
    return NECK.build(cfg)


def build_detector(cfg):
    return DETECTOR.build(cfg)


__all__ = ["BaseModels", "MODELS", "TRAINER", "TESTER",
           "BACKBONES", "NECK", "HEAD", "DETECTOR",
           "build_head", "build_neck", "build_detector"]


