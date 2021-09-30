from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry
from .base_models import BaseModels

MODELS = Registry('models', parent=MMCV_MODELS)
BACKBONES = MODELS
DETECTOR = MODELS
NECK = MODELS
HEAD = MODELS

TRAINER = Registry("trainer")


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_trainer(cfg_trainer, **kwargs):
    for key, value in kwargs.items():
        cfg_trainer[key] = value
    return TRAINER.build(cfg_trainer)


def build_head(cfg):
    return HEAD.build(cfg)


def build_neck(cfg):
    return NECK.build(cfg)


def build_detector(cfg):
    return DETECTOR.build(cfg)


__all__ = ["BaseModels", "MODELS", "TRAINER", "build_trainer",
           "BACKBONES", "NECK", "HEAD", "DETECTOR", "build_backbone",
           "build_head", "build_neck", "build_detector"]


