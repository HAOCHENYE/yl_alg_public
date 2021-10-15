from mmcv.utils import Registry
from builder import TESTER, TRAINER
from .face_detection import *




def build_trainer(cfg_trainer, **kwargs):
    for key, value in kwargs.items():
        cfg_trainer[key] = value
    return TRAINER.build(cfg_trainer)


def build_tester(cfg_tester, **kwargs):
    for key, value in kwargs.items():
        cfg_tester[key] = value
    return TESTER.build(cfg_tester)

