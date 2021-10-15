from mmcv.utils import Registry


EVALUATOR = Registry("evaluator_hooks")


def build_evaluator(evaluator_cfg):
    return EVALUATOR.build(evaluator_cfg)
