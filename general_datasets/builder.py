from mmcv.utils import Registry, build_from_cfg

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')


def build_dataset(cfg):
    return DATASETS.build(cfg)
