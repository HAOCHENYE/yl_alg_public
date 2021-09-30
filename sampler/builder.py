from mmcv.utils import Registry

SAMPLER = Registry('sampler')
BATCH_SAMPLER = Registry('batch_sampler')
INDEX_DECODER = Registry('index_decoder')


def build_sampler(cfg):
    return SAMPLER.build(cfg)


def build_index_coder(cfg):
    return INDEX_DECODER.build(cfg)


def build_batch_sampler(cfg):
    return BATCH_SAMPLER.build(cfg)
