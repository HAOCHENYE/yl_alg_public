# Copyright (c) OpenMMLab. All rights reserved.
from sampler import BATCH_SAMPLER
import random
from functools import partial
import numpy as np
from mmcv.runner import get_dist_info
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data import DistributedSampler
from util import get_default
from sampler import build_batch_sampler
from sampler import InfiniteSampler

def build_train_dataloader(dataset,
                           train_cfg,
                           dist=True,
                           seed=None,
                           runner_type='EpochBasedRunner'):
    batch_size = get_default(train_cfg, "batch_size", 2)
    # TODO 优化batchsampler实例化的方式
    batch_sampler_cfg = get_default(train_cfg, "batch_sampler_cfg",
                                    dict(type="YoloBatchSampler", drop_last=False, input_dimension=(640, 640)))
    num_workers = get_default(train_cfg, "num_workers", 2)

    rank, world_size = get_dist_info()
    batch_sampler_cfg["batch_size"] = batch_size
    if not dist:
        if runner_type == 'EpochBasedRunner':
            sampler = RandomSampler(dataset)
        else:
            sampler = InfiniteSampler(dataset)
        # sampler = SequentialSampler(dataset)

    else:
        if runner_type == 'EpochBasedRunner':
            sampler = DistributedSampler(dataset, rank=rank, seed=seed)
        else:
            sampler = InfiniteSampler(dataset, seed=seed, rank=rank, world_size=world_size)

    batch_sampler_cfg['sampler'] = sampler
    batch_sampler = build_batch_sampler(batch_sampler_cfg)

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=False,
        worker_init_fn=init_fn,
        persistent_workers=(num_workers != 0))

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_val_dataloader(dataset,
                         batch_size=2,
                         num_workers=1,
                         dist=False,
                         collate_fn=None):
    rank, world_size = get_dist_info()
    if dist:
        sampler = DistributedSampler(dataset, world_size, shuffle=False)
    else:
        sampler = SequentialSampler(dataset)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn)

    return data_loader
