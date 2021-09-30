
import warnings
from builder import build_detector
from builder import TRAINER
from general_datasets import build_dataset, build_val_dataloader, build_train_dataloader
from util import get_root_logger, YLDistributedDataParallel, YLDataParallel
import torch
from mmcv.runner import (HOOKS, EpochBasedRunner, Fp16OptimizerHook,
                         OptimizerHook, build_optimizer, DistSamplerSeedHook,
                         build_runner)


@TRAINER.register_module()
class FaceDetectTrain(object):
    def __init__(self, cfg, distributed, validate, timestamp=None, meta=None):
        self.cfg = cfg
        self.distributed = distributed
        self.validate = validate
        self.timestamp = timestamp
        self.meta = meta

    def train(self):
        detector = build_detector(self.cfg.model)
        detector.init_weights()
        train_dataset = build_dataset(self.cfg.data.train.dataset)
        logger = get_root_logger(log_level=self.cfg.log_level)
        data_loaders = [build_train_dataloader(train_dataset, self.cfg.data.train, self.distributed, self.cfg.seed)]

        # put model on gpus
        if self.distributed:
            find_unused_parameters = self.cfg.get('find_unused_parameters', False)
            # Sets the `find_unused_parameters` parameter in
            # torch.nn.parallel.DistributedDataParallel
            model = YLDistributedDataParallel(
                detector.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = YLDataParallel(detector.cuda(self.cfg.gpu_ids[0]), device_ids=self.cfg.gpu_ids)

        optimizer = build_optimizer(model, self.cfg.optimizer)

        if 'runner' not in self.cfg:
            self.cfg.runner = {
                'type': 'EpochBasedRunner',
                'max_epochs': self.cfg.total_epochs
            }
            warnings.warn(
                'config is now expected to have a `runner` section, '
                'please set `runner` in your config.', UserWarning)
        else:
            if 'total_epochs' in self.cfg:
                assert self.cfg.total_epochs == self.cfg.runner.max_epochs

        runner = build_runner(
            self.cfg.runner,
            default_args=dict(
                model=model,
                optimizer=optimizer,
                work_dir=self.cfg.work_dir,
                logger=logger,
                meta=self.meta))

        fp16_cfg = self.cfg.get('fp16', None)
        if fp16_cfg is not None:
            optimizer_config = Fp16OptimizerHook(
                **self.cfg.optimizer_config, **fp16_cfg, distributed=self.distributed)
        elif self.distributed and 'type' not in self.cfg.optimizer_config:
            optimizer_config = OptimizerHook(**self.cfg.optimizer_config)
        else:
            optimizer_config = self.cfg.optimizer_config

        # register hooks
        runner.register_training_hooks(self.cfg.lr_config, optimizer_config,
                                       self.cfg.checkpoint_config, self.cfg.log_config,
                                       self.cfg.get('momentum_config', None))

        if self.distributed:
            if isinstance(runner, EpochBasedRunner):
                runner.register_hook(DistSamplerSeedHook())

        # register eval hooks
        # if self.validate:
        #     # Support batch_size > 1 in validation
        #     val_samples_per_gpu = self.cfg.data.val.pop('samples_per_gpu', 1)
        #     if val_samples_per_gpu > 1:
        #         # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        #         self.cfg.data.val.pipeline = replace_ImageToTensor(
        #             self.cfg.data.val.pipeline)
        #     val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        #     val_dataloader = build_dataloader(
        #         val_dataset,
        #         samples_per_gpu=val_samples_per_gpu,
        #         workers_per_gpu=cfg.data.workers_per_gpu,
        #         dist=distributed,
        #         shuffle=False)
        #     eval_cfg = cfg.get('evaluation', {})
        #     eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        #     eval_hook = DistEvalHook if distributed else EvalHook
        #     # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        #     # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        #     runner.register_hook(
        #         eval_hook(val_dataloader, **eval_cfg), priority='LOW')
        #
        # # user-defined hooks
        # if cfg.get('custom_hooks', None):
        #     custom_hooks = cfg.custom_hooks
        #     assert isinstance(custom_hooks, list), \
        #         f'custom_hooks expect list type, but got {type(custom_hooks)}'
        #     for hook_cfg in cfg.custom_hooks:
        #         assert isinstance(hook_cfg, dict), \
        #             'Each item in custom_hooks expects dict type, but got ' \
        #             f'{type(hook_cfg)}'
        #         hook_cfg = hook_cfg.copy()
        #         priority = hook_cfg.pop('priority', 'NORMAL')
        #         hook = build_from_cfg(hook_cfg, HOOKS)
        #         runner.register_hook(hook, priority=priority)

        if self.cfg.resume_from:
            runner.resume(self.cfg.resume_from)
        elif self.cfg.load_from:
            runner.load_checkpoint(self.cfg.load_from)
        runner.run(data_loaders, self.cfg.workflow)




