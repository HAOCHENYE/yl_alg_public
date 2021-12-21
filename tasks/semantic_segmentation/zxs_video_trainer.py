import warnings
from builder import build_models
from tasks import TRAINER
from general_datasets import build_dataset, build_val_dataloader, build_train_dataloader
from util import get_root_logger, YLDistributedDataParallel, YLDataParallel
import torch
from mmcv.runner import (HOOKS, EpochBasedRunner, Fp16OptimizerHook,
                         OptimizerHook, build_optimizer, DistSamplerSeedHook,
                         build_runner)
from mmcv.utils import build_from_cfg
from custom_hook import EvalHook, DistEvalHook
from mmcv.runner import load_checkpoint
from mmcv import Config


@TRAINER.register_module()
class HumanVideoSegTrain(object):
    def __init__(self,
                 cfg,
                 img_cfg,
                 video_cfg,
                 distributed,
                 validate,
                 timestamp=None,
                 meta=None,):
        self.cfg = cfg
        self.distributed = distributed
        self.validate = validate
        self.timestamp = timestamp
        self.meta = meta
        self.video_cfg = Config(video_cfg)
        self.img_cfg = Config(img_cfg)

    def train(self):
        video_cfg = self.video_cfg
        img_cfg = self.img_cfg

        detector = build_models(self.cfg.model)
        detector.init_weights()

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

        if img_cfg.enabled:
            if img_cfg.load_from:
                load_checkpoint(model, img_cfg.load_from)
            self.run(model, img_cfg)

        if video_cfg.enabled:
            if video_cfg.load_from:
                # 加载3通道预训练模型，如果是联合训练，不需要加载。
                load_checkpoint(model, video_cfg.load_from)
            # 将加载权重，或者未加载权重的3通道模型，转换成4通道模型
            model.module.video = True
            model.module.load_video_weight()
            self.run(model, video_cfg)

    def run(self, model, mode_cfg):
        logger = get_root_logger(log_level=self.cfg.log_level)
        train_dataset = build_dataset(mode_cfg.data.train.dataset)
        runner_type = mode_cfg.get(mode_cfg, 'EpochBasedRunner')
        data_loaders = [build_train_dataloader(train_dataset,
                                               train_cfg=mode_cfg.data.train,
                                               dist=self.distributed,
                                               seed=self.cfg.seed,
                                               runner_type=runner_type)]
        optimizer = build_optimizer(model, mode_cfg.optimizer)

        if 'runner' not in mode_cfg:
            mode_cfg.runner = {
                'type': runner_type,
                'max_epochs': mode_cfg.total_epochs
            }
            warnings.warn(
                'config is now expected to have a `runner` section, '
                'please set `runner` in your config.', UserWarning)

        runner = build_runner(
            mode_cfg.runner,
            default_args=dict(
                model=model,
                optimizer=optimizer,
                work_dir=self.cfg.work_dir,
                logger=logger,
                meta=self.meta))

        runner.timestamp = self.timestamp
        fp16_cfg = self.cfg.get('fp16', None)
        if fp16_cfg is not None:
            optimizer_config = Fp16OptimizerHook(
                **mode_cfg.optimizer_config, **fp16_cfg, distributed=self.distributed)
        elif self.distributed and 'type' not in mode_cfg.optimizer_config:
            optimizer_config = OptimizerHook(**mode_cfg.optimizer_config)
        else:
            optimizer_config = mode_cfg.optimizer_config

        # register hooks
        runner.register_training_hooks(mode_cfg.lr_config, optimizer_config,
                                       self.cfg.checkpoint_config, self.cfg.log_config,
                                       self.cfg.get('momentum_config', None))

        if self.distributed:
            if isinstance(runner, EpochBasedRunner):
                runner.register_hook(DistSamplerSeedHook())

        # register eval hooks
        if self.validate:
            # Support batch_size > 1 in validation
            val_dataset = build_dataset(mode_cfg.data.val.dataset)
            val_dataloader = build_val_dataloader(
                val_dataset,
                batch_size=mode_cfg.data.val.batch_size,
                dist=self.distributed)
            eval_cfg = self.cfg.get('evaluation', {})
            eval_cfg['by_epoch'] = mode_cfg.runner['type'] != 'IterBasedRunner'

            eval_hook = DistEvalHook if self.distributed else EvalHook
            # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
            # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
            self.cfg.evaluator['dataset'] = val_dataset
            runner.register_hook(
                eval_hook(val_dataloader, self.cfg.evaluator, **eval_cfg), priority='LOW')
        #
        # # user-defined hooks
        if self.cfg.get('custom_hooks', None):
            custom_hooks = self.cfg.custom_hooks
            assert isinstance(custom_hooks, list), \
                f'custom_hooks expect list type, but got {type(custom_hooks)}'
            for hook_cfg in self.cfg.custom_hooks:
                assert isinstance(hook_cfg, dict), \
                    'Each item in custom_hooks expects dict type, but got ' \
                    f'{type(hook_cfg)}'
                hook_cfg = hook_cfg.copy()
                priority = hook_cfg.pop('priority', 'NORMAL')
                hook = build_from_cfg(hook_cfg, HOOKS)
                runner.register_hook(hook, priority=priority)

        if mode_cfg.resume_from:
            runner.resume(mode_cfg.resume_from)
        runner.run(data_loaders, self.cfg.workflow)


