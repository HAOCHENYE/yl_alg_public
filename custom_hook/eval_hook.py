# Copyright (c) OpenMMLab. All rights reserved.
from .builder import build_evaluator
import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from mmcv.utils import build_from_cfg
from mmcv.runner.hooks import HOOKS
from torch.nn.modules.batchnorm import _BatchNorm
from test_api import single_gpu_test, multi_gpu_test


@HOOKS.register_module()
class EvalHook(BaseEvalHook):
    def __init__(self, dataloader, evaluator, *args, **kwargs):
        super().__init__(dataloader, *args, **kwargs)
        self.evaluator = build_evaluator(evaluator)
        self.cat_type = evaluator.get('cat_type', 'append')

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        results = single_gpu_test(runner.model, self.dataloader, self.cat_type)
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluator.evaluate(runner, results)
        for key, value in key_score.items():
            runner.log_buffer.output[key] = value
        runner.log_buffer.ready = True
        if self.save_best:
            self._save_ckpt(runner, key_score)


@HOOKS.register_module()
class DistEvalHook(BaseDistEvalHook):
    def __init__(self, dataloader, evaluator, *args, **kwargs):
        super().__init__(dataloader, *args, **kwargs)
        self.evaluator = build_evaluator(evaluator)
        self.cat_type = evaluator.get('cat_type', 'append')

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            self.cat_type)
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluator.evaluate(runner, results)
            for key, value in key_score.items():
                runner.log_buffer.output[key] = value
            runner.log_buffer.ready = True

            if self.save_best:
                self._save_ckpt(runner, key_score)
