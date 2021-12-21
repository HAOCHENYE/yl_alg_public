from builder import build_models
# TODO better import logic
from general_datasets import build_dataset, build_val_dataloader
from util import YLDistributedDataParallel, YLDataParallel
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, load_checkpoint)
from test_api import single_gpu_test, multi_gpu_test
from custom_hook import build_evaluator
from builder import TESTER
import torch
from mmcv.utils import Config
# def collate_fn(batch):
#     return batch


# TODO support multi-gpu test
@TESTER.register_module()
class HumanVideoSegTest(object):
    def __init__(self,
                 cfg,
                 img_cfg,
                 video_cfg,
                 distributed,
                 validate,
                 *args,
                 **kwargs):

        self.cfg = cfg
        self.distributed = distributed
        self.validate = validate
        self.args = args
        self.video_cfg = Config(video_cfg)
        self.img_cfg = Config(img_cfg)

    def test(self):
        model = build_models(self.cfg.model)
        model.init_weights()
        if self.video_cfg.enabled:
            model.video = True
            model.load_video_weight()
            self.run(model, self.video_cfg)
        else:
            self.run(model, self.img_cfg)

    def run(self, model, mode_cfg):
        rank, _ = get_dist_info()
        val_dataset = build_dataset(mode_cfg.data.val.dataset)
        val_dataloader = build_val_dataloader(
            val_dataset,
            batch_size=mode_cfg.data.val.batch_size,
            num_workers=mode_cfg.data.val.num_workers,
            dist=self.distributed)
        load_checkpoint(model, self.cfg.checkpoint, map_location='cpu')
        if self.cfg.fuse_conv_bn:
            model = fuse_conv_bn(model)
        if self.distributed:
            find_unused_parameters = self.cfg.get('find_unused_parameters', False)
            model = YLDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = YLDataParallel(
                model.cuda(
                    self.cfg.gpu_ids[0]),
                device_ids=[0])
        evaluator = build_evaluator(self.cfg.evaluator)
        if not self.distributed:
            outputs = single_gpu_test(
                model,
                val_dataloader,
                self.cfg.evaluator.get(
                    'compose_type', 'append'),
                self.cfg.evaluator.get('data_type'))
        else:
            outputs = multi_gpu_test(model,
                                     val_dataloader,
                                     self.cfg.evaluator.get(
                                         'compose_type', 'append'),
                                     self.cfg.evaluator.get('data_type'))
        if rank == 0:
            evaluator.evaluate(None, outputs)
