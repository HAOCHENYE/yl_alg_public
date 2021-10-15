from builder import build_detector
# TODO better import logic
from tasks import face_detection
from general_datasets import build_dataset, build_val_dataloader
from util import get_root_logger, YLDistributedDataParallel, YLDataParallel
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from test_api import single_gpu_test, multi_gpu_test
from custom_hook import build_evaluator
from builder import TESTER
from util import tensor2numpy

# def collate_fn(batch):
#     return batch


# TODO support multi-gpu test
@TESTER.register_module()
class FaceDetectTest(object):
    def __init__(self, cfg, distributed, validate, *args, **kwargs):
        self.cfg = cfg
        self.distributed = distributed
        self.validate = validate
        self.args = args

    def test(self):
        rank, _ = get_dist_info()
        detector = build_detector(self.cfg.model)
        detector.init_weights()
        val_dataset = build_dataset(self.cfg.data.test.dataset)
        val_dataloader = build_val_dataloader(
            val_dataset,
            batch_size=self.cfg.data.test.batch_size,
            dist=self.distributed)
        load_checkpoint(detector, self.cfg.checkpoint, map_location='cpu')
        if self.cfg.fuse_conv_bn:
            detector = fuse_conv_bn(detector)
        model = YLDataParallel(
            detector.cuda(
                self.cfg.gpu_ids[0]),
            device_ids=[0])
        self.cfg.evaluator['dataset'] = val_dataset
        evaluator = build_evaluator(self.cfg.evaluator)
        if not self.distributed:
            outputs = single_gpu_test(
                model, val_dataloader, self.cfg.evaluator.get(
                    'compose_type', 'append'))
        else:
            outputs = multi_gpu_test(model, val_dataloader)
        if rank == 0:
            evaluator.evaluate(None, outputs)
