from builder import build_detector
from builder import TRAINER
from general_datasets import build_dataset, build_val_dataloader
from util import get_root_logger, YLDistributedDataParallel, YLDataParallel
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from face_detection.utils import single_gpu_test


# TODO support multi-gpu test
@TRAINER.register_module()
class FaceDetectTrain(object):
    def __init__(self, cfg, distributed, validate, args):
        self.cfg = cfg
        self.distributed = distributed
        self.validate = validate
        self.args = args

    def train(self):
        detector = build_detector(self.cfg.model)
        detector.init_weights()
        val_dataset = build_dataset(self.cfg.data.train.dataset)
        val_dataloader = [build_val_dataloader(val_dataset, self.cfg.data.train, self.distributed, self.cfg.seed)]
        load_checkpoint(detector, self.args.checkpoint, map_location='cpu')
        if self.args.fuse_conv_bn:
            detector = fuse_conv_bn(detector)
        model = YLDataParallel(detector.cuda(self.cfg.gpu_ids[0]), device_ids=[0])
        outputs = single_gpu_test(model, val_dataloader)
        val_dataset.evaluate(outputs)

