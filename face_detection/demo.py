import warnings
import argparse

from builder import build_detector
from mmcv import Config
import torch
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import load_checkpoint
from general_datasets import Compose
import os
import mmcv
# TODO 优化import逻辑
import face_detection
import backbone
import plugins

def parse_args():
    parser = argparse.ArgumentParser(
        description='face detection test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('img_file', help='Image file')
    parser.add_argument('out_file', help='output file')
    parser.add_argument('--cfg_options', default=None, help='output file')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    detector = build_detector(cfg.model)
    if args.fuse_conv_bn:
        detector = fuse_conv_bn(detector)
    checkpoint = load_checkpoint(detector, args.checkpoint, map_location='cpu')

    if not os.path.isdir(args.out_file):
        res = input(f"{args.out_file} doesn't exist! Do you want to create it? yes/no")
        if res.startswith('y'):
            os.mkdir(args.out_file)
        else:
            raise FileNotFoundError

    test_pipelines = Compose(cfg.data.test.pipeline)
    assert os.path.isdir(args.out_file)
    img_file_list = os.listdir(args.img_file)
    for img_name in img_file_list:
        if not img_name[-4:] not in ["jpg", "png", "bmp"]:
            warnings.warn(f"{img_name} is not a image name")
            continue
        img_path = os.path.join(args.img_file, img_name)
        out_path = os.path.join(args.out_file, img_name)

        saved_img = mmcv.imread(img_path)
        data = dict(img_info=dict(filename=img_path), img_prefix=None)
        data = test_pipelines(data)
        data['img'] = (data['img'] - 127.5) / 128
        data['img'] = data['img'].unsqueeze(dim=0)
        results = detector.forward_test(**data, demo=True)
        detector.draw_bboxes(saved_img, results["bboxes"], results["score"])
        detector.draw_landmarks(saved_img, results["landmarks"])
        mmcv.imwrite(saved_img, out_path)



