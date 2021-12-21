import warnings
import argparse
from builder import build_models
import tasks
from mmcv import Config
from mmcv.runner import load_checkpoint
from general_datasets import Compose
import os
import mmcv
import numpy as np
# TODO 优化import逻辑
from plugins import *


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


def iou_with_sigmoid_pra(sigmoid, targets, eps=1e-6):
    """
    sigmoid: (torch.float32) shape (N, 1, H, W)
    targets: (torch.float32) shape (N, H, W), value {0,1}
    """
    pred = sigmoid
    targets = targets / 255
    inter = (pred * targets).sum(dim=(2, 3))
    union = (pred + targets).sum(dim=(2, 3))
    wiou = (inter + 1) / (union - inter + 1)
    return wiou.mean()

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
    Segmentor = build_models(cfg.model)
    if args.fuse_conv_bn:
        Segmentor = fuse_conv_bn(Segmentor)

    checkpoint = load_checkpoint(Segmentor, args.checkpoint, map_location='cpu')

    if not os.path.isdir(args.out_file):
        res = input(f"{args.out_file} doesn't exist! Do you want to create it? yes/no")
        if res.startswith('y'):
            os.mkdir(args.out_file)
        else:
            raise FileNotFoundError

    test_pipelines = Compose(cfg.data.test.pipeline)
    assert os.path.isdir(args.out_file)
    img_file_list = os.listdir(args.img_file)
    # TODO 改变no_grad的位置
    with torch.no_grad():
        Segmentor.eval()
        for img_name in img_file_list:
            if not img_name[-4:] not in ["jpg", "png", "bmp"]:
                warnings.warn(f"{img_name} is not a image name")
                continue
            img_path = os.path.join(args.img_file, img_name)
            out_path = os.path.join(args.out_file, img_name[:-4] + '.png')

            saved_img = mmcv.imread(img_path)
            data = dict(img_info=dict(filename=img_path), img_prefix=None)
            data = test_pipelines(data)
            data['img'] = data['img'].unsqueeze(dim=0)
            mask = Segmentor.forward_test(**data, demo=True)['pred_mask']
            mask = mask[0][0]
            mask = mask.detach().cpu().numpy()
            mmcv.imwrite(mask * 255, out_path)




