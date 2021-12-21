import argparse
import warnings
import mmcv
from mmcv import DictAction
from mmcv.runner import load_checkpoint
from builder import build_models
from tasks import build_trainer
from plugins import *
import torch
from onnxsim import simplify
import onnx


@torch.no_grad()
def recursive_fuse_conv(module, prefix=''):
    for name, child in module._modules.items():
        if not hasattr(child, 'fuse_conv'):
            recursive_fuse_conv(child, prefix + name + '.')
        else:
            child.fuse_conv()


def pytorch2onnx(model,
                 input_shape,
                 output_names,
                 input_names,
                 opset_version=11,
                 output_file='tmp.onnx',
                 ):
    model.forward = model.export_onnx
    dummy_input = torch.zeros(input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        output_file,
        input_names=input_names,
        output_names=output_names,
        keep_initializers_as_inputs=True,
        do_constant_folding=True,
        opset_version=opset_version)


def parse_normalize_cfg(test_pipeline):
    transforms = None
    for pipeline in test_pipeline:
        if 'transforms' in pipeline:
            transforms = pipeline['transforms']
            break
    assert transforms is not None, 'Failed to find `transforms`'
    norm_config_li = [_ for _ in transforms if _['type'] == 'Normalize']
    assert len(norm_config_li) == 1, '`norm_config` should only have one'
    norm_config = norm_config_li[0]
    return norm_config


def parse_args():
    parser = argparse.ArgumentParser(description='Convert MMDetection models to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument('--output-names', type=str, nargs='+', default=["score", "bbox"],help='output names of network')
    parser.add_argument('--input-names', type=str, nargs='+', default=["data"], help='input names of network')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[800, 1216],
        help='input image size')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--dynamic-export',
        action='store_true',
        help='Whether to export onnx with dynamic axis.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    warnings.warn('Arguments like `--mean`, `--std`, `--dataset` would be \
        parsed directly from config file and are deprecated and \
        will be removed in future releases.')

    assert args.opset_version == 11, 'MMDet only support opset 11 now'

    try:
        from mmcv.onnx.symbolic import register_extra_symbolics
    except ModuleNotFoundError:
        raise NotImplementedError('please update mmcv to version>=v1.0.4')
    register_extra_symbolics(args.opset_version)

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    model = build_models(cfg.model)
    model.eval()
    # torch.save(model.state_dict(), f"{args.output_file[:-4]}.pth")
    try:
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    except IOError:
        res = input("checkpoint doesn't exist, still export onnx with random init model? yes/no:")
        if res == "yes" or "y":
            pass
    recursive_fuse_conv(model)
    input_shape = args.shape
    # convert model to onnx file
    pytorch2onnx(
        model,
        args.shape,
        args.output_names,
        args.input_names,
        output_file=args.output_file)

    model_opt, check_ok = simplify(args.output_file,
                                   3,
                                   True,
                                   False,
                                   dict(),
                                   None,
                                   False,
                                   )
    onnx.save(model_opt, args.output_file)

