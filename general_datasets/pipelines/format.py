from ..builder import PIPELINES
import numpy as np
import torch
from collections.abc import Sequence
import mmcv


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@PIPELINES.register_module()
class Formatting(object):
    def __init__(self,
                 pad_cfg=dict(key=[], pad_num=500),
                 collect_key=[],
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor',
                            'img_norm_cfg', 'resized_shape')
                 ):

        self.pad_key = pad_cfg["key"]
        self.pad_num = pad_cfg["pad_num"]
        self.collect_key = collect_key
        self.meta_keys = meta_keys


    def _add_default_meta_keys(self, results):
        """Add default meta keys.

        We set default meta keys including `pad_shape`, `scale_factor` and
        `img_norm_cfg` to avoid the case where no `Resize`, `Normalize` and
        `Pad` are implemented during the whole pipeline.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            results (dict): Updated result dict contains the data to convert.
        """
        img = results['img']
        results.setdefault('pad_shape', img.shape)
        results.setdefault('scale_factor', 1.0)
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results.setdefault(
            'img_norm_cfg',
            dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False))
        return results

    def __call__(self, results):
        for key in results["img_fields"]:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            results[key] = (to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))).contiguous()

        if self.pad_key is not None:
            # val shape should be [n, k],n will be padded to pad_num
            for key in self.pad_key:
                val = results[key]
                pad_shape = list(val.shape)
                num_results = val.shape[0]
                # TODO 优化
                results["gt_num"] = num_results
                pad_shape[0] = self.pad_num
                pad_result = np.zeros(pad_shape)
                pad_result[:num_results, ...] = val

                results[key] = pad_result

        results = self._add_default_meta_keys(results)

        meta_dict = dict()
        if self.pad_key and self.pad_num:
            meta_dict['gt_num'] = results['gt_num']
        for key in self.meta_keys:
            meta_dict[key] = results[key]

        data = {}
        for key in self.collect_key:
            data[key] = results[key]

        data["img_metas"] = meta_dict
        return data


