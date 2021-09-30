from ..builder import PIPELINES
import mmcv
import numpy as np
import torch

@PIPELINES.register_module()
class Normalize:
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        img = results["img"]
        if self.to_rgb:
            img = img[..., ::-1]
        results["img"] = img
        results['img_norm_cfg'] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class ResizeImage:
    def __init__(self,
                 img_scale=None,
                 keep_ratio=True,
                 backend='cv2',
                 pad_val=0):

        self.img_scale = img_scale
        self.backend = backend
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.pad_val = pad_val

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``, \
                where ``img_scale`` is the selected image scale and \
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        img = results["img"]
        scale = results["scale"]

        if self.keep_ratio:
            pad_shape = list(img.shape)
            # scale: (w, h), pad_shape: (h, w, c)
            pad_shape[:2] = scale[::-1]
            res_img = np.full(pad_shape, self.pad_val, dtype=np.uint8)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_img, scale_factor = mmcv.imrescale(
                img,
                scale,
                return_scale=True,
                backend=self.backend)
            new_h, new_w = new_img.shape[:2]
            h, w = img.shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
            res_img[:new_h, :new_w, ...] = new_img
            resized_shape = new_img.shape
        else:
            res_img, w_scale, h_scale = mmcv.imresize(
                img,
                scale,
                return_scale=True,
                backend=self.backend)
            resized_shape = res_img.shape
        # shape h w c
        results["resized_shape"] = torch.Tensor(resized_shape)
        results["img"] = res_img
        scale_factor = np.array([w_scale, h_scale],
                                dtype=np.float32)
        results['img_shape'] = torch.Tensor(res_img.shape)
        # in case that there is no padding
        results['pad_shape'] = torch.Tensor(res_img.shape)
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['img'].shape[:2]
                scale_factor = results['scale_factor']
                assert isinstance(scale_factor, float)
                results['scale'] = tuple(
                    [int(x * scale_factor) for x in img_shape][::-1])
            else:
                results['scale'] = self.img_scale

        self._resize_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        return repr_str
