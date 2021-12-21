from ..builder import PIPELINES
import mmcv
import numpy as np
import torch
import copy
import cv2
import random
import warnings
from .compose import Compose
import math


@PIPELINES.register_module()
class Pad:
    """Pad the image & masks & segmentation map.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_to_square (bool): Whether to pad the image into a square.
            Currently only used for YOLOX. Default: False.
        pad_val (dict, optional): A dict for padding value, the default
            value is `dict(img=0, masks=0, seg=255)`.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_to_square=False,
                 pad_val=dict(img=0, masks=0, seg=255),
                 pad_keys=['img']):
        self.size = size
        self.size_divisor = size_divisor
        if isinstance(pad_val, float) or isinstance(pad_val, int):
            warnings.warn(
                'pad_val of float type is deprecated now, '
                f'please use pad_val=dict(img={pad_val}, '
                f'masks={pad_val}, seg=255) instead.', DeprecationWarning)
            pad_val = dict(img=pad_val, masks=pad_val, seg=255)
        assert isinstance(pad_val, dict)
        self.pad_val = pad_val
        self.pad_to_square = pad_to_square
        self.pad_keys = pad_keys
        if pad_to_square:
            assert size is None and size_divisor is None, \
                'The size and size_divisor must be None ' \
                'when pad2square is True'
        else:
            assert size is not None or size_divisor is not None, \
                'only one of size and size_divisor should be valid'
            assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        pad_val = self.pad_val.get('img', 0)
        for key in self.pad_keys:
            if self.pad_to_square:
                max_size = max(results[key].shape[:2])
                self.size = (max_size, max_size)
            if self.size is not None:
                padded_img = mmcv.impad(
                    results[key], shape=self.size, pad_val=pad_val)
            elif self.size_divisor is not None:
                padded_img = mmcv.impad_to_multiple(
                    results[key], self.size_divisor, pad_val=pad_val)
            results[key] = padded_img
        # TODO 区分padshape和imgshape
        results['pad_shape'] = padded_img.shape
        results['img_shape'] = np.array(padded_img.shape)
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_to_square={self.pad_to_square}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


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
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.to_rgb = to_rgb

    def __call__(self, results):
        img = results["img"]
        if self.to_rgb:
            img = img[..., ::-1]
        results["img"] = img
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)

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
                 pad_val=0,
                 resize_keys=['img'],
                 resizer_pipelines=[]):

        self.img_scale = img_scale
        self.backend = backend
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.pad_val = pad_val
        self.resizer_pipelines = Compose(resizer_pipelines)
        self.resize_keys = resize_keys
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
        for key in self.resize_keys:
            img = results[key]
            scale = results["scale"]

            if self.keep_ratio:
                pad_shape = list(img.shape)
                # scale: (w, h), pad_shape: (h, w, c)
                pad_shape[:2] = scale
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
            results[key] = res_img
        # shape h w c
            if key == 'img':
                results["resized_shape"] = np.array(resized_shape)
                scale_factor = np.array([w_scale, h_scale],
                                        dtype=np.float32)
                results['img_shape'] = np.array(res_img.shape)
                # in case that there is no padding
                results['pad_shape'] = np.array(res_img.shape)
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
        if 'batch_info' in results:
            if 'size' in results['batch_info']:
                results['scale'] = results['batch_info']['size']
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
        results = self.resizer_pipelines(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        return repr_str


@PIPELINES.register_module()
class RandomFlip:
    """Flip the image & bbox & mask.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    When random flip is enabled, ``flip_ratio``/``direction`` can either be a
    float/string or tuple of float/string. There are 3 flip modes:

    - ``flip_ratio`` is float, ``direction`` is string: the image will be
        ``direction``ly flipped with probability of ``flip_ratio`` .
        E.g., ``flip_ratio=0.5``, ``direction='horizontal'``,
        then image will be horizontally flipped with probability of 0.5.
    - ``flip_ratio`` is float, ``direction`` is list of string: the image wil
        be ``direction[i]``ly flipped with probability of
        ``flip_ratio/len(direction)``.
        E.g., ``flip_ratio=0.5``, ``direction=['horizontal', 'vertical']``,
        then image will be horizontally flipped with probability of 0.25,
        vertically with probability of 0.25.
    - ``flip_ratio`` is list of float, ``direction`` is list of string:
        given ``len(flip_ratio) == len(direction)``, the image wil
        be ``direction[i]``ly flipped with probability of ``flip_ratio[i]``.
        E.g., ``flip_ratio=[0.3, 0.5]``, ``direction=['horizontal',
        'vertical']``, then image will be horizontally flipped with probability
        of 0.3, vertically with probability of 0.5.

    Args:
        flip_ratio (float | list[float], optional): The flipping probability.
            Default: None.
        direction(str | list[str], optional): The flipping direction. Options
            are 'horizontal', 'vertical', 'diagonal'. Default: 'horizontal'.
            If input is a list, the length must equal ``flip_ratio``. Each
            element in ``flip_ratio`` indicates the flip probability of
            corresponding direction.
    """

    def __init__(self,
                 flip_ratio=None,
                 direction='horizontal',
                 flip_keys=['img'],
                 flip_pipelines=[]):
        if isinstance(flip_ratio, list):
            assert mmcv.is_list_of(flip_ratio, float)
            assert 0 <= sum(flip_ratio) <= 1
        elif isinstance(flip_ratio, float):
            assert 0 <= flip_ratio <= 1
        elif flip_ratio is None:
            pass
        else:
            raise ValueError('flip_ratios must be None, float, '
                             'or list of float')

        self.flip_pipelines = Compose(flip_pipelines)
        self.flip_ratio = flip_ratio

        valid_directions = ['horizontal', 'vertical', 'diagonal']
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
            assert mmcv.is_list_of(direction, str)
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError('direction must be either str or list of str')
        self.direction = direction
        self.flip_keys = flip_keys

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added \
                into result dict.
        """

        if 'flip' not in results:
            if isinstance(self.direction, list):
                # None means non-flip
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) -
                                                    1) + [non_flip_ratio]

            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)

            results['flip'] = cur_dir is not None
        if 'flip_direction' not in results:
            results['flip_direction'] = cur_dir
        if results['flip']:
            # flip image
            for key in self.flip_keys:
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction'])

        results = self.flip_pipelines(results)
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_ratio={self.flip_ratio})'


# TODO 考虑到crop存在失败的情况(bboxes或者landmarks切分不合理，因此只能以if else的形式去做crop)
@PIPELINES.register_module()
class RandomCrop:
    """Random crop the image & bboxes & masks.

    The absolute `crop_size` is sampled based on `crop_type` and `image_size`,
    then the cropped results are generated.

    Args:
        crop_size (tuple): The relative ratio or absolute pixels of
            height and width.
        crop_type (str, optional): one of "relative_range", "relative",
            "absolute", "absolute_range". "relative" randomly crops
            (h * crop_size[0], w * crop_size[1]) part from an input of size
            (h, w). "relative_range" uniformly samples relative crop size from
            range [crop_size[0], 1] and [crop_size[1], 1] for height and width
            respectively. "absolute" crops from an input with absolute size
            (crop_size[0], crop_size[1]). "absolute_range" uniformly samples
            crop_h in range [crop_size[0], min(h, crop_size[1])] and crop_w
            in range [crop_size[0], min(w, crop_size[1])]. Default "absolute".
        allow_negative_crop (bool, optional): Whether to allow a crop that does
            not contain any bbox area. Default False.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.

    Note:
        - If the image is smaller than the absolute crop size, return the
            original image.
        - The keys for bboxes, labels and masks must be aligned. That is,
          `gt_bboxes` corresponds to `gt_labels` and `gt_masks`, and
          `gt_bboxes_ignore` corresponds to `gt_labels_ignore` and
          `gt_masks_ignore`.
        - If the crop does not contain any gt-bbox region and
          `allow_negative_crop` is set to False, skip this image.
    """

    def __init__(self,
                 crop_size,
                 crop_type='absolute',
                 crop_keys=['img'],
                 crop_pipelines=[]):
        self.crop_pipelines = Compose(crop_pipelines)
        if crop_type not in [
                'relative_range', 'relative', 'absolute', 'absolute_range'
        ]:
            raise ValueError(f'Invalid crop_type {crop_type}.')
        if crop_type in ['absolute', 'absolute_range']:
            assert crop_size[0] > 0 and crop_size[1] > 0
            assert isinstance(crop_size[0], int) and isinstance(
                crop_size[1], int)
        else:
            assert 0 < crop_size[0] <= 1 and 0 < crop_size[1] <= 1
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.crop_keys = crop_keys
        # The key correspondence from bboxes to labels and masks.

    def _crop_data(self, results, crop_size):
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (tuple): Expected absolute size after cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area. Default to False.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        ori_results = copy.deepcopy(results)
        img = results['img']
        margin_h = max(img.shape[0] - crop_size[0], 0)
        margin_w = max(img.shape[1] - crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

        # crop the image
        for key in self.crop_keys:
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2, ...]

        img_shape = results['img'].shape
        # results['img'] = img
        results['img_shape'] = img_shape
        results['crop_roi'] = [crop_x1, crop_y1, crop_x2, crop_y2]
        results = self.crop_pipelines(results)
        if not self.crop_pipelines.transforms:
            return results
        if 'cropped' not in results or not results['cropped']:
            return ori_results
        return results

    def _get_crop_size(self, image_size):
        """Randomly generates the absolute crop size based on `crop_type` and
        `image_size`.

        Args:
            image_size (tuple): (h, w).

        Returns:
            crop_size (tuple): (crop_h, crop_w) in absolute pixels.
        """
        h, w = image_size
        if self.crop_type == 'absolute':
            return (min(self.crop_size[0], h), min(self.crop_size[1], w))
        elif self.crop_type == 'absolute_range':
            assert self.crop_size[0] <= self.crop_size[1]
            crop_h = np.random.randint(
                min(h, self.crop_size[0]),
                min(h, self.crop_size[1]) + 1)
            crop_w = np.random.randint(
                min(w, self.crop_size[0]),
                min(w, self.crop_size[1]) + 1)
            return crop_h, crop_w
        elif self.crop_type == 'relative':
            crop_h, crop_w = self.crop_size
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)
        elif self.crop_type == 'relative_range':
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            crop_h, crop_w = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)

    def __call__(self, results):
        """Call function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        image_size = results['img'].shape[:2]
        crop_size = self._get_crop_size(image_size)
        results = self._crop_data(results, crop_size)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size}, '
        repr_str += f'crop_type={self.crop_type}, '
        repr_str += f'allow_negative_crop={self.allow_negative_crop}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@PIPELINES.register_module()
class ColorJitter(object):
    def __init__(self,
                 rgb_bias=(-32, 32),
                 rgb_scale=(0.5, 1.5),
                 hue_bias=(-18, 18),
                 value_scale=(0.5, 1.5),
                 ):
        self.rgb_bias = rgb_bias
        self.rgb_scale = rgb_scale
        self.hue_bias = hue_bias
        self.value_scale = value_scale

    @staticmethod
    def _convert(image,
                 alpha=1,
                 beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    def __call__(self, results):
        img = results['img']
        img = img.copy()

        if random.randrange(2):
            self._convert(img, beta=random.uniform(*self.rgb_bias))

        if random.randrange(2):
            self._convert(img, alpha=random.uniform(*self.rgb_scale))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        if random.randrange(2):
            tmp = img[:, :, 0].astype(int) + random.randint(*self.hue_bias)
            tmp %= 180
            img[:, :, 0] = tmp

        if random.randrange(2):
            self._convert(
                img[:, :, 1], alpha=random.uniform(*self.value_scale))

        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        results['img'] = img

        # saved_img = results['img'].copy()
        # gt_bboxes = results['gt_bboxes']
        # for box in gt_bboxes:
        #     x1, y1, x2, y2 = [int(x) for x in box]
        #     cv2.rectangle(saved_img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        #
        # for landmark in results['gt_landmarks']:
        #     landmark = landmark.reshape(-1, 2)
        #     for point in landmark:
        #         x1, y1 = [int(i) for i in point]
        #         cv2.rectangle(saved_img, (x1, y1), (x1+1, y1+1), (255, 255, 255), 2)
        #
        # import time
        # cv2.imwrite(f'img{time.time()}.jpg', img)
        return results


@PIPELINES.register_module()
class RandomAffine:
    """Random affine transform data augmentation.

    This operation randomly generates affine transform matrix which including
    rotation, translation, shear and scaling transforms.

    Args:
        max_rotate_degree (float): Maximum degrees of rotation transform.
            Default: 10.
        max_translate_ratio (float): Maximum ratio of translation.
            Default: 0.1.
        scaling_ratio_range (tuple[float]): Min and max ratio of
            scaling transform. Default: (0.5, 1.5).
        max_shear_degree (float): Maximum degrees of shear
            transform. Default: 2.
        border (tuple[int]): Distance from height and width sides of input
            image to adjust output shape. Only used in mosaic dataset.
            Default: (0, 0).
        border_val (tuple[int]): Border padding values of 3 channels.
            Default: (114, 114, 114).
        min_bbox_size (float): Width and height threshold to filter bboxes.
            If the height or width of a box is smaller than this value, it
            will be removed. Default: 2.
        min_area_ratio (float): Threshold of area ratio between
            original bboxes and wrapped bboxes. If smaller than this value,
            the box will be removed. Default: 0.2.
        max_aspect_ratio (float): Aspect ratio of width and height
            threshold to filter bboxes. If max(h/w, w/h) larger than this
            value, the box will be removed.
    """

    def __init__(self,
                 max_rotate_degree=10.0,
                 max_translate_ratio=0.1,
                 scaling_ratio_range=(0.5, 1.5),
                 max_shear_degree=2.0,
                 border=(0, 0),
                 border_val=(114, 114, 114),
                 min_bbox_size=2,
                 filter_empty_img=True,
                 use_perspective=True,
                 affine_keys=['img'],
                 affine_pipelines=[]):
        assert 0 <= max_translate_ratio <= 1
        assert scaling_ratio_range[0] <= scaling_ratio_range[1]
        assert scaling_ratio_range[0] > 0
        self.max_rotate_degree = max_rotate_degree
        self.max_translate_ratio = max_translate_ratio
        self.scaling_ratio_range = scaling_ratio_range
        self.max_shear_degree = max_shear_degree
        self.border = border
        self.border_val = border_val
        self.min_bbox_size = min_bbox_size
        self.filter_empty_img = filter_empty_img
        for pipeline in affine_pipelines:
            if 'use_perspective' in pipeline:
                assert pipeline['use_perspective'] == use_perspective, \
                    "affine subpipelines must have the same affine func"
        self.affine_pipelines = Compose(affine_pipelines)
        if use_perspective:
            self.affine_func = cv2.warpPerspective
        else:
            self.affine_func = cv2.warpAffine
        self.affine_keys = affine_keys

    def __call__(self, results):
        ori_results = copy.deepcopy(results)
        img = results['img']
        height = img.shape[0] + self.border[0] * 2
        width = img.shape[1] + self.border[1] * 2

        # Center
        center_matrix = np.eye(3, dtype=np.float32)
        center_matrix[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        center_matrix[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Rotation
        rotation_degree = random.uniform(-self.max_rotate_degree,
                                         self.max_rotate_degree)
        rotation_matrix = self._get_rotation_matrix(rotation_degree)

        # Scaling
        scaling_ratio = random.uniform(self.scaling_ratio_range[0],
                                       self.scaling_ratio_range[1])
        scaling_matrix = self._get_scaling_matrix(scaling_ratio)

        # Shear
        x_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        y_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        shear_matrix = self._get_shear_matrix(x_degree, y_degree)

        # Translation
        trans_x = random.uniform(0.5 - self.max_translate_ratio,
                                 0.5 + self.max_translate_ratio) * width
        trans_y = random.uniform(0.5 - self.max_translate_ratio,
                                 0.5 + self.max_translate_ratio) * height
        translate_matrix = self._get_translation_matrix(trans_x, trans_y)

        warp_matrix = (
            translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix
            @ center_matrix)
        for key in results.get('aug_keys', self.affine_keys):
            val = results[key]
            val = self.affine_func(
                val,
                warp_matrix,
                dsize=(width, height),
                borderValue=self.border_val)
            results[key] = val
        results['img_shape'] = np.array(img.shape)
        results['affine_info'] = dict(affine_matrix=warp_matrix,
                                      scaling_ratio=scaling_ratio)
        results['affine_matrix'] = warp_matrix
        results = self.affine_pipelines(results)
        if self.filter_empty_img and 'gt_bboxes' in results:
            if len(results['gt_bboxes']) == 0:
                return ori_results
        return results

    @staticmethod
    def _get_rotation_matrix(rotate_degrees):
        radian = math.radians(rotate_degrees)
        rotation_matrix = np.array(
            [[np.cos(radian), -np.sin(radian), 0.],
             [np.sin(radian), np.cos(radian), 0.], [0., 0., 1.]],
            dtype=np.float32)
        return rotation_matrix

    @staticmethod
    def _get_shear_matrix(x_shear_degrees, y_shear_degrees):
        x_radian = math.radians(x_shear_degrees)
        y_radian = math.radians(y_shear_degrees)
        shear_matrix = np.array([[1, np.tan(x_radian), 0.],
                                 [np.tan(y_radian), 1, 0.], [0., 0., 1.]],
                                dtype=np.float32)
        return shear_matrix

    @staticmethod
    def _get_translation_matrix(x, y):
        translation_matrix = np.array([[1, 0., x], [0., 1, y], [0., 0., 1.]],
                                      dtype=np.float32)
        return translation_matrix

    @staticmethod
    def _get_scaling_matrix(scale_ratio):
        scaling_matrix = np.array(
            [[scale_ratio, 0., 0.], [0., scale_ratio, 0.], [0., 0., 1.]],
            dtype=np.float32)
        return scaling_matrix


@PIPELINES.register_module()
class NoiseImage:
    def __init__(self,
                 mean=0,
                 std=10):
        self.mean = mean
        self.std = std

    def __call__(self, results):
        img = results['img']
        sigma = np.random.random() * self.std
        noise = np.random.normal(self.mean, sigma, img.shape)
        results['img'] = np.clip(img + noise, 0, 255).astype(np.uint8)
        return results


@PIPELINES.register_module()
class RandomRadiusBlur(object):
    blur_func = dict(gaussion=cv2.GaussianBlur, median=cv2.medianBlur, blur=cv2.blur)

    def __init__(self, prob=0.5, radius=5, std=0):
        self.prob = prob
        self.radius = radius
        self.std =std

    def __call__(self, results):
        image = results['img']
        prob = np.random.random()
        if prob > self.prob:
            func_name = np.random.choice(['gaussion', 'median', 'blur'])
            radius = np.random.randint(1, self.radius) // 2 * 2 + 1
            if func_name == 'mediam':
                image = self.blur_func[func_name](image, radius)
            elif func_name == 'gaussion':
                image = self.blur_func[func_name](image, (radius, radius), self.std)
            elif func_name == 'blur':
                image = self.blur_func[func_name](image, (radius, radius))

        results['img'] = image
        return results
