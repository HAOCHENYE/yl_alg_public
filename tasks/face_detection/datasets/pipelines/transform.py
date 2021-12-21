from general_datasets import PIPELINES
import numpy as np

@PIPELINES.register_module()
class ResizeBboxes:
    def __init__(self,
                 img_scale=None,
                 bbox_clip_border=True,
                 filter_small_bboxes=False,
                 multiscale_mode='range',
                 keep_ratio=True,
                 backend='cv2',
                 pad_val=0,
                 bboxes_thresh=16):

        self.img_scale = img_scale
        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.pad_val = pad_val
        self.bbox_clip_border = bbox_clip_border
        self.filter_small_bboxes = filter_small_bboxes
        self.bbox_thresh = 16

    def _resize_bboxes(self, results):
        bboxes = results["gt_bboxes"]
        # flag = bboxes[:, :2] > bboxes[:, 2:]
        # if flag.any():
        #     print(f"invalid_bboxes x1 y1: {bboxes[:, :2][flag]}, x2 y2: {bboxes[:, 2:][flag]}")
        scale_factor = np.tile(results['scale_factor'], 2)
        bboxes = bboxes * scale_factor
        if self.bbox_clip_border:
            img_shape = results['img_shape']
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])

        if self.filter_small_bboxes:
            img_shape = results['img_shape']
            h, w = img_shape[:2]
            area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            valid_index = area > self.bbox_thresh ** 2
            bboxes = bboxes[valid_index]
            # TODO sync with landmark

        results["gt_bboxes"] = bboxes

        # import cv2
        # img = results['img'].copy()
        # for box in bboxes:
        #     x1, y1, x2, y2 = [int(i) for i in box]
        #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))
        # cv2.imwrite("1.jpg", img)

    def __call__(self, results):
        assert "gt_bboxes" in results
        assert "scale_factor" in results
        self._resize_bboxes(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        return repr_str


@PIPELINES.register_module()
class ResizeLandMarks:
    def __init__(self,
                 bbox_clip_border=True,
                 multiscale_mode='range'):

        self.multiscale_mode = multiscale_mode
        self.bbox_clip_border = bbox_clip_border

    def _resize_landmarks(self, results):
        landmarks = results["gt_landmarks"]
        scale_factor = results['scale_factor']
        scale_factor = np.tile(scale_factor, (len(landmarks), 5))
        valid_index = landmarks != -1
        landmarks[valid_index] = landmarks[valid_index] * scale_factor[valid_index]

        if self.bbox_clip_border:
            img_shape = results['img_shape']
            landmarks[:, 0::2] = np.clip(landmarks[:, 0::2], 0, img_shape[1])
            landmarks[:, 1::2] = np.clip(landmarks[:, 1::2], 0, img_shape[0])
        results["gt_landmarks"] = landmarks

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """

        assert "gt_landmarks" in results
        assert "scale_factor" in results
        self._resize_landmarks(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class FlipLandMarks:
    @staticmethod
    def _flip_landmarks(results):
        landmarks = results['gt_landmarks']
        direction = results['flip_direction']
        img_shape = results['img_shape']
        assert landmarks.shape[-1] % 10 == 0
        flipped = landmarks.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            # TODO only support wider face now, when flipped, left and right will be swapped
            flipped[..., 0::2] = w - landmarks[..., 0::2]
            eye_tmp = flipped[..., :2].copy()
            flipped[..., 0:2] = flipped[..., 2:4]
            flipped[..., 2:4] = eye_tmp
            mouth_tmp = flipped[..., 6::8].copy()
            flipped[..., 6::8] = flipped[..., 8::10]
            flipped[..., 8::10] = mouth_tmp

        elif direction == 'vertical':
            h = img_shape[0]
            flipped[..., 1::2] = h - landmarks[..., 1::2]
        elif direction == 'diagonal':
            w = img_shape[1]
            h = img_shape[0]
            flipped[..., 0::2] = w - landmarks[..., 0::2]
            flipped[..., 1::2] = h - landmarks[..., 1::2]
            eye_tmp = flipped[..., :2].copy()
            flipped[..., 0:2] = flipped[..., 2:4]
            flipped[..., 2:4] = eye_tmp
            mouth_tmp = flipped[..., 6::8].copy()
            flipped[..., 6::8] = flipped[..., 8::10]
            flipped[..., 8::10] = mouth_tmp
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return flipped

    def __call__(self, results):
        if 'flip' in results and results.get('flip_direction', None):
            results['gt_landmarks'] = self._flip_landmarks(results)
        return results


@PIPELINES.register_module()
class FlipBboxes:
    @staticmethod
    def _bbox_flip(results):
        """Flip bboxes horizontally.

        Args:
            bboxes (numpy.ndarray): Bounding boxes, shape (..., 4*k)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical'.

        Returns:
            numpy.ndarray: Flipped bounding boxes.
        """
        bboxes = results['gt_bboxes']
        direction = results['flip_direction']
        img_shape = results['img_shape']
        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
        elif direction == 'vertical':
            h = img_shape[0]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        elif direction == 'diagonal':
            w = img_shape[1]
            h = img_shape[0]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return flipped

    def __call__(self, results):
        if 'flip' in results and results.get('flip_direction', None):
            results['gt_bboxes'] = self._bbox_flip(results)
            return results
        return results


@PIPELINES.register_module()
class LandmarksBboxesAffine:
    def __init__(self,
                 min_bbox_size=2,
                 min_area_ratio=0.2,
                 max_aspect_ratio=20):
        self.min_bbox_size = min_bbox_size
        self.min_area_ratio = min_area_ratio
        self.max_aspect_ratio = max_aspect_ratio

    def __call__(self, results):
        affine_info = results.get('affine_info', None)
        img_shape = results['img_shape']
        width, height = img_shape[1], img_shape[0]
        if affine_info:
            warp_matrix = affine_info['affine_matrix']
            scaling_ratio = affine_info['scaling_ratio']
            for key in results.get('bbox_fields', []):
                bboxes = results[key]
                num_bboxes = len(bboxes)
                if num_bboxes:
                    # homogeneous coordinates
                    xs = bboxes[:, [0, 2, 0, 2]].reshape(num_bboxes * 4)
                    ys = bboxes[:, [1, 3, 3, 1]].reshape(num_bboxes * 4)
                    ones = np.ones_like(xs)
                    points = np.vstack([xs, ys, ones])

                    warp_points = warp_matrix @ points
                    warp_points = warp_points[:2] / warp_points[2]
                    xs = warp_points[0].reshape(num_bboxes, 4)
                    ys = warp_points[1].reshape(num_bboxes, 4)

                    warp_bboxes = np.vstack(
                        (xs.min(1), ys.min(1), xs.max(1), ys.max(1))).T

                    warp_bboxes[:, [0, 2]] = warp_bboxes[:, [0, 2]].clip(0, width)
                    warp_bboxes[:, [1, 3]] = warp_bboxes[:, [1, 3]].clip(0, height)

                    # filter bboxes
                    valid_index = self.filter_gt_bboxes(bboxes * scaling_ratio,
                                                        warp_bboxes)
                    results[key] = warp_bboxes[valid_index]
                    if key in ['gt_bboxes']:
                        if 'gt_labels' in results:
                            results['gt_labels'] = results['gt_labels'][
                                valid_index]
                        if 'gt_masks' in results:
                            raise NotImplementedError(
                                'RandomAffine only supports bbox.')
                        if 'gt_landmarks' in results:
                            landmarks = results['gt_landmarks']
                            num_landmarks = len(landmarks)
                            xs = landmarks[:, 0::2].reshape(num_landmarks * 5)
                            ys = landmarks[:, 1::2].reshape(num_landmarks * 5)
                            ones = np.ones_like(xs)
                            points = np.vstack([xs, ys, ones])

                            warp_points = warp_matrix @ points
                            warp_points = warp_points[:2] / warp_points[2]
                            xs = warp_points[0].reshape(-1, 1)
                            ys = warp_points[1].reshape(-1, 1)
                            warp_landmarks = np.concatenate([xs, ys], axis=1)
                            warp_landmarks = warp_landmarks.reshape(num_landmarks, 10)
                            warp_landmarks[:, [0, 2]] = warp_landmarks[:, [0, 2]].clip(0, width)
                            warp_landmarks[:, [1, 3]] = warp_landmarks[:, [1, 3]].clip(0, height)
                            warp_landmarks = warp_landmarks[valid_index]
                            results['gt_landmarks'] = warp_landmarks
            return results

    def filter_gt_bboxes(self, origin_bboxes, wrapped_bboxes):
        origin_w = origin_bboxes[:, 2] - origin_bboxes[:, 0]
        origin_h = origin_bboxes[:, 3] - origin_bboxes[:, 1]
        wrapped_w = wrapped_bboxes[:, 2] - wrapped_bboxes[:, 0]
        wrapped_h = wrapped_bboxes[:, 3] - wrapped_bboxes[:, 1]
        aspect_ratio = np.maximum(wrapped_w / (wrapped_h + 1e-16),
                                  wrapped_h / (wrapped_w + 1e-16))

        wh_valid_idx = (wrapped_w > self.min_bbox_size) & \
                       (wrapped_h > self.min_bbox_size)
        area_valid_idx = wrapped_w * wrapped_h / (origin_w * origin_h +
                                                  1e-16) > self.min_area_ratio
        aspect_ratio_valid_idx = aspect_ratio < self.max_aspect_ratio
        return wh_valid_idx & area_valid_idx & aspect_ratio_valid_idx


@PIPELINES.register_module()
class CropBboxes:
    def __init__(self,
                 allow_negative_crop=False,
                 keep_bboxes_center_in=False,
                 bbox_clip_border=True
                 ):
        self.allow_negative_crop = allow_negative_crop
        self.keep_bboxes_center_in = keep_bboxes_center_in
        self.bbox_clip_border = bbox_clip_border

    def __call__(self, results):
        cropped = results.get('cropped', True)
        if 'crop_roi' not in results or not cropped:
            results['cropped'] = False
            return results
        crop_x1, crop_y1, crop_x2, crop_y2 = results['crop_roi']
        offset_h = crop_y1
        offset_w = crop_x1
        img_shape = results['img_shape']

        bboxes = results['gt_bboxes']
        if self.keep_bboxes_center_in:
            bboxes_ctx = (bboxes[:, 0] + bboxes[:, 2]) / 2
            bboxes_cty = (bboxes[:, 1] + bboxes[:, 3]) / 2
            valid_center_index = (bboxes_ctx > crop_x1) & \
                                 (bboxes_ctx < crop_x2) & \
                                 (bboxes_cty > crop_y1) & \
                                 (bboxes_cty < crop_y2)
        else:
            valid_center_index = np.ones(bboxes.shape[0])

        bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                               dtype=np.float32)
        bboxes = bboxes - bbox_offset
        if self.bbox_clip_border:
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
        valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
            bboxes[:, 3] > bboxes[:, 1]) & valid_center_index
        results['gt_bboxes'] = bboxes[valid_inds, :]
        results['gt_labels'] = results['gt_labels'][valid_inds]
        results['bboxes_valid_inds'] = valid_inds
        results['cropped'] = len(bboxes) > 0

        return results


@PIPELINES.register_module()
class LandMarksCrop:
    def __init__(self,
                 use_bboxes_index=True):
        self.use_bboxes_index = use_bboxes_index

    def __call__(self, results):
        cropped = results.get('cropped', True)
        if 'crop_roi' not in results or not cropped:
            results['cropped'] = False
            return results
        crop_x1, crop_y1, crop_x2, crop_y2 = results['crop_roi']
        offset_h = crop_y1
        offset_w = crop_x1
        img_shape = results['img_shape']

        gt_landmarks = results['gt_landmarks']
        if self.use_bboxes_index:
            valid_inds = results['bboxes_valid_inds']
        else:
            valid_inds = np.ones(gt_landmarks.shape[0])

        landmark_offset = np.tile([offset_w, offset_h], 5)
        gt_landmarks = gt_landmarks - landmark_offset
        gt_landmarks[:, 0::2] = np.clip(gt_landmarks[:, 0::2], 0, img_shape[1])
        gt_landmarks[:, 1::2] = np.clip(gt_landmarks[:, 1::2], 0, img_shape[0])

        gt_landmarks = gt_landmarks[valid_inds, :]
        results['gt_landmarks'] = gt_landmarks
        results['cropped'] = len(gt_landmarks) > 0
        return results




