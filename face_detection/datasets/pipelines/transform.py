from general_datasets import PIPELINES
import numpy as np


@PIPELINES.register_module()
class ResizeBboxes:
    def __init__(self,
                 img_scale=None,
                 bbox_clip_border=True,
                 multiscale_mode='range',
                 keep_ratio=True,
                 backend='cv2',
                 pad_val=0):

        self.img_scale = img_scale
        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.pad_val = pad_val
        self.bbox_clip_border = bbox_clip_border

    def _resize_bboxes(self, results):
        bboxes = results["gt_bboxes"]
        scale_factor = np.repeat(results['scale_factor'], 2)
        bboxes = bboxes * scale_factor
        if self.bbox_clip_border:
            img_shape = results['img_shape']
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
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
