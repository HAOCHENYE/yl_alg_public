from general_datasets import PIPELINES
import cv2
import numpy as np
from general_datasets import Compose


@PIPELINES.register_module()
class ThinPlateSpline:
    def __init__(self,
                 prob=0.5,
                 ratio=0.05,
                 spline_keys=['img', 'gt_mask']):
        self.prob = prob
        self.ratio = ratio
        # self.tpss = cv2.createThinPlateSplineShapeTransformer
        self.spline_keys = spline_keys

    def __call__(self, results):
        # use_prior = results['use_prior']
        if not hasattr(self, "tps"):
            self.tps = cv2.createThinPlateSplineShapeTransformer()
        if np.random.random() < self.prob:
            return results
        img_shape = results['img_shape']

        # TODO better logic
        img_height, img_width = img_shape[:2]
        bias = np.random.randint(-int(img_height * self.ratio), int(img_width * self.ratio), 16)
        sshape = np.array([[0 + bias[0], 0 + bias[1]], [img_height + bias[2], 0 + bias[3]],
                           [0 + bias[4], img_width + bias[5]], [img_height + bias[6], img_width + bias[7]]], np.float32)
        tshape = np.array([[0 + bias[8], 0 + bias[9]], [img_height + bias[10], 0 + bias[11]],
                           [0 + bias[12], img_width + bias[13]], [img_height + bias[14], img_width + bias[15]]], np.float32)
        sshape = sshape.reshape(1, -1, 2)
        tshape = tshape.reshape(1, -1, 2)
        matches = list()
        matches.append(cv2.DMatch(0, 0, 0))
        matches.append(cv2.DMatch(1, 1, 0))
        matches.append(cv2.DMatch(2, 2, 0))
        matches.append(cv2.DMatch(3, 3, 0))

        self.tps.estimateTransformation(tshape, sshape, matches)
        spline_keys = results.get('aug_keys', self.spline_keys)
        for key in spline_keys:
            results[key] = self.tps.warpImage(results[key])

        return results


@PIPELINES.register_module()
class PersAffine:
    def __init__(self,
                 prob=0.5,
                 ratio=0.05,
                 aug_keys=['img', 'gt_mask'],
                 ):
        self.ratio = ratio
        # self.tpss = cv2.createThinPlateSplineShapeTransformer
        self.aug_keys = aug_keys
        self.prob = prob

    def __call__(self, results):
        # use_prior = results['use_prior']
        if np.random.random() < self.prob:
            return results
        img_shape = results['img_shape']
        # TODO better logic
        height, width = img_shape[:2]
        bias = np.random.randint(-int(height * self.ratio), int(width * self.ratio), 16)
        pts1 = np.float32([[0 + bias[0], 0 + bias[1]], [height + bias[2], 0 + bias[3]],
                           [0 + bias[4], width + bias[5]], [height + bias[6], width + bias[7]]])
        pts2 = np.float32([[0 + bias[8], 0 + bias[9]], [height + bias[10], 0 + bias[11]],
                           [0 + bias[12], width + bias[13]], [height + bias[14], width + bias[15]]])
        M = cv2.getPerspectiveTransform(pts1, pts2)

        aug_keys = results.get('aug_keys', self.aug_keys)
        for key in aug_keys:
            results[key] = cv2.warpPerspective(results[key], M, (width, height))

        return results


@PIPELINES.register_module()
class WarpAffine:
    def __init__(self,
                 prob=0.5,
                 ratio=0.05,
                 aug_keys=['img', 'gt_mask'],
                 ):
        self.prob = prob
        self.ratio = ratio
        # self.tpss = cv2.createThinPlateSplineShapeTransformer
        self.aug_keys = aug_keys

    def __call__(self, results):
        if np.random.random() < self.prob:
            return results
        # TODO better logit
        img_shape = results['img_shape']
        height, width = img_shape[:2]
        bias = np.random.randint(-int(height * self.ratio), int(width * self.ratio), 16)
        pts1 = np.float32([[0 + bias[0], 0 + bias[1]], [width + bias[2], 0 + bias[3]], [0 + bias[4], height + bias[5]]])
        pts2 = np.float32(
            [[0 + bias[6], 0 + bias[7]], [width + bias[8], 0 + bias[9]], [0 + bias[10], height + bias[11]]])
        M = cv2.getAffineTransform(pts1, pts2)
        aug_keys = results.get('aug_keys', self.aug_keys)
        for key in aug_keys:
            results[key] = cv2.warpAffine(results[key], M, (width, height))

        return results


@PIPELINES.register_module()
class MotionBlur:
    def __init__(self,
                 prob=0.5,
                 degree=(5, 30),
                 angle=(0, 360),
                 aug_keys=['img', 'gt_mask'],
                 ):
        self.prob = prob
        self.degree = degree
        self.angle = angle
        # self.tpss = cv2.createThinPlateSplineShapeTransformer
        self.aug_keys = aug_keys

    def __call__(self, results):
        # use_prior = results['use_prior']
        # TODO better logit
        if np.random.random() < self.prob:
            return results
        degree = np.random.randint(*self.degree)
        angle = np.random.randint(*self.angle)
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
        motion_blur_kernel = motion_blur_kernel / degree

        aug_keys = results.get('aug_keys', self.aug_keys)
        for key in aug_keys:
            results[key] = cv2.filter2D(results[key], -1, motion_blur_kernel)

        return results


@PIPELINES.register_module()
class MaskAffine:
    def __init__(self,
                 use_perspective=True,
                 ):
        if use_perspective:
            self.affine_func = cv2.warpPerspective
        else:
            self.affine_func = cv2.warpAffine

    def __call__(self, results):
        if 'affine_info' in results:
            affine_info = results['affine_info']
            affine_matrix = affine_info['affine_matrix']
        else:
            raise KeyError('affine_info must in results attrs!')
        img_shape = results['img_shape']
        img_height, img_width = img_shape[:2]

        mask = results['gt_mask']
        mask = self.affine_func(mask, affine_matrix, (img_width, img_height))
        results['gt_mask'] = mask

        return results


@PIPELINES.register_module()
class TrackMaskAug:
    def __init__(self,
                 prior_prob=0.5,
                 dilate_kernel=5,
                 video=True,
                 prior_aug_pipelines=[]):
        self.prior_aug_pipelines = Compose(prior_aug_pipelines)
        self.prior_prob = prior_prob
        self.dilate_kernel = dilate_kernel
        self.video = video

    def __call__(self, results):
        width, height = results['img'].shape[:2]
        if self.video:
            prior = np.zeros((width, height)).astype(np.uint8)
            if np.random.random() > self.prior_prob:
                gt_mask = results['gt_mask'] .copy()
                if np.random.random() >= 0.5:
                    results['aug_keys'] = ['img', 'gt_mask']
                    results = self.prior_aug_pipelines(results)
                    results['prior_mask'] = gt_mask
                else:
                    results['aug_keys'] = ['gt_mask']
                    results = self.prior_aug_pipelines(results)
                    results['prior_mask'] = results['gt_mask']
                    results['gt_mask'] = gt_mask
            else:
                results['prior_mask'] = prior
            results['prior_mask'] = cv2.dilate(results['prior_mask'], (self.dilate_kernel, self.dilate_kernel))
            results['ori_img'] = results['img'].copy()
            results['img_fields'].extend(['prior_mask', 'ori_img'])
        else:
            results['ori_img'] = results['img'].copy()
            results['img_fields'].extend(['ori_img'])
        return results













