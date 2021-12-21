from custom_hook import EVALUATOR
import torch
import torch.nn.functional as F
from mmcv.utils import print_log
import numpy as np

@EVALUATOR.register_module()
class SegEvaluator:
    def __init__(self,
                 metrics=['custom_pranet_iou'],
                 **kwargs):
        self.metrics = metrics

    def evaluate(self, runner, results):
        metric_result = dict()
        logger = getattr(runner, 'logger', None)
        for metric in self.metrics:
            acc = (getattr(self, metric)(results) / len(results)).item()
            metric_result[metric] = acc
            print_log(f'{metric}: {acc:.4f}', logger=logger)
        return metric_result

    def custom_pranet_iou(self, results):
        """
        logits: (torch.float32) (main_out, feat_os16_sup, feat_os32_sup) of shape (N, C, H, W)
        targets: (torch.float32) shape (N, H, W), value {0,1,...,C-1}
        """
        miou = 0
        for result in results:
            logits = result['pred_mask'].detach().cpu()
            targets = result['gt_mask'].detach().cpu()
            tmp_iou = self.iou_with_sigmoid_pra(logits, targets)
            miou += tmp_iou
        return miou

    @staticmethod
    def iou_with_sigmoid_pra(sigmoid, targets, eps=1e-6):
        """
        sigmoid: (torch.float32) shape (N, 1, H, W)
        targets: (torch.float32) shape (N, H, W), value {0,1}
        """

        pred = sigmoid
        # import cv2
        # pre = pred[0].permute(1, 2, 0) * 255
        # gt = targets[0].permute(1, 2, 0)
        # pre = pre.detach().cpu().numpy()
        # gt = gt.detach().cpu().numpy()
        # cv2.imwrite("pre.png", pre)
        # cv2.imwrite("gt.png", gt)
        targets = targets / 255
        inter = (pred * targets).sum(dim=(2, 3))
        union = (pred + targets).sum(dim=(2, 3))
        wiou = (inter + 1) / (union - inter + 1)

        return wiou.mean()
