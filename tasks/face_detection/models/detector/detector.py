from builder import BaseModels
from backbone import build_backbone
from abc import abstractmethod
from builder import build_head
from builder import build_neck
import cv2
from util import auto_fp16


class SingleStageDetector(BaseModels):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 *,
                 init_cfg=None,
                 fp16_enabled=False):
        super().__init__(init_cfg, fp16_enabled)
        self.backbone = build_backbone(backbone)
        self.head = build_head(bbox_head)
        self.neck = build_neck(neck)

    @auto_fp16(apply_to=['img'])
    def forward_train(self, img, img_metas, **kwargs):
        img = self.normalize(img, img_metas)
        feat = self.backbone(img)
        if self.neck:
            feat = self.neck(feat)
        losses = self.head.loss(feat, img_metas, **kwargs)
        return losses

    def forward_test(self, img, img_metas, demo=False, **kwargs):
        img = self.normalize(img, img_metas)
        feat = self.backbone(img)
        if self.neck:
            feat = self.neck(feat)
        bbox_results = self.head.get_bboxes(feat, img_metas, demo)
        return bbox_results

    def train_step(self, data, optimizer):
        losses = self.forward_train(**data)
        outputs = self._get_runner_input(data, losses)

        return outputs

    def val_step(self, data, optimizer=None):
        losses = self.forward_test(**data)
        outputs = self._get_runner_input(data, losses)

        return outputs

    def draw_bboxes(self, img, bboxes, score):
        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = [int(i) for i in box]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, "{:.4f}".format(float(score[i])), (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

    def draw_landmarks(self, img, landmarks):
        for landmark in landmarks:
            landmark = landmark.reshape(-1, 2)
            for point in landmark:
                x1, y1 = [int(i) for i in point]
                cv2.rectangle(img, (x1, y1), (x1+1, y1+1), (0, 0, 255), 2)

    @abstractmethod
    def export_onnx(self, img):
        pass
