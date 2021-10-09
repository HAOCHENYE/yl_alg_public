from builder import BaseModels
from builder import build_backbone
from abc import abstractmethod
from builder import build_head
from builder import build_neck
import cv2

class SingleStageDetector(BaseModels):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 *,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        self.head = build_head(bbox_head)
        self.neck = build_neck(neck)

    def forward_train(self, img, img_metas, **kwargs):
        feat = self.backbone(img)
        if self.neck:
            feat = self.neck(feat)
        losses = self.head.loss(feat, img_metas, **kwargs)
        return losses

    def forward_test(self, img, img_metas, demo=False, **kwargs):
        # TODO 整合normalize位置
        feat = self.backbone(img)
        if self.neck:
            feat = self.neck(feat)
        bbox_results = self.head.get_bboxes(feat, img_metas, demo)
        return bbox_results

    def train_step(self, data, optimizer):
        self.normalize(data)
        losses = self.forward_train(**data)
        outputs = self._get_runner_input(data, losses)

        return outputs

    def val_step(self, data, optimizer=None):
        losses = self.forward_test(**data)
        outputs = self._get_runner_input(data, losses)

        return outputs

    def _get_runner_input(self, data, losses):
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def draw_bboxes(self, img, bboxes, score):
        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = [int(i) for i in box]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))
            cv2.putText(img, "{:.4f}".format(float(score[i])), (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

    def draw_landmarks(self, img, landmarks):
        for landmark in landmarks:
            landmark = landmark.reshape(-1, 2)
            for point in landmark:
                x1, y1 = [int(i) for i in point]
                cv2.rectangle(img, (x1, y1), (x1+1, y1+1), (0, 0, 255))

    @abstractmethod
    def export_onnx(self, img):
        pass
