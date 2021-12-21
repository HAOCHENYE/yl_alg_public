from builder import BaseModels
from backbone import build_backbone
from abc import abstractmethod
from builder import build_head
from builder import build_neck
from util import auto_fp16


class Segmentor(BaseModels):
    def __init__(self,
                 backbone,
                 neck,
                 seg_head,
                 *,
                 init_cfg=None,
                 fp16_enabled=False):
        super().__init__(init_cfg, fp16_enabled)
        self.backbone = build_backbone(backbone)
        self.head = build_head(seg_head)
        if neck:
            self.neck = build_neck(neck)
        else:
            self.neck = None

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
        pred_mask = self.head.get_mask(feat, img_metas, demo)
        return pred_mask

    def train_step(self, data, optimizer):
        losses = self.forward_train(**data)
        outputs = self._get_runner_input(data, losses)

        return outputs

    def val_step(self, data, optimizer=None):
        losses = self.forward_test(**data)
        outputs = self._get_runner_input(data, losses)

        return outputs

    @abstractmethod
    def export_onnx(self, img):
        pass
