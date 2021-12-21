from .detector import SingleStageDetector
from builder import CUMSTOM_MODELS
import torch
from util import multi_apply


@CUMSTOM_MODELS.register_module()
class RetinaNet(SingleStageDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def export_onnx(self, img):
        out = self.backbone(img)
        out = self.neck(out)
        logits, bboxes, landmarks = self.head.simple_forward(out)
        return logits, bboxes, landmarks

