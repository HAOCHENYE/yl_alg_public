from .detector import SingleStageDetector
from builder import DETECTOR


@DETECTOR.register_module()
class RetinaNet(SingleStageDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def export_onnx(self, img):
        pass
