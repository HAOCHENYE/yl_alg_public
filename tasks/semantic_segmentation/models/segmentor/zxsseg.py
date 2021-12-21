from .segmentor import Segmentor
from builder import CUMSTOM_MODELS
from util import auto_fp16
import torch
from mmcv.cnn import ConvModule
from mmcv.cnn import build_conv_layer

@CUMSTOM_MODELS.register_module()
class ZxsSeg(Segmentor):
    def __init__(self, video=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video = video

    @auto_fp16(apply_to=['img', 'ori_img', 'gt_mask', 'prior_mask'])
    def forward_train(self, img, img_metas, **kwargs):
        mask = kwargs['gt_mask']
        img_ori = kwargs['ori_img']

        img = self.normalize(img, img_metas)
        img_ori = self.normalize(img_ori, img_metas)

        pos_inds = mask >= 128
        mask[pos_inds] = 1
        mask[~pos_inds] = 0

        if self.video:
            prior = kwargs['prior_mask']
            prior = prior / 255
            img = torch.cat([img, prior], 1)
            img_ori = torch.cat([img_ori, prior], 1)

        feat_ori = self._extract_feat(img_ori)
        feat = self._extract_feat(img)
        losses = self.head.loss(feat, feat_ori, img_metas, mask)
        return losses

    def _extract_feat(self, feat):
        img_size = feat.shape[2:]
        self.head.img_size = img_size
        feat = self.backbone(feat)
        if self.neck:
            feat = self.neck(feat)
        return feat

    def export_onnx(self, img):
        if isinstance(img, list):
            img3, mask1 = img
            img_size = img3.shape[2:]
            self.head.img_size = img_size
            img = torch.cat(img, dim=1)
        else:
            img_size = img.shape[2:]
            self.head.img_size = img_size

        feat = self.backbone(img)
        if self.neck:
            feat = self.neck(feat)
        mask = self.head(feat)[0]
        return torch.sigmoid(mask)

    def forward_test(self, img, img_metas, demo=False, **kwargs):
        img_size = img.shape[2:]
        self.head.img_size = img_size
        n, c, h, w = img.shape
        img = self.normalize(img, img_metas)

        if self.video:
            prior = img.new_full((n, 1, h, w), 0)
            prior = prior.float() / 255
            img = torch.cat([img, prior], 1)

        feat = self.backbone(img)
        if self.neck:
            feat = self.neck(feat)
        bbox_results = self.head.get_mask(feat, img_metas, **kwargs)
        return bbox_results

    @torch.no_grad()
    def load_video_weight(self):
        stem_conv = self.backbone.stem
        stem_weight = stem_conv.conv.weight
        new_conv = build_conv_layer(stem_conv.conv_cfg,
                                    4,
                                    stem_conv.out_channels,
                                    stem_conv.kernel_size,
                                    stem_conv.stride,
                                    stem_conv.padding,
                                    stem_conv.dilation,
                                    stem_conv.groups,
                                    stem_conv.with_bias).to(stem_weight.device)

        new_conv.weight[:, :3, :, :] = stem_weight.clone()
        self.backbone.stem.conv = new_conv



