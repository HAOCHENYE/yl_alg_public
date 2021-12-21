from itertools import product as product
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from util import multi_apply
import torch
from math import ceil
from .util_func import log_sum_exp, py_cpu_nms, match_nolandms
from builder import HEAD
import numpy as np
from torchvision.ops import batched_nms

class PriorBox(object):
    def __init__(self, cfg, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = cfg["image_size"]
        self.name = "s"

    def get_feature_maps(self):
        '''
        feature_maps: [h, w]
        :return:
        '''
        self.feature_maps = [[ceil(self.image_size[0] / step),
                              ceil(self.image_size[1] / step)] for step in self.steps]

    def forward(self):
        anchors1 = []
        for k, f in enumerate(self.feature_maps):
            x = torch.arange(f[1])
            y = torch.arange(f[0])
            xx, yy = torch.meshgrid(x, y)
            xx = xx.permute(1, 0)
            yy = yy.permute(1, 0)

            cxx = (xx.to(self.image_size.device) + 0.5) * self.steps[k] / self.image_size[1]
            cyy = (yy.to(self.image_size.device) + 0.5) * self.steps[k] / self.image_size[0]
            s_kxx = torch.full(f, self.min_sizes[k][0] / self.image_size[1],
                               device=self.image_size.device)
            s_kyy = torch.full(f, self.min_sizes[k][0] / self.image_size[0],
                               device=self.image_size.device)

            s_kxx1 = torch.full(f, self.min_sizes[k][1] / self.image_size[1],
                                device=self.image_size.device)
            s_kyy1 = torch.full(f, self.min_sizes[k][1] / self.image_size[0],
                                device=self.image_size.device)

            cxx = cxx.view(f[0], f[1], 1)
            cyy = cyy.view(f[0], f[1], 1)
            s_kxx = s_kxx.view(f[0], f[1], 1)
            s_kyy = s_kyy.view(f[0], f[1], 1)
            s_kxx1 = s_kxx1.view(f[0], f[1], 1)
            s_kyy1 = s_kyy1.view(f[0], f[1], 1)
            anchor_tmp = torch.cat([cxx, cyy, s_kxx, s_kyy, cxx, cyy, s_kxx1, s_kyy1], axis=2).view(-1, 4)
            anchors1.append(anchor_tmp)
        res = torch.cat(anchors1, axis=0)
        if self.clip:
            res.clamp_(max=1, min=0)
        return res

        # anchors = []
        # for k, f in enumerate(self.feature_maps):
        #     min_sizes = self.min_sizes[k]
        #     for i, j in product(range(f[0]), range(f[1])):
        #         for min_size in min_sizes:
        #             s_kx = min_size / self.image_size[1]
        #             s_ky = min_size / self.image_size[0]
        #             dense_cx = [x * self.steps[k] / self.image_size[1]
        #                         for x in [j + 0.5]]
        #             dense_cy = [y * self.steps[k] / self.image_size[0]
        #                         for y in [i + 0.5]]
        #             for cy, cx in product(dense_cy, dense_cx):
        #                 anchors += [cx, cy, s_kx, s_ky]
        #
        # # back to torch land
        # output = torch.Tensor(anchors).view(-1, 4)
        # if self.clip:
        #     output.clamp_(max=1, min=0)
        # return output


class SSH(nn.Module):
    def __init__(self, in_channel, out_channel, conv_cfg=dict(type=None)):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        self.conv3x3 = ConvModule(
            in_channel,
            out_channel // 2,
            kernel_size=3,
            padding=1,
            stride=1,
            act_cfg=None,
            norm_cfg=dict(type="BN"),
            conv_cfg=conv_cfg)

        self.conv5x5_1 = ConvModule(
            in_channel,
            out_channel // 4,
            kernel_size=5,
            padding=2,
            stride=1,
            norm_cfg=dict(type="BN"))

        self.conv5x5_2 = ConvModule(
            out_channel // 4,
            out_channel // 4,
            kernel_size=5,
            padding=2,
            stride=1,
            norm_cfg=dict(type="BN"))

        self.conv7x7_1 = ConvModule(
            out_channel // 4,
            out_channel // 4,
            kernel_size=7,
            padding=3,
            stride=1,
            norm_cfg=dict(type="BN"))

        self.conv7x7_2 = ConvModule(
            out_channel // 4,
            out_channel // 4,
            kernel_size=7,
            padding=3,
            stride=1,
            act_cfg=None,
            norm_cfg=dict(type="BN"))

    def forward(self, input):
        conv3x3 = self.conv3x3(input)

        conv5x5_1 = self.conv5x5_1(input)
        conv5x5 = self.conv5x5_2(conv5x5_1)

        conv7x7_2 = self.conv7x7_1(conv5x5_1)
        conv7x7 = self.conv7x7_2(conv7x7_2)

        out = torch.cat([conv3x3, conv5x5, conv7x7], dim=1)
        out = F.relu(out)
        return out


class ClassHead(nn.Module):
    def __init__(
            self,
            inchannels=512,
            num_anchors=3,
            conv_cfg=dict(
            type=None)):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = ConvModule(
            in_channels=inchannels,
            out_channels=num_anchors * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            act_cfg=None,
            conv_cfg=conv_cfg)

    def forward(self, x):
        out = self.conv1x1(x)
        return out


class BboxHead(nn.Module):
    def __init__(
            self,
            inchannels=512,
            num_anchors=3,
            conv_cfg=dict(
            type=None)):
        super(BboxHead, self).__init__()
        self.conv1x1 = ConvModule(
            in_channels=inchannels,
            out_channels=num_anchors * 4,
            kernel_size=1,
            stride=1,
            padding=0,
            act_cfg=None,
            conv_cfg=conv_cfg)

    def forward(self, x):
        out = self.conv1x1(x)
        return out


class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3, conv_cfg=dict(
            type=None)):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = ConvModule(
            in_channels=inchannels,
            out_channels=num_anchors * 10,
            kernel_size=1,
            stride=1,
            padding=0,
            act_cfg=None,
            conv_cfg=conv_cfg)

    def forward(self, x):
        out = self.conv1x1(x)
        return out


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(
            self,
            num_classes,
            overlap_thresh,
            prior_for_matching,
            bkg_label,
            neg_mining,
            neg_pos,
            neg_overlap,
            encode_target):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self,
                predictions,
                priors,
                gt_bboxes,
                gt_labels,
                img_metas):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        gt_nums = img_metas['gt_num']
        img_shape = img_metas['img_shape']
        loc_data, conf_data = predictions
        priors = priors
        num = loc_data.size(0)
        num_priors = (priors.size(0))

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        # loc_t = torch.zeros(num, num_priors, 4)
        # landm_t = torch.zeros(num, num_priors, 10)
        # conf_t = torch.zeros(num, num_priors).to(torch.int64)
        for idx in range(num):
            img_width = img_shape[idx][1]
            img_height = img_shape[idx][0]
            gt_num = gt_nums[idx]
            truths = gt_bboxes[idx][:gt_num].detach()
            labels = gt_labels[idx][:gt_num].detach()

            truths[:, 0::2] /= img_width
            truths[:, 1::2] /= img_height

            labels[labels == 0] = 1
            defaults = priors.data
            match_nolandms(
                self.threshold,
                truths,
                defaults,
                self.variance,
                labels,
                loc_t,
                conf_t,
                idx)

        loc_t = loc_t.cuda()
        conf_t = conf_t.cuda()

        zeros = torch.tensor(0).cuda()
        # landm Loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        pos1 = conf_t > zeros
        num_pos_landm = pos1.long().sum(1, keepdim=True)
        N1 = max(num_pos_landm.data.sum().float(), 1)
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        landm_p = landm_data[pos_idx1].view(-1, 10)

        pos = conf_t != zeros
        conf_t[pos] = 1

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        # flag = (torch.isnan(loc_t)==True).any()
        # print(f"gt bboxes: {flag}")
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - \
            batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)
                           ].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N

        return loss_l, loss_c


@HEAD.register_module()
class RetinaFace(nn.Module):
    def __init__(self,
                 in_channels,
                 fpn_num=3,
                 pri_cfg=dict(image_size=(640, 640),
                              min_sizes=[[16, 32], [64, 128], [256, 512]],
                              steps=[8, 16, 32],
                              clip=False
                              ),
                 conv_cfg=dict(type='Conv2d'),
                 loss_cfg=dict(num_classes=2,
                               overlap_thresh=0.35,
                               prior_for_matching=True,
                               bkg_label=0,
                               neg_mining=True,
                               neg_pos=7,
                               neg_overlap=0.35,
                               encode_target=False),
                 test_cfg=dict(conf_threshold=0.6,
                               nms_threshold=0.6)
                 ):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        # TODO encode bbox
        super(RetinaFace, self).__init__()
        self.prior_box = PriorBox(pri_cfg)
        # self.register_buffer("priors", priors)
        self.fpn_num = fpn_num
        self.test_cfg = test_cfg
        self.SSHHead = self._make_ssh_layers(
            fpn_num=fpn_num, in_channels=in_channels, conv_cfg=conv_cfg)
        self.total_loss = MultiBoxLoss(**loss_cfg)
        self.ClassHead = self._make_class_head(
            fpn_num=fpn_num, inchannels=in_channels, conv_cfg=conv_cfg)
        self.BboxHead = self._make_bbox_head(
            fpn_num=fpn_num, inchannels=in_channels, conv_cfg=conv_cfg)
        if self.with_landmark:
            self.LandmarkHead = self._make_landmark_head(
                fpn_num=fpn_num, inchannels=in_channels, conv_cfg=conv_cfg)

        # with torch.no_grad():
        #     self.prior_box.image_size = torch.Tensor(self.prior_box.image_size)
        #     self.prior = self.prior_box.get_feature_maps()
        #     self.prior = self.prior_box.forward().cuda()

    def _make_class_head(
            self,
            fpn_num,
            inchannels=64,
            anchor_num=2,
            conv_cfg=dict(
            type=None)):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num, conv_cfg))
        return classhead

    def _make_bbox_head(
            self,
            fpn_num,
            inchannels=64,
            anchor_num=2,
            conv_cfg=dict(
            type=None)):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num, conv_cfg))
        return bboxhead

    def _make_landmark_head(
            self,
            fpn_num,
            inchannels=64,
            anchor_num=2,
            conv_cfg=dict(
            type=None)):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num, conv_cfg))
        return landmarkhead

    def _make_ssh_layers(self, fpn_num, in_channels, conv_cfg):
        ssh_layers = nn.ModuleList()
        for i in range(fpn_num):
            ssh_layers.append(SSH(in_channels, in_channels, conv_cfg))
        return ssh_layers

    def forward_single(self, feat, level):
        feat = self.SSHHead[level](feat)
        bbox_reg = self.BboxHead[level](feat)
        logits = self.ClassHead[level](feat)

        return logits, bbox_reg

    def simple_forward(self, feats):
        logits, bbox_regs = multi_apply(
            self.forward_single, feats, [
                level for level in range(
                    self.fpn_num)])
        return logits, bbox_regs

    def forward(self, feats):
        batch_size = feats[0].shape[0]
        assert len(feats) == self.fpn_num

        logits, bbox_regs = self.simple_forward(feats)
        logits = [logit.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2) for logit in logits]
        bbox_regs = [bbox_reg.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4) for bbox_reg in bbox_regs]

        bbox_regs = torch.cat(bbox_regs, dim=1)
        logits = torch.cat(logits, dim=1)

        return bbox_regs, logits

    def loss(self,
             feats,
             img_metas,
             gt_bboxes,
             gt_labels):
        output = self(feats)
        batch_shape = img_metas['img_shape'][0]
        self.prior_box.image_size = batch_shape[:2]
        self.prior_box.get_feature_maps()
        priors = self.prior_box.forward()
        # priors = self.prior
        loss_bboxes, loss_cls, loss_landmarks = self.total_loss(
            output, priors, gt_bboxes, gt_labels, img_metas)

        return dict(
            loss_bboxes=2.0*loss_bboxes,
            loss_cls=loss_cls,
            loss_landmarks=loss_landmarks)

    def _decode_landmarks(self, pre, priors):
        variances = self.total_loss.variance
        landms = torch.cat((priors[..., :2] + pre[..., :2] * variances[0] * priors[..., 2:],
                            priors[..., :2] + pre[..., 2:4] * variances[0] * priors[..., 2:],
                            priors[..., :2] + pre[..., 4:6] * variances[0] * priors[..., 2:],
                            priors[..., :2] + pre[..., 6:8] * variances[0] * priors[..., 2:],
                            priors[..., :2] + pre[..., 8:10] * variances[0] * priors[..., 2:],
                            ), dim=2)
        return landms

    def _decode_bbox(self, loc, priors):
        variances = self.total_loss.variance

        boxes = torch.cat((
            priors[..., :2] + loc[..., :2] * variances[0] * priors[..., 2:],
            priors[..., 2:] * torch.exp(loc[..., 2:] * variances[1])), 2)
        boxes[..., :2] -= boxes[..., 2:] / 2
        boxes[..., 2:] += boxes[..., :2]
        return boxes

    def get_bboxes(self, feats, img_metas, demo=False):
        """

        :param feats:list[Tensor], Tensor shape: [n c h w]:
        :param img_metas:
        :param demo:
        :return:
        """
        batch_size = feats[0].shape[0]
        img_shape = img_metas['img_shape']
        if len(img_shape.shape) == 1:
            self.prior_box.image_size = img_shape[:2]
        else:
            self.prior_box.image_size = img_shape[0][:2]
        self.prior_box.get_feature_maps()

        loc, conf, landms = self(feats)
        priors = self.prior_box.forward().to(loc.device)
        conf = F.softmax(conf, dim=-1)
        img_num = conf.shape[0]
        bboxes = self._decode_bbox(loc, priors)
        landms = self._decode_landmarks(landms, priors)


        if len(img_shape.shape) == 1:
            img_w = img_metas['img_shape'][1]
            img_h = img_metas['img_shape'][0]
            scale_factor = img_metas['scale_factor']
        else:
            img_w = img_metas['img_shape'][0][1]
            img_h = img_metas['img_shape'][0][0]
            scale_factor = img_metas['scale_factor'][0]

        scale = torch.tensor([img_w, img_h]).to(bboxes.device)
        # TODO to be validated
        landms_scale = scale.repeat((img_num, landms.shape[1], 5))
        bboxes_scale = scale.repeat((img_num, bboxes.shape[1], 2))
        landms *= landms_scale
        bboxes *= bboxes_scale

        topk_scores, topk_inds = conf[..., 1].topk(2000, dim=1, largest=True)
        batch_inds = torch.arange(batch_size).view(-1, 1).expand_as(topk_inds)

        bboxes = bboxes[batch_inds, topk_inds, :]
        landms = landms[batch_inds, topk_inds, :]


        batch_inds, valid_inds = torch.where(topk_scores > self.test_cfg["conf_threshold"])
        bboxes = bboxes[batch_inds, valid_inds, :]
        landms = landms[batch_inds, valid_inds, :]
        scores = topk_scores[batch_inds, valid_inds]

        # dets = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        keep = batched_nms(bboxes, scores, batch_inds, 0.3)
        bboxes = bboxes[keep, :]
        landms = landms[keep, :]
        scores = scores[keep]
        batch_inds = batch_inds[keep]
        # keep = py_cpu_nms(dets, self.test_cfg['nms_threshold'])

        # dets = dets[keep, :]
        # landms = landms[keep]
        # TODO ugly code
        if len(bboxes.shape) == 2:
            bboxes[:, 0:4:2] = bboxes[:, 0:4:2] / scale_factor[0]
            bboxes[:, 1:4:2] = bboxes[:, 1:4:2] / scale_factor[1]
            landms[:, 0::2] = landms[:, 0::2] / scale_factor[0]
            landms[:, 1::2] = landms[:, 1::2] / scale_factor[1]

        img_metas['batch_inds'] = batch_inds
        return dict(bboxes=bboxes,
                    landmarks=landms,
                    score=scores,
                    img_metas=img_metas)
        # TODO support batch demo











