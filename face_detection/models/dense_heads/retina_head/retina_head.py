from itertools import product as product
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from util import multi_apply
import torch
from math import ceil
from .util_func import match, log_sum_exp, py_cpu_nms
from builder import HEAD
import numpy as np


class PriorBox(object):
    def __init__(self, cfg, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = cfg["image_size"]
        self.feature_maps = [[ceil(self.image_size[0] / step),
                              ceil(self.image_size[1] / step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1]
                                for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0]
                                for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


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
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 2)


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
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 4)


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
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 10)


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
                gt_landmarks,
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
        loc_data, conf_data, landm_data = predictions
        priors = priors
        num = loc_data.size(0)
        num_priors = (priors.size(0))

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        landm_t = torch.Tensor(num, num_priors, 10)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            img_width = img_shape[idx][1]
            img_height = img_shape[idx][0]
            gt_num = gt_nums[idx]
            truths = gt_bboxes[idx][:gt_num].detach()
            labels = gt_labels[idx][:gt_num].detach()
            landms = gt_landmarks[idx][:gt_num].detach()

            truths[:, 0::2] /= img_width
            truths[:, 1::2] /= img_height
            landms[:, 0::2] /= img_width
            landms[:, 1::2] /= img_height

            labels[labels == 0] = 1
            labels[landms[:, 0] < 0] = -1
            defaults = priors.data
            match(
                self.threshold,
                truths,
                defaults,
                self.variance,
                labels,
                landms,
                loc_t,
                conf_t,
                landm_t,
                idx)

        loc_t = loc_t.cuda()
        conf_t = conf_t.cuda()
        landm_t = landm_t.cuda()

        zeros = torch.tensor(0).cuda()
        # landm Loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        pos1 = conf_t > zeros
        num_pos_landm = pos1.long().sum(1, keepdim=True)
        N1 = max(num_pos_landm.data.sum().float(), 1)
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        landm_p = landm_data[pos_idx1].view(-1, 10)
        landm_t = landm_t[pos_idx1].view(-1, 10)
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')

        pos = conf_t != zeros
        conf_t[pos] = 1

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
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
        loss_landm /= N1

        return loss_l, loss_c, loss_landm


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
        prior_box = PriorBox(pri_cfg)
        with torch.no_grad():
            priors = prior_box.forward()
            priors = priors
        self.register_buffer("priors", priors)
        self.fpn_num = fpn_num
        self.test_cfg = test_cfg
        self.SSHHead = self._make_ssh_layers(
            fpn_num=fpn_num, in_channels=in_channels, conv_cfg=conv_cfg)
        self.total_loss = MultiBoxLoss(**loss_cfg)
        self.ClassHead = self._make_class_head(
            fpn_num=fpn_num, inchannels=in_channels, conv_cfg=conv_cfg)
        self.BboxHead = self._make_bbox_head(
            fpn_num=fpn_num, inchannels=in_channels, conv_cfg=conv_cfg)
        self.LandmarkHead = self._make_landmark_head(
            fpn_num=fpn_num, inchannels=in_channels, conv_cfg=conv_cfg)

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
        land_mark = self.LandmarkHead[level](feat)
        return logits, bbox_reg, land_mark

    def forward(self, feats):
        assert len(feats) == self.fpn_num

        logits, bbox_reg, land_mark = multi_apply(
            self.forward_single, feats, [
                level for level in range(
                    self.fpn_num)])
        bbox_reg = torch.cat(bbox_reg, dim=1)
        logits = torch.cat(logits, dim=1)
        land_mark = torch.cat(land_mark, dim=1)

        return bbox_reg, logits, land_mark

    def loss(self,
             feats,
             img_metas,
             gt_bboxes,
             gt_labels,
             gt_landmarks):
        output = self(feats)
        loss_bboxes, loss_cls, loss_landmarks = self.total_loss(
            output, self.priors, gt_bboxes, gt_labels, gt_landmarks, img_metas)

        return dict(
            loss_bboxes=loss_bboxes,
            loss_cls=loss_cls,
            loss_landmarks=loss_landmarks)

    def _decode_landmarks(self, pre):
        priors = self.priors
        variances = self.total_loss.variance
        landms = torch.cat((priors[..., :2] + pre[..., :2] * variances[0] * priors[..., 2:],
                            priors[..., :2] + pre[..., 2:4] * variances[0] * priors[..., 2:],
                            priors[..., :2] + pre[..., 4:6] * variances[0] * priors[..., 2:],
                            priors[..., :2] + pre[..., 6:8] * variances[0] * priors[..., 2:],
                            priors[..., :2] + pre[..., 8:10] * variances[0] * priors[..., 2:],
                            ), dim=2)
        return landms

    def _decode_bbox(self, loc):
        priors = self.priors
        variances = self.total_loss.variance

        boxes = torch.cat((
            priors[..., :2] + loc[..., :2] * variances[0] * priors[..., 2:],
            priors[..., 2:] * torch.exp(loc[..., 2:] * variances[1])), 2)
        boxes[..., :2] -= boxes[..., 2:] / 2
        boxes[..., 2:] += boxes[..., :2]
        return boxes

    def get_bboxes(self, feats, img_metas, demo=False):
        loc, conf, landms = self(feats)
        conf = F.softmax(conf, dim=-1)
        img_num = conf.shape[0]
        bboxes = self._decode_bbox(loc)
        landms = self._decode_landmarks(landms)
        scale_factor = img_metas['scale_factor']

        img_w = img_metas['img_shape'][1]
        img_h = img_metas['img_shape'][0]
        scale = torch.Tensor([img_w, img_h])
        landms_scale = scale.repeat((img_num, landms.shape[1], 5))
        bboxes_scale = scale.repeat((img_num, bboxes.shape[1], 2))
        landms *= landms_scale
        bboxes *= bboxes_scale

        # TODO support batch demo
        if demo:
            assert img_num == 1
            scores = conf[0]
            bboxes = bboxes[0]
            landms = landms[0]

            topk_score, topk_index = scores[:, 1].topk(2000, largest=True)
            bboxes = bboxes[topk_index]
            landms = landms[topk_index]

            valid_index = (topk_score > self.test_cfg["conf_threshold"])

            bboxes = bboxes[valid_index].cpu().detach().numpy()
            landms = landms[valid_index].cpu().detach().numpy()
            scores = topk_score[valid_index].unsqueeze(dim=1).cpu().detach().numpy()

            dets = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, self.test_cfg['nms_threshold'])

            dets = dets[keep, :]
            landms = landms[keep]

            dets[:, 0:4:2] = dets[:, 0:4:2] / scale_factor[0]
            dets[:, 1:4:2] = dets[:, 1:4:2] / scale_factor[1]
            landms[:, 0::2] = landms[:, 0::2] / scale_factor[0]
            landms[:, 1::2] = landms[:, 1::2] / scale_factor[1]
            return dict(bboxes=dets[:, :4],
                        landmarks=landms,
                        score=scores)











