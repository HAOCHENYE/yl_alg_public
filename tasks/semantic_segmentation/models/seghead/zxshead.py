from ...utils import SSIM, DynamicSSIM
import torch
from builder import HEAD
import torch.nn as nn
from mmcv.cnn import ConvModule
import torch.nn.functional as F


def create_window(window_size: int, sigma: float, channel: int):
    '''
    Create 1-D gauss kernel
    :param window_size: the size of gauss kernel
    :param sigma: sigma of normal distribution
    :param channel: input channel
    :return: 1D kernel
    '''
    coords = torch.arange(window_size, dtype=torch.float)
    coords -= window_size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    g = g.reshape(1, 1, 1, -1).repeat(channel, 1, 1, 1)
    return g


@HEAD.register_module()
class ZxsHead(nn.Module):
    up_kwargs = {'mode': 'bilinear', 'align_corners': True}

    def __init__(self,
                 in_channels=[32, 32, 48, 64],
                 maskloss_weight=[1.0, 0.8, 0.6, 0.4],
                 KLloss_weight=10,
                 out_channel=1,
                 out_idx=0,
                 dynamic_ssim=False):
        super().__init__()
        if not dynamic_ssim:
            self.ssim_loss = SSIM(
                window_size=11,
                window_sigma=1.5,
                data_range=1.,
                channel=1,
                use_padding=False,
                size_average=True)
        else:
            self.ssim_loss = DynamicSSIM(window_sigma=1.5, channel=1)
        self.dynamic_ssim = dynamic_ssim
        self.out_idx = out_idx
        self.num_in = len(in_channels)
        self.maskloss_weight = maskloss_weight
        self.KLloss_weight = KLloss_weight
        self.mask_heads = nn.ModuleList()

        for i in range(self.num_in):
            self.mask_heads.append(
                ConvModule(
                    in_channels[i],
                    out_channel,
                    1,
                    act_cfg=None))

    def forward(self, feats):
        masks = [F.interpolate(self.mask_heads[i](
            feats[i]), self.img_size, **self.up_kwargs) for i in range(self.num_in)]
        return masks

    def loss(self,
             feats,
             feats_ori,
             img_metas,
             gt_mask):
        self.eval()
        feats = self(feats)
        feats_ori = self(feats_ori)
        self.train()
        loss = self.custom_ppa_ssim_loss_multi(
            feats, gt_mask)
        loss_ori = self.custom_ppa_ssim_loss_multi(
            feats_ori, gt_mask)
        KL_loss = self.loss_KL(feats[self.out_idx],
                               feats_ori[self.out_idx],
                               1) * self.KLloss_weight
        res = dict(ssim_loss=loss,
                   simm_loss_ori=loss_ori,
                   KL_loss=KL_loss)
        return res

    @staticmethod
    def loss_KL(student_outputs, teacher_outputs, T):
        proba = torch.sigmoid(student_outputs)
        log_prob = torch.log(torch.clamp(
            torch.stack([proba, 1 - proba], dim=1), 1e-6))

        proba_t = torch.sigmoid(teacher_outputs)
        targets = torch.clamp(torch.stack([proba_t, 1 - proba_t], dim=1), 1e-6)

        KD_loss = torch.nn.KLDivLoss()(log_prob, targets)

        return KD_loss

    def get_mask(self, feat, img_metas, **kwargs):
        feat = self(feat)
        result = dict()
        pred_mask = F.sigmoid(feat[self.out_idx])
        result['pred_mask'] = pred_mask
        result['img_metas'] = img_metas
        if 'gt_mask' in kwargs:
            result['gt_mask'] = kwargs['gt_mask']
        return result

    def custom_ppa_ssim_loss_multi(self, logits, targets):
        loss = 0
        for i in range(self.num_in):
            loss += self.structure_ssim_loss(logits[i],
                                             targets) * self.maskloss_weight[i]
        return loss

    def structure_ssim_loss(self, pred, mask):
        if self.dynamic_ssim:
            kernel_size = int(min(self.img_size) / 11 // 2 * 2 + 1)
            weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2) - mask)
        else:
            weit = 1 + 5 * torch.abs(
                F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        if not self.dynamic_ssim:
            ssim_out = 1 - self.ssim_loss(pred, mask)
        else:
            ssim_out = 1 - self.ssim_loss(pred, mask, self.img_size)

        return (wbce + wiou).mean() + ssim_out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
