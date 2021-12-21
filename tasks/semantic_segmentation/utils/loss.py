import torch
import torch.nn.functional as F
import torch.nn as nn


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


def _gaussian_filter(x, window_1d, use_padding: bool):
    '''
    Blur input with 1-D kernel
    :param x: batch of tensors to be blured
    :param window_1d: 1-D gauss kernel
    :param use_padding: padding image before conv
    :return: blured tensors
    '''
    C = x.shape[1]
    padding = 0
    if use_padding:
        window_size = window_1d.shape[3]
        padding = window_size // 2
    out = F.conv2d(x, window_1d, stride=1, padding=(0, padding), groups=C)
    out = F.conv2d(out, window_1d.transpose(2, 3), stride=1, padding=(padding, 0), groups=C)
    return out



def ssim(X, Y, window, data_range: float, use_padding: bool=False):
    '''
    Calculate ssim index for X and Y
    :param X: images
    :param Y: images
    :param window: 1-D gauss kernel
    :param data_range: value range of input images. (usually 1.0 or 255)
    :param use_padding: padding image before conv
    :return:
    '''

    K1 = 0.01
    K2 = 0.03
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu1 = _gaussian_filter(X, window, use_padding)
    mu2 = _gaussian_filter(Y, window, use_padding)
    sigma1_sq = _gaussian_filter(X * X, window, use_padding)
    sigma2_sq = _gaussian_filter(Y * Y, window, use_padding)
    sigma12 = _gaussian_filter(X * Y, window, use_padding)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (sigma1_sq - mu1_sq)
    sigma2_sq = compensation * (sigma2_sq - mu2_sq)
    sigma12 = compensation * (sigma12 - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    # Fixed the issue that the negative value of cs_map caused ms_ssim to output Nan.
    cs_map = F.relu(cs_map)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_val = ssim_map.mean(dim=(1, 2, 3))  # reduce along CHW
    cs = cs_map.mean(dim=(1, 2, 3))

    return ssim_val, cs

def create_window_(window_size: int, sigma: float, channel: int, device):
    '''
    Create 1-D gauss kernel
    :param window_size: the size of gauss kernel
    :param sigma: sigma of normal distribution
    :param channel: input channel
    :return: 1D kernel
    '''
    coords = torch.arange(window_size, dtype=torch.float, device=device)
    coords -= window_size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    g = g.reshape(1, 1, 1, -1).repeat(channel, 1, 1, 1)
    return g


class DynamicSSIM(nn.Module):
    __constants__ = ['data_range', 'use_padding']

    def __init__(self, window_sigma=1.5, kernel_ratio=32, data_range=255., channel=3, use_padding=False, size_average=True):
        super().__init__()
        self.data_range = data_range
        self.window_sigma = window_sigma
        self.use_padding = use_padding
        self.size_average = size_average
        self.channel = channel
        self.kernel_ratio = kernel_ratio

    def forward(self, x, y, img_size):
        window_size = min(img_size) // self.kernel_ratio / 2 * 2 + 1
        window = create_window_(window_size, self.window_sigma, self.channel, x.device)
        r = ssim(x, y, window=window, data_range=self.data_range, use_padding=self.use_padding)
        if self.size_average:
            return r[0].mean()
        else:
            return r[0]


class SSIM(nn.Module):
    def __init__(self, window_size=11, window_sigma=1.5, data_range=255., channel=3, use_padding=False, size_average=True):
        '''
        :param window_size: the size of gauss kernel
        :param window_sigma: sigma of normal distribution
        :param data_range: value range of input images. (usually 1.0 or 255)
        :param channel: input channels (default: 3)
        :param use_padding: padding image before conv
        '''
        super().__init__()
        assert window_size % 2 == 1, 'Window size must be odd.'
        window = create_window(window_size, window_sigma, channel)
        self.register_buffer('window', window)
        self.data_range = data_range
        self.use_padding = use_padding
        self.size_average = size_average

    def forward(self, X, Y):
        r = ssim(X, Y, window=self.window, data_range=self.data_range, use_padding=self.use_padding)
        if self.size_average:
            return r[0].mean()
        else:
            return r[0]


ssim_loss = SSIM(window_size=11, window_sigma=1.5, data_range=1., channel=1, use_padding=False ,size_average=True)


def structure_ssim_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    ssim_out = 1 - ssim_loss(pred, mask)

    # print(wbce.mean(), wiou.mean(), ssim_out)
    return (wbce + wiou).mean() + ssim_out


def custom_ppa_ssim_loss_multi(logits, targets):
    if type(logits) == tuple:
        loss0 = structure_ssim_loss(logits[0], targets)
        loss1 = structure_ssim_loss(logits[1], targets)
        loss2 = structure_ssim_loss(logits[2], targets)
        loss3 = structure_ssim_loss(logits[3], targets)

        loss = loss0 + loss1*0.8 + loss2*0.6 + loss3*0.4 # + 5.0*lossa
        # loss = loss0 + loss1 * 0.6 + loss2 * 0.4 + loss3 * 0.2  # + 5.0*lossa

        return loss0, loss1, loss2, loss3, loss
    else:
        loss = structure_ssim_loss(logits, targets)
        return loss, loss, loss, loss, loss