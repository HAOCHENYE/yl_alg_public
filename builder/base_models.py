from abc import abstractmethod
from mmcv.runner import BaseModule
import torch
from collections import OrderedDict
import torch.distributed as dist


# TODO 更新基类逻辑，方便扩展
class BaseModels(BaseModule):
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)

    @abstractmethod
    def train_step(self, data, optimizer):
        pass

    @abstractmethod
    def val_step(self, data, optimizer=None):
        pass

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        return NotImplemented

    @abstractmethod
    def forward_test(self, imgs, img_metas, **kwargs):
        return NotImplemented

    @staticmethod
    def normalize(img, img_metas):
        img_norm_cfg = img_metas['img_norm_cfg']
        std = img_norm_cfg['std']
        mean = img_norm_cfg['mean']
        # for batch data
        if len(std.shape) != 1:
            std = std[0]
            mean = mean[0]
        if img.dtype != torch.uint8:
            return TypeError, "img.dtype must be uint when normalized in detector"
        img = img.float().permute(0, 2, 3, 1)
        img = (img - std) / mean
        img = img.permute(0, 3, 1, 2)
        return img

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars
