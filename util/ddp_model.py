from torch.nn.parallel.distributed import DistributedDataParallel
from torch.nn.parallel import DataParallel
from itertools import chain
import torch
import torch.distributed as dist
import logging
import itertools


def _find_tensors(obj):
    r"""
    Recursively find all tensors contained in the specified object.
    """
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return itertools.chain(*map(_find_tensors, obj))
    if isinstance(obj, dict):
        return itertools.chain(*map(_find_tensors, obj.values()))
    return []


class YLDataParallel(DataParallel):
    def forward_test(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module.forward_test(*inputs, **kwargs)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("module must have its parameters and buffers "
                                   "on device {} (device_ids[0]) but found one of "
                                   "them on device: {}".format(self.src_device_obj, t.device))

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module.forward_test(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def train_step(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module.train_step(*inputs, **kwargs)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("module must have its parameters and buffers "
                                   "on device {} (device_ids[0]) but found one of "
                                   "them on device: {}".format(self.src_device_obj, t.device))

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module.train_step(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)


class YLDistributedDataParallel(DistributedDataParallel):
    def forward_test(self, *inputs, **kwargs):
        if self.ddp_join_enabled:
            ones = torch.ones(
                1, device=self.device
            )
            work = dist.all_reduce(ones, group=self.process_group, async_op=True)
            self.reducer._set_forward_pass_work_handle(
                work, self.ddp_join_divide_by_initial_world_size
            )

        # Calling _rebuild_buckets before forward compuation,
        # It may allocate new buckets before deallocating old buckets
        # inside _rebuild_buckets. To save peak memory usage,
        # call _rebuild_buckets before the peak memory usage increases
        # during forward computation.
        # This should be called only once during whole training period.
        if self.reducer._rebuild_buckets():
            logging.info("Reducer buckets have been rebuilt in this iteration.")

        if self.require_forward_param_sync:
            self._sync_params()

        if self.ddp_join_enabled:
            # Notify joined ranks whether they should sync in backwards pass or not.
            self._check_global_requires_backward_grad_sync(is_joined_rank=False)

        if self.device_ids:
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            if len(self.device_ids) == 1:
                output = self.module.forward_test(*inputs[0], **kwargs[0])
            else:
                outputs = self.parallel_apply(self._module_copies[:len(inputs)], inputs, kwargs)
                output = self.gather(outputs, self.output_device)
        else:
            output = self.module.forward_test(*inputs, **kwargs)

        if torch.is_grad_enabled() and self.require_backward_grad_sync:
            self.require_forward_param_sync = True
            # We'll return the output object verbatim since it is a freeform
            # object. We need to find any tensors in this object, though,
            # because we need to figure out which parameters were used during
            # this forward pass, to ensure we short circuit reduction for any
            # unused parameters. Only if `find_unused_parameters` is set.
            if self.find_unused_parameters:
                self.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        else:
            self.require_forward_param_sync = False

        return output

    def train_step(self, *inputs, **kwargs):
        if self.ddp_join_enabled:
            ones = torch.ones(
                1, device=self.device
            )
            work = dist.all_reduce(ones, group=self.process_group, async_op=True)
            self.reducer._set_forward_pass_work_handle(
                work, self.ddp_join_divide_by_initial_world_size
            )

        # Calling _rebuild_buckets before forward compuation,
        # It may allocate new buckets before deallocating old buckets
        # inside _rebuild_buckets. To save peak memory usage,
        # call _rebuild_buckets before the peak memory usage increases
        # during forward computation.
        # This should be called only once during whole training period.
        if self.reducer._rebuild_buckets():
            logging.info("Reducer buckets have been rebuilt in this iteration.")

        if self.require_forward_param_sync:
            self._sync_params()

        if self.ddp_join_enabled:
            # Notify joined ranks whether they should sync in backwards pass or not.
            self._check_global_requires_backward_grad_sync(is_joined_rank=False)

        if self.device_ids:
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            if len(self.device_ids) == 1:
                output = self.module.train_step(*inputs[0], **kwargs[0])
            else:
                outputs = self.parallel_apply(self._module_copies[:len(inputs)], inputs, kwargs)
                output = self.gather(outputs, self.output_device)
        else:
            output = self.module.train_step(*inputs, **kwargs)

        if torch.is_grad_enabled() and self.require_backward_grad_sync:
            self.require_forward_param_sync = True
            # We'll return the output object verbatim since it is a freeform
            # object. We need to find any tensors in this object, though,
            # because we need to figure out which parameters were used during
            # this forward pass, to ensure we short circuit reduction for any
            # unused parameters. Only if `find_unused_parameters` is set.
            if self.find_unused_parameters:
                self.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        else:
            self.require_forward_param_sync = False

        return output
