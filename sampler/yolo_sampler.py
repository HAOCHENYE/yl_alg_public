#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.


from torch.utils.data.sampler import BatchSampler as torchBatchSampler
from .builder import BATCH_SAMPLER


@BATCH_SAMPLER.register_module()
class YoloBatchSampler(torchBatchSampler):
    """
    This batch sampler will generate mini-batches of (dim, index) tuples from another sampler.
    It works just like the :class:`torch.utils.data.sampler.BatchSampler`,
    but it will prepend a dimension, whilst ensuring it stays the same across one mini-batch.
    """

    def __init__(self,
                 sampler,
                 batch_size,
                 drop_last,
                 input_dimension=None,
                 mosaic=True):
        super().__init__(sampler, batch_size, drop_last)
        self.input_dim = input_dimension
        self.new_input_dim = None
        self.mosaic = mosaic

    def __iter__(self):
        self.__set_input_dim()
        for batch in super().__iter__():
            yield [(self.input_dim, idx) for idx in batch]
            self.__set_input_dim()

    def __set_input_dim(self):
        """ This function randomly changes the the input dimension of the dataset. """
        if self.new_input_dim is not None:
            self.input_dim = (self.new_input_dim[0], self.new_input_dim[1])
            self.new_input_dim = None



