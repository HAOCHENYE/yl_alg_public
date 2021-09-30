from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler
from .builder import SAMPLER


SAMPLER._register_module(Sampler, "Sampler")
SAMPLER._register_module(RandomSampler, "RandomSampler")
SAMPLER._register_module(SequentialSampler, "SequentialSampler")


