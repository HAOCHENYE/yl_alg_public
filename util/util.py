from functools import partial
import random
import torch
import numpy as np

def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def get_default(cfg, key, default):
    try:
        return getattr(cfg, key)
    except AttributeError or KeyError:
        setattr(cfg, key, default)
        return default


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def tensor2numpy(results):
    if isinstance(results, torch.Tensor):
        return results.detach().cpu().numpy()
    if isinstance(results, dict):
        return {key: tensor2numpy(results[key]) for key in results.keys()}
    if isinstance(results, list):
        return [tensor2numpy(ele) for ele in results]
    else:
        return results


def numpy2tensor(results):
    if isinstance(results, np.ndarray):
        return torch.from_numpy(results)
    if isinstance(results, dict):
        return {key: numpy2tensor(results[key]) for key in results.keys()}
    if isinstance(results, list):
        return [numpy2tensor(ele) for ele in results]
    if isinstance(results, tuple):
        return [numpy2tensor(ele) for ele in results]
    else:
        return results


def tensor2cpu(results):
    if isinstance(results, torch.Tensor):
        return results.detach().cpu()
    if isinstance(results, dict):
        return {key: tensor2cpu(results[key]) for key in results.keys()}
    if isinstance(results, list):
        return [tensor2cpu(ele) for ele in results]
    else:
        return results

