import time
import torch
import mmcv
from mmcv.runner import get_dist_info
import itertools
from util import dist_comm, tensor2numpy


def single_gpu_test(model, data_loader, type='append'):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model.forward_test(**data)
        batch_size = len(result['img_metas']['filename'])
        if type == 'append':
            results.append(result)
        else:
            results.extend(result)
        for _ in range(batch_size):
            prog_bar.update()
    results = tensor2numpy(results)
    return results


def multi_gpu_test(model, data_loader, type='append'):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model.forward_test(**data)

        if type == 'append':
            results.append(result)
        else:
            results.extend(result)

        if rank == 0:
            batch_size = len(result['img_metas']['filename'])
            for _ in range(batch_size * world_size):
                prog_bar.update()

    dist_comm.synchronize()
    results = dist_comm.gather(results, dst=0)
    results = list(itertools.chain(*results))
    results = tensor2numpy(results)
    # collect results from all ranks
    return results
