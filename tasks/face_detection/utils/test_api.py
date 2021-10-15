import torch
import mmcv


def single_gpu_test(model, data_loader, type='append'):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model.forward_test(**data)

        batch_size = len(result)
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results
