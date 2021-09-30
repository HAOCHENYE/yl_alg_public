from general_datasets import Compose


def process_data_cpu(cfg, img):
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(img)
    return data
