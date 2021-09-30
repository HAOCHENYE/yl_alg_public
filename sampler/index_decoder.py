from .builder import INDEX_DECODER


@INDEX_DECODER.register_module()
class CustomDecoder(object):
    def __init__(self):
        pass

    def __call__(self, index):
        res = dict()
        if not isinstance(index, int):
            res["index"] = index[1]
            res["batch_info"] = dict(size=index[0])
        else:
            res["index"] = index
        return res
