from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIGenericIterator

CIFAR_IMAGES_NUM_TRAIN = 50000
CIFAR_IMAGES_NUM_TEST = 10000

class DaliCifar10Dataloader(DALIGenericIterator):
    def __init__(self, pipelines, type = "train", reader_name=None, auto_reset=False, fill_last_batch=True, dynamic_shape=False, last_batch_padded=False):
        self._output_map = ["image", "lable"]
        self._batch_size = pipelines.batch_size
        size = CIFAR_IMAGES_NUM_TRAIN if type == "train" else CIFAR_IMAGES_NUM_TEST
        super().__init__(pipelines, self._output_map, size, reader_name, auto_reset, fill_last_batch, dynamic_shape, last_batch_padded)

    def __next__(self):
            # [0] means the first GPU. In the distributed case, each replica only
            # has one GPU.
            data =  super().__next__()[0]
            # 为了兼容pytorch的输出样式，这里对output进行一定的修改. reshape是将[128,1]修改为[128]
            return [data[self._output_map[0]], data[self._output_map[1]].reshape([self._batch_size])]

