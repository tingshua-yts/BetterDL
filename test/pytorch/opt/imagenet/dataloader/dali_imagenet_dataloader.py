from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator

class ImageNetDataloader(DALIGenericIterator):
    def __init__(self, pipelines, size=-1, reader_name=None, auto_reset=False, fill_last_batch=True, dynamic_shape=False, last_batch_padded=False, last_batch_policy= LastBatchPolicy.FILL):
        self._output_map = ["image", "lable"]
        self._batch_size = pipelines.batch_size
        super().__init__(pipelines, self._output_map, size, reader_name, auto_reset, fill_last_batch, dynamic_shape, last_batch_padded, last_batch_policy)

    def __next__(self):
            # [0] means the first GPU. In the distributed case, each replica only
            # has one GPU.
            data =  super().__next__()[0]
            # 为了兼容pytorch的输出样式，这里对output进行一定的修改. Remove single-dimensional entries from the shape of the Tensor
            return [data[self._output_map[0]], data[self._output_map[1]].squeeze(-1).long()]