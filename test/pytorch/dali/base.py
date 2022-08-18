import logging
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import  logging
logging.basicConfig(format='[%(asctime)s] %(filename)s %(funcName)s():%(lineno)i [%(levelname)s] %(message)s', level=logging.DEBUG)
class DALIDataloader(DALIGenericIterator):
    def __init__(self, pipeline, size, batch_size, output_map=["data", "label"], auto_reset=True, onehot_label=False):
        self._size = size
        self.batch_size = batch_size
        self.onehot_label = onehot_label
        self.output_map = output_map
        super().__init__(pipelines=pipeline, size=size, auto_reset=auto_reset, output_map=output_map)

    def __next__(self):
        # todo: 不知道为什么这返回的不是tensor类型，是dict类型
        # if self._first_batch is not None:
        #     logging.info("firt batch is not None")
        #     batch = self._first_batch
        #     self._first_batch = None
        #     return batch
        data = super().__next__()[0]
        #logging.info(type(data[self.output_map[0]]))
        #logging.info(type(data[self.output_map[1]]))

        if self.onehot_label:
            return [data[self.output_map[0]], data[self.output_map[1]].squeeze().long()]
        else:
            return [data[self.output_map[0]], data[self.output_map[1]]]

    def __len__(self):
        if self.size%self.batch_size==0:
            return self.size//self.batch_size
        else:
            return self.size//self.batch_size+1