import types
import collections
import numpy as np
from random import shuffle
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import torch
import  logging
#logging.basicConfig(format='[%(asctime)s] %(filename)s %(funcName)s():%(lineno)i [%(levelname)s] %(message)s', level=logging.ERROR)

batch_size = 16

# 1. 创建迭代器用于迭代外部数据
class ExternalInputIterator(object):
    def __init__(self, batch_size):
        self.images_dir = "./tmp/data/DALI_data/images/"
        self.batch_size = batch_size
        with open(self.images_dir + "file_list.txt", 'r') as f:
            self.files = [line.rstrip() for line in f if line != '']
        shuffle(self.files)

    def __iter__(self):
        self.i = 0
        self.n = len(self.files)
        return self


    def __next__(self):
        if self.i >= self.n:
            self.__iter__()
            raise StopIteration
        batch = []
        labels = []
        true_batch = min(self.batch_size, self.n - self.i)
        for _ in range(true_batch):
            jpeg_filename, label = self.files[self.i].split(' ')
            f = open(self.images_dir + jpeg_filename, 'rb')
            batch.append(np.frombuffer(f.read(), dtype = np.uint8))
            labels.append(np.array([label], dtype = np.uint8))
            self.i = (self.i + 1)
        return (batch, labels)

eii = ExternalInputIterator(batch_size)

@pipeline_def
def external_pipeline():
    pipe = Pipeline.current()
    external_iter = ExternalInputIterator(pipe.batch_size)
    jpegs, labels = fn.external_source(source=external_iter, num_outputs=2,dtype=types.UINT8  )
        ## 2.2 decode image，"mixed" operators that accept CPU inputs and produce GPU outputs,
    decode = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)

    images = fn.resize(decode, resize_x=300, resize_y=300)
    return images, labels

def run_pipeline():
    # 2. 构建并运行pipeline
    pipe = Pipeline(batch_size=batch_size, num_threads=2, device_id=0)
    with pipe:
        ## 2.1 Creates a data node which is populated with data from a Python source.
        # The data can be provided by the source function or iterable
        jpegs, labels = fn.external_source(source=eii, num_outputs=2, dtype=types.UINT8)

        ## 2.2 decode image，"mixed" operators that accept CPU inputs and produce GPU outputs,
        decode = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)

        ## 2.3 Adjusts the brightness and contrast of the images.
        enhance = fn.brightness_contrast(decode, contrast=2)

        images = fn.resize(enhance, resize_x = 400, resize_y=300)
        ## 2.4 set outputs
        pipe.set_outputs(images, labels)

    pipe.build()
    pipe_out = pipe.run()

    batch_cpu = pipe_out[0].as_cpu()
    labels_cpu = pipe_out[1]

    import matplotlib.pyplot as plt
    img = batch_cpu.at(2)
    for image in batch_cpu:
        print(image.shape)
    # plt.axis('off')

    # plt.imsave('img.png', img)

#run_pipeline()
if __name__ == "__main__":
    pipe = external_pipeline(batch_size = 2,num_threads=2, device_id=0)
    dali_iter = DALIGenericIterator(pipe, ['data', 'label'])
    for i, data in enumerate(dali_iter):
        for d in data:
            label: torch.Tensor = d["label"]
            print(type(label))
            print(f"label device id: {label.get_device()}")
            image: torch.Tensor = d["data"]
            print(f"image device id: {image.get_device()}")

