import os.path
import numpy as np
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali import pipeline_def, Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import torch

# image dir with plain jpeg files
image_dir = "./tmp/data/DALI_data/images/"

N = 8             # number of GPUs
BATCH_SIZE = 128  # batch size per GPU
IMAGE_SIZE = 3


def common_pipeline(jpegs, labels):
    images = fn.decoders.image(jpegs, device='mixed')
    images = fn.resize(
        images,
        resize_shorter=fn.random.uniform(range=(256, 480)),
        interp_type=types.INTERP_LINEAR)
    images = fn.crop_mirror_normalize(
        images,
        crop_pos_x=fn.random.uniform(range=(0.0, 1.0)),
        crop_pos_y=fn.random.uniform(range=(0.0, 1.0)),
        dtype=types.FLOAT,
        crop=(227, 227),
        mean=[128., 128., 128.],
        std=[1., 1., 1.])
    return images, labels
   # return images.gpu(), labels.gpu()

@pipeline_def
def file_reader_pipeline(num_gpus):
    jpegs, labels = fn.readers.file(
        file_root=image_dir,
        random_shuffle=True,
        shard_id=Pipeline.current().device_id,
        num_shards=num_gpus,
        name='Reader')

    return common_pipeline(jpegs, labels)


pipe_types = [
    [file_reader_pipeline, (0, 1)],
]

def run_pipeline():
        # pipes = [file_reader_pipeline(
        #     batch_size=BATCH_SIZE, num_threads=2, device_id=device_id, num_gpus=N) for device_id in range(N)]
    pipe= file_reader_pipeline(
            batch_size=BATCH_SIZE, num_threads=2, device_id=0, num_gpus=N)
    pipe.build()
    pipe_out = pipe.run()
    print(type(pipe_out[0]))
    print(type(pipe_out[1]))

run_pipeline()

def run_dataloader():
    for pipe_t in pipe_types:
        pipe_name, label_range = pipe_t
        print ("RUN: "  + pipe_name.__name__)

        # 创建dali pipeline,这里要注意下，pipeline的个数同gpu的卡数相同
        pipes = [pipe_name(
            batch_size=BATCH_SIZE, num_threads=2, device_id=device_id, num_gpus=N) for device_id in range(N)]

        # 创建pytorch的迭代器
        dali_iter = DALIGenericIterator(pipes, ['data', 'label'], reader_name='Reader')

        for i, data in enumerate(dali_iter):
            # Testing correctness of labels
            for d in data:
                label: torch.Tensor = d["label"]
                print(type(label))
                print(f"label device id: {label.get_device()}")
                image: torch.Tensor = d["data"]
                print(f"image device id: {image.get_device()}")

                # ## labels need to be integers
                # assert(np.equal(np.mod(label, 1), 0).all())
                # ## labels need to be in range pipe_name[2]
                # assert((label >= label_range[0]).all())
                # assert((label <= label_range[1]).all())
        print("OK : " + pipe_name.__name__)
#run_dataloader()