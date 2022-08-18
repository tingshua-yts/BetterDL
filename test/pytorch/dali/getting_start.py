from nvidia.dali.pipeline import Pipeline

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

image_dir = "data/images"
max_batch_size = 8


@pipeline_def
def simple_pipeline():
    jpegs, labels = fn.readers.file(file_root=image_dir)
    images = fn.decoders.image(jpegs, device='cpu')

    return images, labels

pipe = simple_pipeline(batch_size=max_batch_size, num_threads=1, device_id=0)
pipe.build()

pipe_out = pipe.run()
print(pipe_out)