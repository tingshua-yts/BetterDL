from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator

@pipeline_def
def create_dali_pipeline(data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True):
    images, labels = fn.readers.file(file_root=data_dir,
                                     shard_id=shard_id, # used for multi GPU
                                     num_shards=num_shards,
                                     random_shuffle=is_training,
                                     pad_last_batch=True,
                                     name="Reader")
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'

    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0

    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0

    """_summary_

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
    """
    if is_training:
        images = fn.decoders.image_random_crop(images,
                                               device=decoder_device, output_type=types.RGB,
                                               #nvJPEG 解码时使用的device memory，避免reallocate
                                               device_memory_padding=device_memory_padding,
                                               #nvJPEG 解码时使用的host pin memory，避免reallocate
                                               host_memory_padding=host_memory_padding,

                                               preallocate_width_hint=preallocate_width_hint,
                                               preallocate_height_hint=preallocate_height_hint,
                                               random_aspect_ratio=[0.8, 1.25],
                                               random_area=[0.1, 1.0],
                                               num_attempts=100)
        images = fn.resize(images,
                           device=dali_device,
                           resize_x=crop,
                           resize_y=crop,
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images,
                                   device=decoder_device,
                                   output_type=types.RGB)
        images = fn.resize(images,
                           device=dali_device,
                           size=size,
                           mode="not_smaller",
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = False

    images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                      std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                      mirror=mirror)
    labels = labels.gpu()
    return images, labels



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
