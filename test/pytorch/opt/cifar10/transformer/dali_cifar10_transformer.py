from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali import pipeline_def
from opt.cifar10.dataset.dali_cifar10_dataset import DaliCifarDataset
import logging
"""train transform

    transform_train = transforms.Compose([
    #随机裁减, 方式过拟合
    transforms.RandomCrop(CROP_SIZE, padding=4),
    # 水平旋转图片
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])
"""
"""test transform

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])
"""

CIFAR10_CROP = 32
@pipeline_def
def DaliCifarTransformer(type="train", root="./tmp/data"):
    pipe = Pipeline.current()
    external_iter = DaliCifarDataset(pipe.batch_size, type, root=root)
    images, labels = fn.external_source(source=external_iter, num_outputs=2 )
    if type == "train":
        logging.info("aaaaa")
        # 1) transforms.RandomCrop(CROP_SIZE, padding=4) dali中的对应实现
        # 1.1) 先通过paste实现padding
        images = fn.paste(images.gpu(), device="gpu", ratio=1.25, fill_value=0)

        # 1.2) 借助fn.uniform和fn.crop实现random crop
        crop_pos_x = fn.uniform(range=(0., 1.))
        crop_pos_y = fn.uniform(range=(0., 1.))
        images = fn.crop(images, crop_h=CIFAR10_CROP, crop_w=CIFAR10_CROP, crop_pos_x=crop_pos_x, crop_pos_y=crop_pos_y)

        # 2) RandomHorizontalFlip + Normalize
        # 2.1) coin flip
        mirror_factor = fn.coin_flip(probability=0.5)
        # 2.2) crop_mirror_normalize
        images = fn.crop_mirror_normalize(images,
                                          device="gpu",
                                          mirror = mirror_factor,
                                          output_dtype=types.FLOAT,
                                          output_layout=types.NCHW,
                                          image_type=types.RGB,
                                        #   mean=[0.49139968 * 255., 0.48215827 * 255., 0.44653124 * 255.],
                                        #   std=[0.24703233 * 255., 0.24348505 * 255., 0.26158768 * 255.]
                                          mean=[0.4914* 255., 0.4822* 255., 0.4465* 255.],
                                          std=[0.2023* 255., 0.1994* 255., 0.2010* 255.]
                                            )

    else: # for "test"
        images = fn.crop_mirror_normalize(images.gpu(),
                                          device="gpu",
                                          output_dtype=types.FLOAT,
                                          output_layout=types.NCHW,
                                          image_type=types.RGB,
                                           mean=[0.4914* 255., 0.4822* 255., 0.4465* 255.],
                                          std=[0.2023* 255., 0.1994* 255., 0.2010* 255.]
                                            )

    return images, labels


