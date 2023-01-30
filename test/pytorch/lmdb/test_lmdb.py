import lmdb
import numpy as np
import cv2

image_path='./tmp/data/DALI_data/images/dog/dog_10.jpg'
image_label='dog'
lmdb_dir="./tmp/lmdb"
env = lmdb.open(lmdb_dir)
cache = {}  # 存储键值对

# 1. read image to binary
with open(image_path, 'rb') as f:
    # 读取图像文件的二进制格式数据
    image_bin = f.read()

# 2. write image to lmdb
cache['image_000'] = image_bin
cache['label_000'] = image_label

with env.begin(write=True) as txn:
    for k, v in cache.items():
        if isinstance(v, bytes):
            # 图片类型为bytes
            txn.put(k.encode(), v)
        else:
            # 标签类型为str, 转为bytes
            txn.put(k.encode(), v.encode())  # 编码

# 3. read from lmdb
with env.begin(write=False) as txn:
    # 获取图像数据
    image_bin_r = txn.get('image_000'.encode())
    label_r = txn.get('label_000'.encode()).decode()  # 解码

    # 将二进制文件转为十进制文件（一维数组）
    image_buf = np.frombuffer(image_bin_r, dtype=np.uint8)
    # 将数据转换(解码)成图像格式
    # cv2.IMREAD_GRAYSCALE为灰度图，cv2.IMREAD_COLOR为彩色图
    img = cv2.imdecode(image_buf_r, cv2.IMREAD_COLOR)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    cv2.imwrite("./test_image", img)