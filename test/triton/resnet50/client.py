import numpy as np
import tritonclient.http as httpclient
from PIL import Image


if __name__ == '__main__':
    # 创建triton client
    triton_client = httpclient.InferenceServerClient(url='127.0.0.1:8000')

    # 加载图片
    image = Image.open('./mug.jpg')

    # 对图片进行预处理，以满足resnet的input需求
    image = image.resize((224, 224), Image.ANTIALIAS)
    image = np.asarray(image)
    image = image / 255
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image, axes=[0, 3, 1, 2])
    image = image.astype(np.float32)

    # 创建inputs
    inputs = []
    inputs.append(httpclient.InferInput('INPUT__0', image.shape, "FP32"))
    inputs[0].set_data_from_numpy(image, binary_data=False)

    # 创建outputs
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('OUTPUT__0', binary_data=False, class_count=3))  # class_count 表示 topN 分类

    # 向triton server发起请求
    results = triton_client.infer('resnet50', request_id="001xxx", inputs=inputs, outputs=outputs)
    output_data0 = results.as_numpy('OUTPUT__0')

    # 打印结果
    print(output_data0)


