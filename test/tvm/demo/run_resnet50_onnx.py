import onnxruntime as ort
import numpy as np

ort_session = ort.InferenceSession("./tmp/resnet50.onnx", providers=['CUDAExecutionProvider'])

outputs = ort_session.run(
    None,
    {"input0": np.random.randn(1, 3, 244, 244).astype(np.float32)},
)
print(outputs[0])