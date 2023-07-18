
import numpy as np

import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit

class ONNXClassifierWrapper():
    def __init__(self, file, num_classes, target_dtype = np.float32):

        self.target_dtype = target_dtype
        self.num_classes = num_classes
        # 模型加载
        self.load(file)

        self.stream = None

    def load(self, file):
        # 打开weight文件
        f = open(file, "rb")
        # 创建python runtime
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        # 创建 engine
        engine = runtime.deserialize_cuda_engine(f.read())
        # 创建context
        self.context = engine.create_execution_context()

    def allocate_memory(self, batch):
        self.output = np.empty(self.num_classes, dtype = self.target_dtype) # Need to set both input and output precisions to FP16 to fully enable FP16

        # Allocate device memory
        self.d_input = cuda.mem_alloc(1 * batch.nbytes)
        self.d_output = cuda.mem_alloc(1 * self.output.nbytes)

        self.bindings = [int(self.d_input), int(self.d_output)]

        self.stream = cuda.Stream()

    def predict(self, batch): # result gets copied into output
        # 分配内存
        if self.stream is None:
            self.allocate_memory(batch)

        # Transfer input data to device
        cuda.memcpy_htod_async(self.d_input, batch, self.stream)

        # Execute model
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)

        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)

        # Syncronize threads
        self.stream.synchronize()

        return self.output


N_CLASSES = 1000 # Our ResNet-50 is trained on a 1000 class ImageNet task
BATCH_SIZE=32
PRECISION=np.float32
model_path="/mnt/model/resnet50-onnx/resnet50/resnet_engine.trt"
trt_model = ONNXClassifierWrapper(model_path, [BATCH_SIZE, N_CLASSES], target_dtype = PRECISION)

dummy_input_batch = np.zeros((BATCH_SIZE, 224, 224, 3), dtype = PRECISION)
predictions = trt_model.predict(dummy_input_batch)

print("dummy result")
print(predictions[0])