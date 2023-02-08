import onnx
from tvm.contrib.download import download_testdata
from PIL import Image
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor
from scipy.special import softmax

################
### load model
onnx_model = onnx.load("./tmp/resnet50-new.onnx")


# Seed numpy's RNG to get consistent results
np.random.seed(0)


###############
### pre process image
img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
img_path = download_testdata(img_url, "imagenet_cat.png", module="data")

# Resize it to 224x224
resized_image = Image.open(img_path).resize((244, 244))
img_data = np.asarray(resized_image).astype("float32")

# Our input image is in HWC layout while ONNX expects CHW input, so convert the array
img_data = np.transpose(img_data, (2, 0, 1))

# Normalize according to the ImageNet input specification
imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
imagenet_stddev = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
norm_img_data = (img_data / 255 - imagenet_mean) / imagenet_stddev

# Add the batch dimension, as we are expecting 4-dimensional input: NCHW.
img_data = np.expand_dims(norm_img_data, axis=0)


################
### compile
target = "llvm"
# The input name may vary across model types. You can use a tool
# like Netron to check input names
input_name = "data"
shape_dict = {input_name: img_data.shape}

# Convert a ONNX model into an equivalent Relay Function.
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

# Helper function that builds a Relay function to run on TVM graph executor.
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)




################
### run
dtype = "float32"
# 构建tvm runtime module
dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))
module.set_input(input_name, img_data)
# 通过runtime module 执行推理
module.run()
output_shape = (1, 1000)
tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()

################
### post process
# Download a list of labels
labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
labels_path = download_testdata(labels_url, "synset.txt", module="data")

with open(labels_path, "r") as f:
    labels = [l.rstrip() for l in f]

# Open the output and read the output tensor
scores = softmax(tvm_output)
scores = np.squeeze(scores)
ranks = np.argsort(scores)[::-1]
for rank in ranks[0:5]:
    print("class='%s' with probability=%f" % (labels[rank], scores[rank]))