
import onnx
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor


# load model
onnx_model = onnx.load("tmp/distilbert-base-uncased/onnx/model.onnx")

# compile
target = "llvm"
shape_dict= {} # todo
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
with tvm.transform.PassContext(opt_level=3):
    lib =  relay.build(mod, target=target, params=params)

# run
dtype="float32"
dev=tvm.device(str(target),0)
module = graph_executor.GraphModule(lib["default"](dev))
module.set_input(input_name, test_testdata)
module.run()
output_shape=()
tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy

