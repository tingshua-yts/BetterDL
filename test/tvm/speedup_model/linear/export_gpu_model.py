from scipy import rand
import torch
import torch.nn as nn
import tvm
import tvm.relay
from tvm.contrib import graph_executor
torch.manual_seed(42)

############# define LinearModel ###############
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.bias = nn.Parameter(torch.tensor([1.,2.,3.,4.,5.]))

    def forward(self, x, y):
        z = x+ y + self.bias * 2
        return z
model = LinearModel()
model.eval()
# x = torch.tensor([1.,2.,3.,4.,5.])
# y = torch.tensor([1.,2.,3.,4.,5.])
x = torch.tensor([1, 2, 3, 4, 5])
y = torch.tensor([1, 2, 3, 4, 5])
# x = torch.ones(5)
# y = torch.ones(5)

output = model(x, y)
print(output)

########### get torchscript ################
traced_model = torch.jit.trace(model, [x, y])
print(traced_model.code)
print(traced_model.graph)
torch.jit.save(traced_model, "tmp/linear/torchscript/lieanr-1.pt")
################### convert to tvm ################
shape_list = [("x", x.size()),("y", y.size())]

relay_mod, params_bert = tvm.relay.frontend.pytorch.from_pytorch(traced_model,
                        shape_list, default_dtype="float32")

################### comile  ################
target = tvm.target.Target(target="cuda", host="llvm")
dev = tvm.device(target.kind.name, 0)


with tvm.transform.PassContext(opt_level=3):
    lib = tvm.relay.build(relay_mod, target=target, params=params_bert)

#################### save compiled module ###################
from tvm.contrib import utils
export_path="tmp/linear/tvm/deploy_lib_gpu.so"
lib.export_library(export_path)
print(f"success to export tvm module to :{export_path}")


################### create tvm runtime and run  ################
module = graph_executor.GraphModule(lib["default"](dev))
tvm_x = tvm.nd.array(x.numpy(), dev)
tvm_y = tvm.nd.array(y.numpy(), dev)

module.set_input("x", tvm_x)
module.set_input("y", tvm_y)
module.run()
tvmOutput = module.get_output(0) # index为output的下标，每个output中包含自己的batch size
print(f"tvmOutput:{tvmOutput}")

