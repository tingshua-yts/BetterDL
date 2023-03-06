import time
import torch
import tvm
import tvm.relay
from tvm.contrib import graph_executor
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy
torch.manual_seed(42)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", )

# 看文档说设置torchscript会强制将use_return_dict设置为false
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", torchscript=True)

inputs = tokenizer("Hello, my dog is cute or not", return_tensors="pt")
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

traced_model = torch.jit.trace(model, [inputs["input_ids"], inputs["attention_mask"]])
print(traced_model.code)
print(traced_model.graph)


################### convert to tvm ################
shape_list = [("input_ids", inputs["input_ids"].size()), ("attention_mask", inputs["attention_mask"].size())]

mod_bert, params_bert = tvm.relay.frontend.pytorch.from_pytorch(traced_model,
                        shape_list, default_dtype="float32")

################### comile  ################
target = tvm.target.Target("llvm")
dev = tvm.device(target.kind.name, 0)
tt_a = tvm.nd.array(inputs["input_ids"].numpy(), dev)
st_a = tvm.nd.array(inputs["attention_mask"].numpy(), dev)
print(f"tt_a:{tt_a}, st_a:{st_a}")

with tvm.transform.PassContext(opt_level=3):
    lib = tvm.relay.build(mod_bert, target=target, params=params_bert)

#################### save compiled module ###################
from tvm.contrib import utils
export_path="tmp/distilbert-base-uncased/tvm/deploy_lib_cpu2.so"
lib.export_library(export_path)
print(f"success to export tvm module to :{export_path}")


################### create tvm runtime and run  ################
module = graph_executor.GraphModule(lib["default"](dev))
module.set_input("input_ids", tt_a)
module.set_input("attention_mask", st_a)
#module.set_input(**params_bert)
module.run()

# Blog中的output由两个index分别获取结果，但是在实际测试中，仅有index0来表示结果。说明模型的output在blog中为2，但是当前测试为1

tvmOutput = module.get_output(0) # index为output的下标，每个output中包含自己的batch size
print(f"tvmOutput:{tvmOutput}")