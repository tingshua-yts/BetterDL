import transformers

from transformers import BertModel, BertTokenizer, BertConfig
import numpy
import time
import torch
import tvm
import tvm.relay
import tvm.relay as relay
from tvm.contrib import graph_executor
from tvm.autotvm.tuner import XGBTuner
from tvm import autotvm
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import PretrainedConfig

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", )

## TODO 通过参数来关闭return dict, 为啥设置在设立都直接生效
# config=PretrainedConfig
# config.use_return_dict=False
# 看文档说设置torchscript会强制将use_return_dict设置为false
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", torchscript=True)

#model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)

inputs = tokenizer("Hello, my dog is cute or not", return_tensors="pt")
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

# Creating the trace

indexed_tokens = segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# 转换为batch size为1的tensor。Creating a dummy input
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# TODO 这里要设置为strict，否则会报错，但是不理解为什么模型中会有dict; 这里需要进一步理解distilbert-base-uncased的output
# traced_model = torch.jit.trace(model, [tokens_tensor, segments_tensors])
traced_model = torch.jit.trace(model, [inputs["input_ids"], inputs["attention_mask"]])
# TODO torch.jit.trace如何使用kw的形式给forward传递参数, 这种写法不行，要求dict的value必须是相同该类型
#traced_model = torch.jit.trace(model, ({"input_ids": inputs["input_ids"], "attention_mask":inputs["attention_mask"]}))

traced_model.save("tmp/distilbert-base-uncased/onnx/model.torchscript")
traced_model.eval()
for p in traced_model.parameters():
    p.requires_grad_(False)
model.cuda()
tt_c = inputs["input_ids"].cuda()
st_c = inputs["attention_mask"].cuda()
rest_pt = model(tt_c, st_c)
torch.cuda.synchronize()
def y():
    for i in range(100):
        model(tt_c, st_c)
    torch.cuda.synchronize()
start = time.time()
y()
end = time.time()
print(f"pytorch cuda total time: {(end-start)}")

################### convert to onnx ###############

# onnx_inputs = ({{"input_ids": inputs["input_ids"], "attention_mask":inputs["attention_mask"], "return_dict": False}})
# input_names=["input_ids", "attention_mask", "return_dict"]
# output_names=["output0"]

# torch.onnx.export(model, onnx_inputs, '.tmp/distilbert-base-uncased/onnx/model.onnx', verbose=True,input_names=input_names, output_names=output_names)

################### convert to tvm ################
shape_list = [("input_ids", inputs["input_ids"].size()), ("attention_mask", inputs["attention_mask"].size())]

# TODO: NotImplementedError: The following operators are not implemented: ['prim::DictConstruct']
# 测试先转换为onnx，再load是否可以
# 是否可以在导出前修改下模型，使其不返回Dict；是否可以为tvm添加op
mod_bert, params_bert = tvm.relay.frontend.pytorch.from_pytorch(traced_model,
                        shape_list, default_dtype="float32")

################### comile  ################
target = tvm.target.Target(target="cuda", host="llvm")
dev = tvm.device(target.kind.name, 0)
tt_a = tvm.nd.array(inputs["input_ids"].numpy(), dev)
st_a = tvm.nd.array(inputs["attention_mask"].numpy(), dev)

with tvm.transform.PassContext(opt_level=3):
    lib = tvm.relay.build(mod_bert, target=target, params=params_bert)

################### create tvm runtime and run  ################
module = graph_executor.GraphModule(lib["default"](dev))
module.set_input("input_ids", tt_a)
module.set_input("attention_mask", st_a)
module.run()

# todo, 模型结果assert比较
def x():
    for i in range(100):
        module.run()
    dev.sync()
s=time.time()
x()
e=time.time()
print(f"pytorch cuda total time: {(end-start)}")
print(f"tvm cuda total time:     {e-s}")