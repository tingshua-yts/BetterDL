import time
import torch
import tvm
import tvm.relay
from tvm.contrib import graph_executor
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy

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


# TODO 这里要设置为strict，否则会报错，但是不理解为什么模型中会有dict; 这里需要进一步理解distilbert-base-uncased的output
# traced_model = torch.jit.trace(model, [tokens_tensor, segments_tensors])
traced_model = torch.jit.trace(model, [inputs["input_ids"], inputs["attention_mask"]])
# TODO torch.jit.trace如何使用kw的形式给forward传递参数, 这种写法不行，要求dict的value必须是相同该类型
#traced_model = torch.jit.trace(model, ({"input_ids": inputs["input_ids"], "attention_mask":inputs["attention_mask"]}))

traced_model.save("tmp/distilbert-base-uncased/onnx/model.torchscript")
traced_model.eval()
for p in traced_model.parameters():
    p.requires_grad_(False)
traced_model.cuda()
tt_c = inputs["input_ids"].cuda()
st_c = inputs["attention_mask"].cuda()
res_py = traced_model(tt_c, st_c)
torch.cuda.synchronize()
print(f"res_py:{res_py}")
def y():
    for i in range(100):
        traced_model(tt_c, st_c)
    torch.cuda.synchronize()
start = time.time()
y()
end = time.time()
print(f"pytorch cuda total time: {(end-start)}")

################### convert to tvm ################
shape_list = [("input_ids", inputs["input_ids"].size()), ("attention_mask", inputs["attention_mask"].size())]

# TODO: NotImplementedError: The following operators are not implemented: ['prim::DictConstruct']
# 测试先转换为onnx，再load是否可以
# 是否可以在导出前修改下模型，使其不返回Dict；是否可以为tvm添加op
mod_bert, params_bert = tvm.relay.frontend.pytorch.from_pytorch(traced_model,
                        shape_list, default_dtype="float32")

################### comile  ################
target = tvm.target.Target(target="cuda", host="llvm")
with tvm.transform.PassContext(opt_level=3):
    lib = tvm.relay.build(mod_bert, target=target, params=params_bert)

#################### save compiled module ###################
from tvm.contrib import utils
export_path="tmp/distilbert-base-uncased/tvm/deploy_lib.so"
lib.export_library(export_path)
print(f"success to export tvm module to :{export_path}")

################### create tvm runtime and run  ################
dev = tvm.device(target.kind.name, 0)
module = graph_executor.GraphModule(lib["default"](dev))
tt_a = tvm.nd.array(inputs["input_ids"].numpy(), dev)
st_a = tvm.nd.array(inputs["attention_mask"].numpy(), dev)
print(f"tt_a:{tt_a}, st_a:{st_a}")
module.set_input("input_ids", tt_a)
module.set_input("attention_mask", st_a)
module.run()

# Blog中的output由两个index分别获取结果，但是在实际测试中，仅有index0来表示结果。说明模型的output在blog中为2，但是当前测试为1

tvmOutput = module.get_output(0) # index为output的下标，每个output中包含自己的batch size
torchOutput = res_py[0]
# res_py为一个tuple类型，其内容为：res_py:(tensor([[0.0831, 0.0436]], device='cuda:0'),),我们想要的output在index为0的位置
# tvmOutput为TVM中的DNArray，其已经是从tvm 的output index为0的位置获取完的结果
# 下面的比较思路都是将torchOutput和tvmOutput转换为numpy进行比较
print(f"torchOutput:{torchOutput}\t tvmOutput:{tvmOutput}")
pyRes0, pyRes1 = torchOutput.cpu().numpy()[0][0], torchOutput.cpu().numpy()[0][1]
tvmRes0, tvmRes1 = tvmOutput.asnumpy()[0][0], tvmOutput.asnumpy()[0][1]
print(f"pyRes0:{pyRes0}\tpyRes1:{pyRes1}\ttvmRes0:{tvmRes0}\ttvmRes1:{tvmRes1}")
diff = (numpy.abs((pyRes0 - tvmRes0)), numpy.abs((pyRes1 - tvmRes1)))
print(f"diff: {diff}")

def x():
    for i in range(100):
        module.run()
    dev.sync()
s=time.time()
x()
e=time.time()
print(f"pytorch cuda total time: {(end-start)}")
print(f"tvm cuda total time:     {e-s}")