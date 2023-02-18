import transformers

from transformers import BertModel, BertTokenizer, BertConfig
import numpy
import time
import torch
import tvm
import tvm.relay
from tvm.contrib import graph_executor




################### preprocess ################
enc = BertTokenizer.from_pretrained("bert-base-uncased")

# 分词为text的list，Tokenizing input text
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = enc.tokenize(text)
print(f"tokenized_text:{tokenized_text}")

# 在分词列表中设置MASK，Masking one of the input tokens
masked_index = 8
tokenized_text[masked_index] = '[MASK]'
print(f"tokenized_text after mask:{tokenized_text}")

# 获取分词index
indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
# 设置分句ids
# TODO，生产环境应该如何分句
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# 转换为batch size为1的tensor。Creating a dummy input
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
print(f"tokens_tensor:{tokens_tensor}")
print(f"segments_tensors:{segments_tensors}")
dummy_input = [tokens_tensor, segments_tensors]

################### convert model to pytorch ################
# If you are instantiating the model with `from_pretrained` you can also easily set the TorchScript flag
model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)

model.eval()
for p in model.parameters():
    p.requires_grad_(False)

transformers.__version__


# Creating the trace
traced_model: torch.ScriptModule = torch.jit.trace(model, [tokens_tensor, segments_tensors])
traced_model.save("tmp/distilbert-base-uncased/onnx/model.torchscript")
traced_model.eval()
for p in traced_model.parameters():
    p.requires_grad_(False)

################### run model use python ################

model.cuda()
tt_c = tokens_tensor.cuda()
st_c = segments_tensors.cuda()
res_pt = model(tt_c, st_c)
torch.cuda.synchronize()
def y():
    for i in range(100):
        model(tt_c, st_c)
    torch.cuda.synchronize()
start = time.time()
y()
end = time.time()
print(f"pytorch cuda total time: {(end-start)/100}")


################### convert to tvm ################
# get shape list, 从netron也能够获取
# TODO：这里的shape应该是随着input的length变化而变化，那请求只能是固定input size吗？

for i in  list(traced_model.graph.inputs()):
    print(f"name:{i.debugName()}")
"""
name:self.1
name:input_ids
name:attention_mask.1
"""
shape_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(traced_model.graph.inputs())[1:]]
print(f"shape_list:{shape_list}")
mod_bert, params_bert = tvm.relay.frontend.pytorch.from_pytorch(traced_model,
                        shape_list, default_dtype="float32")

################### comile  ################

#target = tvm.target.rocm(model='gfx906')
#target=tvm.target.cuda()
target = tvm.target.Target(target="cuda", host="llvm")

#ctx = tvm.context(target.id.name)

#target_host = 'llvm'
dev = tvm.device(target.kind.name, 0)
tt_a = tvm.nd.array(tokens_tensor.numpy(), dev)
st_a = tvm.nd.array(segments_tensors.numpy(), dev)
#tvm.relay.backend.compile_engine.get().clear() # just to be sure, see https://github.com/apache/incubator-tvm/pull/5724

with tvm.transform.PassContext(opt_level=3):
        lib = tvm.relay.build(mod_bert,
                                     target=target,
                                     params=params_bert)

################### create tvm runtime and run  ################

#module = tvm.contrib.graph_runtime.create(graph, lib, ctx)
module = graph_executor.GraphModule(lib["default"](dev))


module.set_input("input_ids", tt_a)
module.set_input("attention_mask", st_a)
module.run()
o0 = module.get_output(0)
o1 = module.get_output(1)
(numpy.abs((res_pt[0].cpu().numpy() - o0.asnumpy())).max(),
 numpy.abs((res_pt[1].cpu().numpy() - o1.asnumpy())).max())


def x():
    for i in range(100):
        module.run()
    dev.sync()
s=time.time()
x()
e=time.time()
print(f"pytorch cuda total time: {(end-start)}")
print(f"tvm cuda total time:     {e-s}")

"""
    pytorch cuda total time: 0.8873512744903564
    tvm cuda total time:     0.5091218948364258
"""