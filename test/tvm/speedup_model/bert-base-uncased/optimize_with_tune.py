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




################### preprocess ################
enc = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenizing input text
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = enc.tokenize(text)
print(f"tokenized_text:{tokenized_text}")

# Masking one of the input tokens
masked_index = 8
tokenized_text[masked_index] = '[MASK]'
print(f"tokenized_text after mask:{tokenized_text}")

indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)

segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# Creating a dummy input
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
# TODO,如何进行后处理，是否有beam search的过程，要具有id转token的过程，是否有detoken或组合sentence的过程
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



################ tvm model with tune ############
## extract task
target = tvm.target.Target(target="cuda", host="llvm")

tasks = tvm.autotvm.task.extract_from_program(mod_bert["main"], target=target, params=params_bert)
print(f"task:{tasks}")

## create runner
number = 2
repeat = 1
min_repeat_ms = 0  # since we're tuning on a CPU, can be set to 0
timeout = 10  # in seconds
runner = autotvm.LocalRunner(
    number=number,
    repeat=repeat,
    timeout=timeout,
    min_repeat_ms=min_repeat_ms,
    enable_cpu_cache_flush=True,
)


### create tuning option
tuning_option = {
    "tuner": "xgb", #use an XGBoost algorithim for guiding the search.
    "trials": 20,
    "early_stopping": 100,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func="default"), runner=runner
    ),
    "tuning_records": "tmp/distilbert-base-uncased/autotuning.json",
}

### tune task
for i, task in enumerate(tasks):
    prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
    tuner_obj = XGBTuner(task, loss_type="rank")
    tuner_obj.tune(
        n_trial=min(tuning_option["trials"], len(task.config_space)),
        early_stopping=tuning_option["early_stopping"],
        measure_option=tuning_option["measure_option"],
        callbacks=[
            autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
            autotvm.callback.log_to_file(tuning_option["tuning_records"]),
        ],
    )

### compile
target = tvm.target.Target(target="cuda", host="llvm")
with autotvm.apply_history_best(tuning_option["tuning_records"]):
    with tvm.transform.PassContext(opt_level=3, config={}):
        lib = relay.build(mod_bert, target=target, params=params_bert)

### run
dev = tvm.device(target.kind.name, 0)
tt_a = tvm.nd.array(tokens_tensor.numpy(), dev)
st_a = tvm.nd.array(segments_tensors.numpy(), dev)
module = graph_executor.GraphModule(lib["default"](dev))

module.set_input("input_ids", tt_a)
module.set_input("attention_mask", st_a)
module.run()
o0 = module.get_output(0)
o1 = module.get_output(1)

## todo 结果的校验方式
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
    pytorch cuda total time: 0.8834323883056641
    tvm cuda total time:     0.5105140209197998
"""