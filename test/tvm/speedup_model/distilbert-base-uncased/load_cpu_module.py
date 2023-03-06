import tvm
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from tvm.contrib import graph_executor
import numpy

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", )

# load the module back.
export_path="tmp/distilbert-base-uncased/tvm/deploy_lib_cpu2.so"
loaded_lib = tvm.runtime.load_module(export_path)

# set target
target = tvm.target.Target("llvm")
dev = tvm.device(target.kind.name, 0)

# preprocess
inputs = tokenizer("Hello, my dog is cute or not", return_tensors="pt")
tt_a = tvm.nd.array(inputs["input_ids"].numpy(), dev)
st_a = tvm.nd.array(inputs["attention_mask"].numpy(), dev)
inputs = tokenizer("Hello, my dog is cute or not", return_tensors="pt")

# config model
module = graph_executor.GraphModule(loaded_lib["default"](dev))
module.set_input("input_ids", tt_a)
module.set_input("attention_mask", st_a)

# run
module.run()

# get output
output = module.get_output(0).numpy()

print(f"output:{output}")

del module