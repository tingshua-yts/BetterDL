import torch
from torchinfo import summary
from transformers import BertModel, BertTokenizer
model = BertModel.from_pretrained("bert-base-uncased")
tokenize = BertTokenizer.from_pretrained("bert-base-uncased", torchscript=True)
inputs=tokenize(["My Name is Tom", "what is your first and last Name"], padding=True, return_tensors='pt')
summary(model, input_size=(8, 512), dtypes=[torch.long], col_names=["input_size", "output_size"])


# inputs=tokenize("my name is tom", return_tensors='pt')

# #onnx_inputs = ({"input_ids":inputs["input_ids"], "attention_mask":inputs["attention_mask"]})
# #onnx_inputs=(inputs["input_ids"], inputs["attention_mask"])
# onnx_inputs=inputs["input_ids"]
# input_names=["input_ids"]
# output_names=["output0"]

# traced_model: torch.ScriptModule = torch.jit.trace(model, [tokens_tensor, segments_tensors])
# traced_model.save("tmp/bert-base-uncased/model.torchscript")