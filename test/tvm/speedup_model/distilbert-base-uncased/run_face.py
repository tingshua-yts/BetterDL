import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", torchscript=True)

inputs = tokenizer("Hello, my dog is cute or not", return_tensors="pt")
#inputs = tokenizer.tokenize("Hello, my dog is cute")
print("inputs:")
for v in inputs.data:
    print(f"\t {v}")
"""
inputs:
         input_ids
         attention_mask
"""
with torch.no_grad():
    logits = model(**inputs).logits

input_ids=inputs["input_ids"]
print(f"type(input_ids):{type(input_ids)}")
"""
    type(input_ids):<class 'torch.Tensor'>
"""
print(f"input_ids.size:{input_ids.size()}")
"""
    input_ids.size:torch.Size([1, 10])
"""
attention_mask=inputs["attention_mask"]
print(f"attention_mask.size:{attention_mask.size()}")
"""
    attention_mask.size:torch.Size([1, 10])
"""
predicted_class_id = logits.argmax().item()
print("output:")
print(model.config.id2label[predicted_class_id])
print(model.config.id2label)
"""
output:
LABEL_0
{0: 'LABEL_0', 1: 'LABEL_1'}
"""