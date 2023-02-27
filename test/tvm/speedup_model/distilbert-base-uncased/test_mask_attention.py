import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
text = ["Hello, my dog is tom", "and you ?"]
inputs = tokenizer(text, padding=True, return_tensors='pt')
print(inputs)
