from torchviz import make_dot, make_dot_from_trace
import torch
import torch
from torchinfo import summary
from transformers import BertModel, BertTokenizer
model = BertModel.from_pretrained("bert-base-uncased")
tokenize = BertTokenizer.from_pretrained("bert-base-uncased")

inputs=tokenize("my name is tom", return_tensors='pt')

make_dot(model(**inputs), params=dict(model.named_parameters()))