from transformers import AutoTokenizer
from onnxruntime import InferenceSession

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
session = InferenceSession("tmp/distilbert-base-uncased/onnx/model.onnx")
# ONNX Runtime expects NumPy arrays as input
inputs = tokenizer("Using DistilBERT with ONNX Runtime!", return_tensors="np")
print(type(inputs))
outputs = session.run(output_names=["last_hidden_state"], input_feed=dict(inputs))
print(type(outputs))
print(len(outputs))
print(outputs[0].size)