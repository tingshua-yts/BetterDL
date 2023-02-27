from transformers import BertTokenizer, BertModel

# load model
model = BertModel.from_pretrained("bert-base-uncased")

# tokenize
text = "Replace me by any text you'd like."
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoded_input = tokenizer(text, return_tensors='pt')

# infer
output = model(**encoded_input)

# detoken
print(f"output type:{type(output)}")
 # output type:<class 'transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions'>

print(output.keys())