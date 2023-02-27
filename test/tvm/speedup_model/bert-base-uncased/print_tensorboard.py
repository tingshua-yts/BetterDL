from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained("bert-base-uncased")
tokenize = BertTokenizer.from_pretrained("bert-base-uncased", torchscript=True)
inputs=tokenize(["My Name is Tom", "what is your first and last Name"], padding=True, return_tensors='pt')

# print(inputs)
# model.eval()
# output=model(inputs["input_ids"])
# print(type(output))
# print(size(output))


writer = SummaryWriter("tmp/bert-base-uscased/torchlogs/")

writer.add_graph(model, inputs["input_ids"], verbose=True, use_strict_trace=False)
writer.close()