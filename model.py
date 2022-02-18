import torch

from transformers import BertForTokenClassification

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model2 = model =BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=7)
model2.to(device)

def get_model():
    return model2

