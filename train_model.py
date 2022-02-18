import torch

from data import train_dataloader
from train import train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Starting training.....")
train(train_dataloader)



