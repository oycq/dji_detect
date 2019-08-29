import torch
import os
import b as my_model
from torchsummary import summary
import torch.nn as nn
import a as my_dataloader
import torch.optim as optim
import datetime


batch_size = my_dataloader.batch_size
model = my_model.Model()
#wandb.watch(model)
model.load_state_dict(torch.load('../data/lstm_history/2019-08-29 10:45:01.081341/1090:7.adam'))
model.eval()
optimizer = optim.Adam(model.parameters())

input_batch = my_dataloader.train_loader[0]
train_loss = model(input_batch)



