import torch
import torch.nn as nn
import my_model
import test_dataloader
import torch.optim as optim 

model = my_model.Model()
model.eval()
model.load_state_dict(torch.load('../data/history/2019-07-07 23:33:24.843776/3:13500.model',map_location={'cuda:0': 'cpu'}))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
for j, (input_batch, label_batch) in enumerate(test_dataloader.test_loader):
    input_batch = input_batch.float()
    input_batch = input_batch / 255
    label_batch = label_batch.squeeze()
    loss_1, outputs = model(input_batch)
    loss = criterion(outputs, label_batch)# + loss_1
#    print(loss)
#    loss = loss_1
#    print(loss)

    loss.backward()
    optimizer.step()
    for param in model.parameters():
        print(param.grad_fn)
        break
    break

#    print(values[i].shape)
#for i,module in enumerate(model.features.children()):
#    print(i,'....................')
#    for k in module.state_dict():
#        print(k)
