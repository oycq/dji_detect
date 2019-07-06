import torch
import os
import my_model
import progressbar
from torchsummary import summary
import torch.nn as nn
import my_dataloader
import torch.optim as optim
import datetime
import wandb

wandb.init()

batch_size = my_dataloader.batch_size
model = my_model.Model().cuda().half()
wandb.watch(model)
#model.load_state_dict(torch.load('../data/history/2019-07-05 23:44:17.697687/1:6000.model'))
model.train()
optimizer = optim.Adam(model.parameters(),lr = 0.0003, eps=1e-5)
#optimizer.load_state_dict(torch.load('../data/history/2019-07-05 23:44:17.697687/1:6000.adam'))
criterion = nn.CrossEntropyLoss().cuda().half()
bar = progressbar.ProgressBar(maxval=len(my_dataloader.test_loader.dataset)/batch_size, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

history_directory = '../data/history/%s'%datetime.datetime.now()
os.mkdir(history_directory)

def test(model):
    with torch.no_grad():
        bar.start()
        model.eval()
        correct_count = 0 
        sum_count = 0
        for j, (input_batch, label_batch) in enumerate(my_dataloader.test_loader):
            input_batch = input_batch.cuda()
            input_batch = input_batch.half()
            input_batch = input_batch / 255
            label_batch = label_batch.cuda().squeeze()
            loss_1, outputs = model(input_batch)
            loss = criterion(outputs, label_batch)
            _, predicted = torch.max(outputs,1)
            sum_count += batch_size
            correct_count += (predicted == label_batch).sum().item()
            bar.update(j)
        bar.finish()
        return correct_count / sum_count * 100


for epoch in range(200):
    print('----- epoch %d -----'%epoch) 
    for i, (input_batch, label_batch) in enumerate(my_dataloader.train_loader):
        print("%6d"%i,end = '\r')
        model.train()
        input_batch = input_batch.cuda()
        input_batch = input_batch.half() / 255
        label_batch = label_batch.cuda().squeeze()
        loss_1, outputs = model(input_batch)
        _, predicted = torch.max(outputs,1)
        loss = criterion(outputs, label_batch)
        L = loss + loss_1 / 5
        optimizer.zero_grad() 
        L.backward()
        optimizer.step()


        wandb.log({'i':i,
                   'train_loss':loss.item(),
                   'train_loss_1':loss_1.item(),
                   'train_L':L.item(),
                   'train_accuracy':(predicted == label_batch).sum().item()/batch_size*100.0})
        if i % 2000 == 0:
            torch.save(optimizer.state_dict(),'%s/%d:%d.adam'%(history_directory,epoch,i))
        if i % 500 == 0:
            torch.save(model.state_dict(),'%s/%d:%d.model'%(history_directory,epoch,i))
            os.system('cp *.py "%s"'%history_directory)
            test_accuracy = test(model)
            print("test_accuracy: %.2f"%test_accuracy)
            wandb.log({'test_accuracy':test_accuracy})

