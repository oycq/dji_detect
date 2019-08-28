import torch
import os
import b as my_model
import progressbar
from torchsummary import summary
import torch.nn as nn
import a as my_dataloader
import torch.optim as optim
import datetime
import wandb

wandb.init()

batch_size = my_dataloader.batch_size
model = my_model.Model().cuda()
#wandb.watch(model)
#model.load_state_dict(torch.load('../data/history/2019-07-10 16:51:25.453881/1:20000.model'))
model.train()
optimizer = optim.Adam(model.parameters())
#optimizer.load_state_dict(torch.load('../data/history/2019-07-10 16:51:25.453881/1:20000.adam'))

history_directory = '../data/lstm_history/%s'%datetime.datetime.now()
os.mkdir(history_directory)
os.system('cp *.py "%s"'%history_directory)

#def test(model):
#    with torch.no_grad():
#        bar.start()
#        model.eval()
#        correct_count = 0 
#        sum_count = 0
#        for j, (input_batch, label_batch) in enumerate(my_dataloader.test_loader):
#            input_batch = input_batch.cuda()
#            input_batch = input_batch.float()
#            input_batch = input_batch / 255
#            label_batch = label_batch.cuda().squeeze()
#            dense, outputs = model(input_batch)
##            loss = criterion(outputs, label_batch)
#            _, predicted = torch.max(outputs,1)
#            sum_count += batch_size
#            correct_count += (predicted == label_batch).sum().item()
#            bar.update(j)
#        bar.finish()
#        return correct_count / sum_count * 100

#loss_ori = 10
for epoch in range(1000000000):
    print('----- epoch %d -----'%epoch) 
    for i, input_batch in enumerate(my_dataloader.train_loader):
        train_loss = model(input_batch)
        optimizer.zero_grad() 
        train_loss.backward()
        optimizer.step()
        print("train   %15.3f%%"%(train_loss * 100), end = '\r')
    print('\n')
    for i, input_batch in enumerate(my_dataloader.test_loader):
        test_loss = model(input_batch)
        print("test   %15.3f%%"%(test_loss * 100), end = '\r')
    print('\n')
    wandb.log({'epoch':epoch,
               'train_loss':train_loss.item(),
               'test_loss':test_loss.item(),
                })
    if epoch % 10 == 0:
        torch.save(optimizer.state_dict(),'%s/%d:%d.adam'%(history_directory,epoch,i))
        torch.save(model.state_dict(),'%s/%d:%d.model'%(history_directory,epoch,i))



