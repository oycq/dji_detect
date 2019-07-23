import torch
import os
import my_model as my_model
import progressbar
from torchsummary import summary
import torch.nn as nn
import my_dataloader as my_dataloader
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

history_directory = '../data/history/%s'%datetime.datetime.now()
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

loss_ori = 10
for epoch in range(200):
    print('----- epoch %d -----'%epoch) 
    for i, (input_batch, mask_batch) in enumerate(my_dataloader.train_loader):
        print("%6d"%i,end = '\r')
        input_batch = input_batch.cuda()
        input_batch = input_batch.float() / 255
        mask_batch = mask_batch.cuda().squeeze()
        loss,info = model(input_batch, mask_batch)
        if len(info) == 1:
            print('\n***%d***i\n'%i)
            continue
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()


        wandb.log({'i':i,
                   'train_loss':loss.item(),
                   'train_loss_min':info[0].item(),
                   'train_loss_max':info[1]})
        if i % 500 == 0:
            torch.save(optimizer.state_dict(),'%s/%d:%d.adam'%(history_directory,epoch,i))
            torch.save(model.state_dict(),'%s/%d:%d.model'%(history_directory,epoch,i))

