import torch
import os
import my_model
import progressbar
from torchsummary import summary
import torch.nn as nn
import my_dataloader
import torch.optim as optim
import datetime

batch_size = 5 
model = my_model.Model().cuda().half()
optimizer = optim.Adam(model.parameters(),lr = 0.0003, eps=1e-5)
criterion = nn.CrossEntropyLoss().cuda().half()
bar = progressbar.ProgressBar(maxval=len(my_dataloader.test_loader.dataset)/batch_size, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

history_directory = '../data/history/%s'%datetime.datetime.now()
os.mkdir(history_directory)
log_file = open(history_directory + '/log.txt', 'w+')
for epoch in range(200):
    print('----- epoch %d -----'%epoch) 
    log_file.write('----- epoch %d -----\n')
    correct_count = 0
    sum_count = 0
    for i, (input_batch, label_batch) in enumerate(my_dataloader.train_loader):
        model = model.train()
        input_batch = input_batch.cuda()
        input_batch = input_batch.half() / 255
        label_batch = label_batch.cuda().squeeze()
        outputs = model(input_batch)
        _, predicted = torch.max(outputs,1)
        sum_count += batch_size
        correct_count += (predicted == label_batch).sum().item()
        loss = criterion(outputs, label_batch)
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print('%d, %10.9f %.2f%%'%(i,float(loss.cpu()),correct_count/sum_count*100.0))
            log_file.write('%d, %10.9f %.2f%%\n'%(i,float(loss.cpu()),correct_count/sum_count*100.0))
            correct_count = 0
            sum_count = 0
        if i % 1000 == 0:
            torch.save(model,'%s/%d:%d'%(history_directory,epoch,i))
            bar.start()
            model = model.eval()
            for i, (input_batch, label_batch) in enumerate(my_dataloader.test_loader):
                input_batch = input_batch.cuda()
                input_batch = input_batch.half()
                label_batch = label_batch.cuda().squeeze()
                outputs = model(input_batch)
                loss = criterion(outputs, label_batch)
                print(outputs)
                _, predicted = torch.max(outputs,1)
                print(predicted)
                sum_count += batch_size
                correct_count += (predicted == label_batch).sum().item()
                bar.update(i)
                #print(i,loss.item(),correct_count,correct_count/sum_count)
            print('testing accurate: %.2f%%'%(correct_count/sum_count*100.0))
            log_file.write('testing accurate: %.2f%%\n'%(correct_count/sum_count*100.0))
            bar.finish()
            correct_count = 0
            sum_count = 0

