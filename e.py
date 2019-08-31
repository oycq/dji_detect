import os
import time
import torch
from torch.utils.data import Dataset
from torch.utils import data
import torchvision
import numpy as np
import b as my_model
import d as new_model
import matplotlib.pyplot as plt
import glob
import torch.optim as optim
t = time.time() * 2000

l_before = 50
l_after = 30
batch_size = 1000
log_list = glob.glob('.data/*.log')
samples = []

for log_file in log_list:
    data_array = np.loadtxt(log_file, dtype = 'float32')
    data_array[:,0] /= 50
    data_array[:,2] /= 600
    data_array[:,3] /= 960
    data_array[:,4] /= 200
    data_array[:,5] /= 200


    data_array[:,6:] = (data_array[:,6:] - np.average(data_array[:,6:], 0)) / np.std(data_array[:,6:], 0)

    for i in range(len(data_array) - 2 - l_after - l_before):

        a = data_array[i + 1: i + l_before + 1, [0, 1, 2, 3 ,6, 7, 8]]
        b = data_array[i : i + l_before, [4, 5]]
        prior_info = np.concatenate((a, b), 1)
        control = data_array[i+ l_before :i + l_before + l_after, [4, 5]]
        control_effect = data_array[i+ l_before + 1 :i + 1 + l_before + l_after, [2, 3]]
        invalid = data_array[i+ l_before + 1 :i + 1 + l_before + l_after, [1]]
        if np.sum(invalid > 0) == 0:
            samples.append([prior_info, control, control_effect])


class MyDataset(Dataset):
    def __init__(self, samples):
        self.samples =samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        prior_info = self.samples[index][0] 
        control = self.samples[index][1]
        control_effect = self.samples[index][2]
        return prior_info, control, control_effect

train_samples = samples[0:len(samples) // 3 * 2]
test_samples = samples[len(samples) // 3 * 2:]
train_set = MyDataset(train_samples)
train_loader = data.DataLoader(train_set, batch_size, shuffle=False, num_workers = 8, drop_last=True)
test_set = MyDataset(test_samples)
test_loader = data.DataLoader(test_set, batch_size, shuffle=False, num_workers = 8, drop_last=True)




model = my_model.Model().cuda()
model.load_state_dict(torch.load('../data/lstm_history/2019-08-29 10:45:01.081341/1090:7.model'))
new_model = new_model.Model().cuda()
new_model.load_state_dict(torch.load('fuck.model'))
new_optimizer = optim.Adam(new_model.parameters())

#model.eval()
if __name__ == '__main__':

    figure,axes = plt.subplots(1,2)
    for k in range(20):
        print(' ')
        print(k)
        for i, input_batch in enumerate(train_loader):
            control = input_batch[1].cuda()
     
            control.fill_(0)
            control.requires_grad = True
            optimizer = optim.Adam([control])
            input_batch = [input_batch[0].cuda(),control.cuda(),input_batch[2].cuda()]

            for j in range(50):
                loss,predict = model(input_batch)
                optimizer.zero_grad() 
                loss.backward()
                optimizer.step()
            
            new_input_batch  = [input_batch[0], predict[:,0].detach()]
            for i in range(50):
                new_loss,e = new_model(new_input_batch)
                print(new_loss.item() * 200 ,end = '\r')
                new_optimizer.zero_grad() 
                new_loss.backward()
                new_optimizer.step()
            print('*******train  ', e[0].detach() * 200,predict[0,0].detach()*200)
        for i, input_batch in enumerate(test_loader):
            control = input_batch[1].cuda()
     
            control.fill_(0)
            control.requires_grad = True
            optimizer = optim.Adam([control])
            input_batch = [input_batch[0].cuda(),control.cuda(),input_batch[2].cuda()]

            for j in range(50):
                loss,predict = model(input_batch)
                optimizer.zero_grad() 
                loss.backward()
                optimizer.step()
            
            new_input_batch  = [input_batch[0], predict[:,0].detach()]
            for i in range(50):
                new_loss,e = new_model(new_input_batch)
                print(new_loss.item() * 200 ,end = '\r')
                new_optimizer.zero_grad() 
                new_loss.backward()
                new_optimizer.step()
            print('*******test  ', e[0].detach() * 200,predict[0,0].detach()*200)
    torch.save(new_model.state_dict(),'fuck.model')

