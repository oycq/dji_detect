import os
import time
import glob
import torch
from torch.utils.data import Dataset
from torch.utils import data
import torchvision
import numpy as np
import b as my_model
import matplotlib.pyplot as plt
t = time.time() * 1000

l_before = 50
l_after = 30
batch_size = 1
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


if __name__ == '__main__':
    model = my_model.Model()
    model.load_state_dict(torch.load('../data/lstm_history/2019-08-29 10:45:01.081341/1090:7.model'))
    model.eval()
    with torch.no_grad():
        figure,axes = plt.subplots(1,2)
        for i, input_batch in enumerate(test_loader):
            loss,predict,control_effect = model(input_batch)
            control_effect = control_effect.numpy()[0,:,0]
            predict = predict.numpy()[0,:,0]
#            axes.scatter(range(control_effect.size),control_effect,label='effect')
#            axes.scatter(range(control_effect.size),predict,label='predict')
            axes[0].plot(range(control_effect.size),control_effect,label='effect')
            axes[0].plot(range(control_effect.size),predict,label='predict')
            axes[0].set_ylim(-0.1,0.1)
            axes[1].plot(range(control_effect.size),np.abs(predict-control_effect),label='delta')
            axes[1].set_ylim(0,0.02)
            axes[0].legend()
            plt.draw()
            plt.pause(0.2)
            axes[0].clear()
            axes[1].clear()

  #  plt.show()
