import os
import time
import glob
import torch
from torch.utils.data import Dataset
from torch.utils import data
import torchvision
import numpy as np

l_before = 50
l_after = 30
batch_size = 1000
log_list = glob.glob('.data/*.log')
samples = []
for log_file in log_list:
    data_array = np.loadtxt(log_file, dtype = 'float32')
    data_array = (data_array - np.average(data_array, 0)) / np.std(data_array, 0)
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
train_set = MyDataset(samples)
train_loader = data.DataLoader(train_set, batch_size, shuffle=True, num_workers = 5, drop_last=True)
test_set = MyDataset(samples)
test_loader = data.DataLoader(test_set, batch_size, shuffle=True, num_workers = 5, drop_last=True)

if __name__ == '__main__':
    max_epochs = 1000
    for i, _ in enumerate(train_loader):
        print(i, _[2].shape)

