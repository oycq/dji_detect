import os
import time
import glob
import torch
from torch.utils.data import Dataset
from torch.utils import data
import torchvision
import cv2
import numpy as np

batch_size = 5

train_files_list = np.load('../data/image_name_list.npy')
mark_data = np.load('../data/mark.npy')
label = []
for item in train_files_list:
    label.append(int(item.split('/')[-3]))
label = np.array(label)
keep = ((label > 0) == (mark_data.sum(1) > 0))
train_files_list = train_files_list[keep]
mark_data = mark_data[keep]
label = label[keep]


class MyDataset(Dataset):
    def __init__(self, files_list, mark_data, label):
        self.files_list = files_list 
        self.mark_data = mark_data
        self.label = label

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, index):
        image = cv2.imread(self.files_list[index])
        X = torch.tensor(image.transpose((2,0,1)))
        mask = torch.zeros((2,40,60)) - 1
        if self.label[index] != 0:
            mask[self.label[index] -1,mark_data[index][1] // 32 : mark_data[index][3] // 32 + 1,mark_data[index][0] // 32 : mark_data[index][2] // 32 + 1] = 1
        return X, mask

train_set = MyDataset(train_files_list, mark_data, label)

train_loader = data.DataLoader(train_set, batch_size, shuffle=True, num_workers = 5, drop_last=True)

if __name__ == '__main__':
    max_epochs = 1000
    for i, (input_batch, mask_batch) in enumerate(train_loader):
        print(i)

#if __name__ == '__main__':
#    max_epochs = 1000
#    for i, (input_batch, label_batch) in enumerate(train_loader):
#        raw = input_batch[0]
#        reframe = raw.data.numpy().astype('float32')*255
#        reframe = reframe.astype('uint8')
#        reframe = reframe.transpose(1,2,0)
#        print(raw.shape,str(label_batch[0].numpy()[0]))
#        cv2.imshow(str(label_batch[0].numpy()[0]),reframe)
#        return_key = cv2.waitKey(0)
#        if return_key == ord(' '):
#            pass
#        if return_key == ord('q'):
#            break
