import os
import glob
import torch
from torch.utils.data import Dataset
from torch.utils import data
import cv2

train_files_list = []
test_files_list = []

categories = [0, 1]
train_files_in_each_category = []
short_board_number = 99999999 # caculate this number for samples balance
for item in categories:
    files = glob.glob('../data/train/%d/*/*'%item)
    train_files_in_each_category.append(files)
    if len(files) < short_board_number:
        short_board_number = len(files) 
for item in train_files_in_each_category:
    train_files_list += item[:short_board_number]
print(len(train_files_list))

test_files_list = glob.glob('../data/test/*/*/*')

class MyDataset(Dataset):
    def __init__(self, files_list):
        self.files_list = files_list 

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, index):
        image = cv2.imread(self.files_list[index])
        X = torch.tensor(image.transpose((2,0,1)) / 255.0, dtype = torch.float16)
        label = self.files_list[index].split('/')[-3]
        y = torch.tensor([int(label)], dtype=torch.long)
        return X, y

train_set = MyDataset(train_files_list)
test_set  = MyDataset(test_files_list)

train_loader = data.DataLoader(train_set, 10, True, num_workers = 5)
test_loader  = data.DataLoader(test_set , 10, True, num_workers = 5)

if __name__ == '__main__':
    max_epochs = 1000
    for i, (input_batch, label_batch) in enumerate(test_loader):
        print(i)
#
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
