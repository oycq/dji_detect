import os
import torch
from torch.utils.data import Dataset
from torch.utils import data

file_name_list   = os.listdir('../data/disk')
label_list       = []

train_index_list = []
test_index_list  = []
valid_index_list = []

for file_name in file_name_list:#fill label_list
    if file_name[6:8] in ['83','85']:
        label_list.append(torch.tensor([1], dtype=torch.float16))
    if file_name[6:8] in ['86','87']:
        label_list.append(torch.tensor([0], dtype=torch.float16))

for i, file_name in enumerate(file_name_list):#file train,test,valid list
    if file_name[6:8] in ['83','86']:
        train_index_list.append(i)
    if file_name[6:8] in ['85','87']:
        if int(file_name.split('.')[0].split('_')[1]) % 10 == 0:
            valid_index_list.append(i)
        else:
            test_index_list.append(i)

#check samples balance
positive_label = 0
nagative_label = 0
for index in train_index_list:
    if label_list[index].numpy() == True:
        positive_label += 1
    else:
        nagative_label += 1
print('Check samples balance:',positive_label,nagative_label)


class MyDataset(Dataset):
    def __init__(self, index_list):
        self.index_list = index_list

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, pos):
        X = torch.load('../data/disk/' + file_name_list[self.index_list[pos]])
        y = label_list[self.index_list[pos]]
        return X, y

train_set = MyDataset(train_index_list)
test_set  = MyDataset(test_index_list)
valid_set = MyDataset(valid_index_list)

train_loader = data.DataLoader(train_set, 20, True, num_workers = 2)
test_loader  = data.DataLoader(test_set , 20, True, num_workers = 2)
valid_loader = data.DataLoader(valid_set, 20, True, num_workers = 2)

if __name__ == '__main__':
    max_epochs = 1000
    for i, (input_batch, label_batch) in enumerate(train_loader):
        print(i)
