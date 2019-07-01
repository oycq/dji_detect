import torch
import cv2
import time
import progressbar
from torchsummary import summary
import torch.nn as nn
import my_dataloader
import torch.optim as optim
from my_model import Model as FMD

model = torch.load('../data/history/2019-06-30 10:16:25.401992/7:4000') 
model.eval()
model = model.cuda().half()
progress = progressbar.ProgressBar(maxval=len(my_dataloader.test_loader.dataset)/10, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

total_count = 0
total_correct_count = 0
aaa = [0,0,0]

progress.start()
for i, (input_batch, label_batch) in enumerate(my_dataloader.test_loader):
    input_batch = input_batch.cuda()
    input_batch = input_batch.half() / 255 
    label_batch = label_batch.cuda().squeeze()
    outputs = model(input_batch)
    _, predicted = torch.max(outputs,1)
    for j in range(3):
        aaa[j] += (predicted == j).sum()
    predicted = (predicted >= 1)
    label_batch = (label_batch >= 1)
    incorrect_count = (predicted != label_batch).sum().item() 
#    if incorrect_count == 0:
#        continue
#    input_rwt_error = input_batch[predicted != label_batch] * 255
#    label_rwt_error = predicted[predicted != label_batch]
#    input_rwt_error = input_rwt_error.cpu().numpy().astype('uint8').transpose((0,2,3,1))
#    cv2.imshow(str(label_rwt_error[0].item()),input_rwt_error[0])
#    return_key = cv2.waitKey(0)
#    if return_key == ord(' '):
#        pass
#    if return_key == ord('q'):
#        break
#    print(incorrect_count)
    total_count += 10
    total_correct_count += 10 - incorrect_count
    progress.update(i)
progress.finish()
print('finished')
print(aaa)
print('total:%d correct:%.2f%%'%(total_count,total_correct_count/total_count*100))


