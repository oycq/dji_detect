import torch
import cv2
import time
import progressbar
from torchsummary import summary
import torch.nn as nn
import test_dataloader as my_dataloader
import torch.optim as optim
from torch.autograd import Variable

cfg = [8, 'M', 16, 'M', 32, 'M', 64, 'M', 32, 'M', 16, 'M', 8, 'M']
num_classes = 3 

class FMD(nn.Module):

    def __init__(self):
        super(FMD, self).__init__()
        self.features = self._make_layers(cfg, batch_norm=True)
        self.classifier = nn.Sequential(
            nn.Linear(8 * 15 * 10, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layers(self,cfg,batch_norm):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

fmd = torch.load('../data/history/2019-06-30 10:16:25.401992/7:4000') 
fmd.eval()
fmd = fmd.cuda().half()
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
    outputs = fmd(input_batch)
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


