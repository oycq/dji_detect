import torch
import cv2
import time
import progressbar
from torchsummary import summary
import torch.nn as nn
import my_dataloader
import torch.optim as optim
from torch.autograd import Variable

cfg = [8, 'M', 16, 'M', 32, 'M', 64, 'M', 32, 'M', 16, 'M', 8, 'M']
num_classes = 2

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

fmd = FMD().cuda().half()
fmd.load_state_dict(torch.load('../data/history/check_point12'))
optimizer = optim.Adam(fmd.parameters(),lr = 0.0003, eps=1e-5)
criterion = nn.CrossEntropyLoss().cuda().half()
bar = progressbar.ProgressBar(maxval=len(my_dataloader.test_loader.dataset)/10, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

for i, (input_batch, label_batch) in enumerate(my_dataloader.train_loader):
    input_batch = input_batch.cuda()
    input_batch = input_batch.half() / 255 
    label_batch = label_batch.cuda().squeeze()
    outputs = fmd(input_batch)
    loss = criterion(outputs, label_batch)
    _, predicted = torch.max(outputs,1)
    incorrect_count = (predicted != label_batch).sum().item() 
    if incorrect_count == 0:
        continue
    input_rwt_error = input_batch[predicted != label_batch] * 255
    label_rwt_error = predicted[predicted != label_batch]
    input_rwt_error = input_rwt_error.cpu().numpy().astype('uint8').transpose((0,2,3,1))
    cv2.imshow(str(label_rwt_error[0].item()),input_rwt_error[0])
    return_key = cv2.waitKey(0)
    if return_key == ord(' '):
        pass
    if return_key == ord('q'):
        break

    print(incorrect_count)


