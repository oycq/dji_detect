import torch
import progressbar
from torchsummary import summary
import torch.nn as nn
import test_loader as my_dataloader
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

for i, (input_batch, label_batch) in enumerate(my_dataloader.test_loader):
    input_batch = input_batch.cuda().half() / 255
    label_batch = label_batch.cuda().squeeze()
    outputs = fmd(input_batch)
    loss = criterion(outputs, label_batch)
    _, predicted = torch.max(outputs,1)
    input_rwt_error = input_batch[predicted != label_batch].float().cpu()
    print(input_rwt_error.shape,predicted != label_batch)


