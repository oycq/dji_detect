import torch
import torch.nn as nn
cfg = [8,8, 'M', 16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 'v', 'M']
num_classes = 3 

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.features = self._make_layers(cfg, batch_norm=True)
        self.classifier = nn.Sequential(
            nn.Linear(2 * 15 * 10, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        loss_1 = x.sum()
        x = self.classifier(x)
        return loss_1, x 

    def _make_layers(self,cfg,batch_norm):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'v':
                conv2d = nn.Conv2d(256 , 2, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                continue
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

