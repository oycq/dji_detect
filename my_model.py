import torch
import torch.nn as nn
cfg = [8, 'M', 16, 'M', 32, 'M', 64, 'M', 32, 'M', 16, 'M', 'v', 'M']
num_classes = 3 

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.features = self._make_layers(cfg, batch_norm=True)
        self.classifier = nn.Sequential(
            nn.Linear(4 * 15 * 10, 128),
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
        loss_1 = (x > 0).sum().float() / x.shape[0] / x.shape[1]
        x = self.classifier(x)
        loss_2 = -1/torch.max(self.features[25].bias)
        return loss_1,loss_2,x 

    def _make_layers(self,cfg,batch_norm):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'v':
                conv2d = nn.Conv2d(16 , 4, kernel_size=3, padding=1, bias=False)
                layers += [conv2d, nn.BatchNorm2d(4), nn.ReLU(inplace=True)]
                continue
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for i,m in enumerate(self.modules()):
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
        for name,param in self.named_parameters():
            if name == 'features.25.weight':
                param.requires_grad = False
            if name == 'features.25.bias':
                param.data = torch.Tensor([-2,-2,-2,-2])


