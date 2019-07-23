import torch
import os
import torch.nn as nn
cfg = [8, 'M', 16, 'M', 32, 'M', 64, 'M', 32, 'M', 'v']
num_classes = 3 

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.features = self._make_layers(cfg, batch_norm=True)
        self.threshold = torch.nn.Parameter(torch.Tensor([1]))
        self._initialize_weights()


    def forward(self, x, mask):
        x = self.features(x)
        x = x.view(x.size(0),x.size(1),-1)
        mask = mask.view(x.size(0),x.size(1),-1)
        x = x * mask
        x_min, _ = torch.min(x, dim = 2) 
        x_max, _ = torch.max(x, dim = 2) 
        if (x_max > 0).sum() > 0:
#            a = torch.exp(torch.log(- x_min).mean())
#            b = torch.exp(torch.log(x_max[x_max>0]).mean())
            a = - x_min.mean()
            b = 1 / torch.mean( 1 / x_max[x_max > 0])
            if (b - a) > 1:
                loss = (a) / (b - a)
            else:
                loss = a - b

            return loss, [a, b]

        else:
            return 0, [0]

    def _make_layers(self,cfg,batch_norm):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'v':
                conv2d = nn.Conv2d(32 , 2, kernel_size=3, padding=1, bias=False)
                layers += [conv2d, nn.BatchNorm2d(2, affine=False), nn.ReLU(inplace=False)]
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
                if m.weight is not None:
                   nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
