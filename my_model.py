import torch
import torch.nn as nn
cfg = [8, 'M', 16, 'M', 32, 'M', 64, 'M', 32, 'M', 2]
num_classes = 3 

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.features = self._make_layers(cfg, batch_norm=True)
        self.threshold = torch.nn.Parameter(torch.Tensor([1]))
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),x.size(1),-1)
        dense = (x > 0).sum().float() / x.numel()
        peak, _ = x.max(dim = 2)
        second_large, _ = x.topk(2, dim = 2)
        second_large = second_large.sum(dim = 2) - peak
        sencond_class, _ = x.topk(100, dim = 2)
        sencond_class = sencond_class.mean(dim = 2)
        predict = peak - second_large + peak - sencond_class
        threshold = self.threshold.repeat(x.size(0),1)
        x = torch.cat((predict,threshold), dim = 1)
        return dense,x

    def _make_layers(self,cfg,batch_norm):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'v':
                conv2d = nn.Conv2d(16 , 4, kernel_size=3, padding=1, bias=True)
                layers += [conv2d, nn.ReLU(inplace=False)]
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
