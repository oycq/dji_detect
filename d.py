import torch
import os
import torch.nn as nn
import torch.nn.init as init

lstm_h = 100
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lstm_a = nn.LSTM(9, lstm_h, 1, batch_first = True)
        self.linear1 = nn.Linear(lstm_h, 1024, bias=True)
        self.linear2 = nn.Linear(1024, 2, bias=True)
        self._weight_init(self.lstm_a)
        self._weight_init(self.linear1)
        self._weight_init(self.linear2)

    def forward(self, data):
        prior_info = data[0]
#        print('1111',prior_info.shape)
        prefer_control = data[1]
#        print(prefer_control.shape)
        a, (h_n, c_n) = self.lstm_a(prior_info)
        c_n = c_n[0,:]
#        print('2222',c_n.shape)
        d = self.linear1(c_n)
#        print('3333',d.shape)
        c = self.linear2(d)
#        print('4444',c.shape)
#        asd()
        loss = torch.mean((c - prefer_control).abs())
        return loss,c

    def _weight_init(self, m):
        '''
        Usage:
            model = Model()
            model.apply(weight_init)
        '''
        if isinstance(m, nn.Conv1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm3d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.LSTMCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRU):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRUCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)

if __name__ == '__main__':
    ha = Model().cuda()
    for i, data in enumerate(a.train_loader):
        loss = ha(data)
        print(loss)
