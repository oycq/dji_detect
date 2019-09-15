import threading
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init
import torch.optim as optim
import random
import os
import time
import wandb

wandb.init()
lstm_width = 256
lstm_layers = 3
input_width = 2
internal_width = 5
output_width = 2
pre_length = 20
suf_length = 8
cuda = 1
database_size = 5000000
internal_params_name = {'time':0, 'object':1, 
                        'imu_speed_x':2, 'imu_speed_y':3, 'imu_speed':4}
train_batch_size = 500
old_history_ratio = 0.95
energy_loss_k = 1 / 60 / 40000
w_to_loss_exp = - 1


class AiController():
    def __init__(self):
        self.database = self.Database()
        self.ai = self.Ai()
#        train_thread = threading.Thread(target = self.train, daemon = True)
#        train_thread.start()
        self.ai.model.load_state_dict(torch.load('../data/supper/50000.model'))
        self.ai.optimizer.load_state_dict(torch.load('../data/supper/50000.adam'))

    def train(self): #return l == (loss, predict_loss, optimize_loss)
        i = 0 
        while(1):
            with self.database.lock:
                a = self.database.get_train_batch()
            if a is None:
                continue
#                print('Need preheat..')
#                os._exit(0)
            if i % 10000 == 0:
                torch.save(self.ai.optimizer.state_dict(),'%s/%d.adam'%('../data/supper',i))
                torch.save(self.ai.model.state_dict(),'%s/%d.model'%('../data/supper',i))
            l = self.ai.train(a['prior'], a['input'], a['output'])
            i += 1

        
    def get_optimized_control(self):
        a = self.database.get_current_prior_info()
        if a is None:
            print('Need preheat..')
            os._exit(0)
        c = self.ai.get_optimized_control(a)
        return c

    def append_internal_data(self,data):
        self.database.append_internal_data(data)
    
    def append_output_data(self,data):
        self.database.append_output_data(data)

    def append_input_data(self,data):
        self.database.append_input_data(data)

    
    class Database():
        def __init__(self):
            self.lock = threading.Lock()
            self.input_array = np.zeros((database_size, input_width), dtype ='float32')
            self.internal_array = np.zeros((database_size, internal_width), dtype ='float32')
            self.output_array = np.zeros((database_size, output_width), dtype ='float32')
            self.merged_array = np.zeros((database_size,input_width + internal_width + output_width), dtype ='float32')
            self.array_pointer = -1
            self.input_batch = np.zeros((database_size, suf_length, input_width), dtype ='float32')
            self.prior_batch = np.zeros((database_size, pre_length, input_width + internal_width + output_width), dtype ='float32')
            self.output_batch = np.zeros((database_size, suf_length, output_width), dtype ='float32')
            self.batch_pointer = 0


        def append_input_data(self, data):
            with self.lock:
                if self.array_pointer >= 0: 
                    self.input_array[self.array_pointer] = data
                    self.merged_array[self.array_pointer,:input_width] = data
        
        def append_internal_data(self, data):
            with self.lock:
                if self.array_pointer >= 0: 
                    self.internal_array[self.array_pointer] = data
                    self.merged_array[self.array_pointer,input_width : input_width + internal_width] = data

        def append_output_data(self, data):
            with self.lock:
                if self.array_pointer >= 0: 
                    self.output_array[self.array_pointer] = data
                    self.merged_array[self.array_pointer,-output_width:] = data
                if self._check_can_append_to_batch():
                    self.input_batch[self.batch_pointer] = self.input_array[self.array_pointer - suf_length + 1 : self.array_pointer + 1, :]
                    self.output_batch[self.batch_pointer] = self.output_array[self.array_pointer - suf_length + 1 : self.array_pointer + 1, :]
                    self.prior_batch[self.batch_pointer] = self.merged_array[self.array_pointer - suf_length - pre_length + 1 : self.array_pointer - suf_length + 1, :]
                    self.batch_pointer += 1
                self.array_pointer += 1
        
        def _check_can_append_to_batch(self):
            if self.array_pointer < pre_length + suf_length:
                return False
            bar = self.internal_array[self.array_pointer - suf_length + 1 : self.array_pointer + 1, internal_params_name['object']]
            if np.sum(bar) == suf_length:
                return True
            else:
                return False

        def get_train_batch(self):
            if self.batch_pointer == 0:
                return None
            if self.batch_pointer > train_batch_size:
                index = random.sample(range(self.batch_pointer - int(train_batch_size * (1 - old_history_ratio))), int(train_batch_size * old_history_ratio))
                index += [int(self.batch_pointer - train_batch_size * (1 - old_history_ratio) + x) for x in range(int(train_batch_size - train_batch_size * old_history_ratio))]
            else:
                index = range(self.batch_pointer)
            if cuda == 0: 
                batch_data = {'prior':torch.as_tensor(self.prior_batch[index]), 'input':torch.as_tensor(self.input_batch[index]), 'output':torch.as_tensor(self.output_batch[index])}
            else:
                batch_data = {'prior':torch.as_tensor(self.prior_batch[index]).cuda(), 'input':torch.as_tensor(self.input_batch[index]).cuda(), 'output':torch.as_tensor(self.output_batch[index]).cuda()}
            return batch_data

        def get_current_prior_info(self):
            if self.array_pointer < pre_length:
                return None
            data = self.merged_array[self.array_pointer - pre_length : self.array_pointer]
            data = data.reshape((1, pre_length, internal_width + input_width + output_width))
            if cuda == 0:
                data = torch.as_tensor(data)
            else:
                data = torch.as_tensor(data).cuda()
            return data


    class Ai():
        def __init__(self):
            super().__init__()
            self.lock = threading.Lock()
            self.model = self.Model() 
            self.optimizer = optim.Adam(self.model.parameters())
            if cuda:
                self.model.cuda()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                w = input_width + internal_width + output_width
                self.lstm_a = nn.LSTM(w, lstm_width, batch_first = True, num_layers = lstm_layers)
                self.lstm_b = nn.LSTM(input_width, lstm_width, batch_first = True, num_layers=lstm_layers)
                self.linear_a = nn.Linear(lstm_width, output_width)
                b0 = nn.Linear(lstm_width, lstm_width,)
                b1 = nn.Linear(lstm_width, lstm_width)
                b2 = nn.Linear(lstm_width, output_width)
                self._init_params(b0)
                self._init_params(b1)
                self._init_params(b2)
                sequent = [b0,nn.Dropout(),nn.ReLU(),b1,nn.Dropout(),nn.ReLU(),b2]
                self.linear_b = nn.Sequential(*sequent)
                self.w_to_loss = torch.tensor(np.exp(np.arange(suf_length) / suf_length * w_to_loss_exp)).cuda().unsqueeze_(-1).repeat(1,2).float()
                self.w_to_loss /= torch.mean(self.w_to_loss)
                self._init_params(self.lstm_b)
                self._init_params(self.linear_a)
#                self._init_params(self.linear_b)
            
            def _init_params(self, layer):
                if isinstance(layer, nn.LSTM):
                    for param in layer.parameters():
                        if len(param.shape) >= 2:
                            nn.init.orthogonal_(param.data)
                        else:
                            nn.init.normal_(param.data)
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight.data)
                    nn.init.normal_(layer.bias.data)
                
            
            def forward(self, prior_batch, input_batch = None, output_batch = None):
                if input_batch is not None:
                    a, (h_n, c_n) = self.lstm_a(prior_batch)
                    b, _ = self.lstm_b(input_batch, (h_n, c_n))
                    predict = self.linear_a(b)

                    for param in self.lstm_a.parameters():
                        param.requires_grad = False
                    for param in self.lstm_b.parameters():
                        param.requires_grad = False
                    for param in self.linear_a.parameters():
                        param.requires_grad = False
                    a, (h_n, c_n) = self.lstm_a(prior_batch)
                    b = self.linear_b(c_n[-1])
                    optimize_control = b.view(-1, 1, output_width)#.fill_(0.5)
                    optimize_control = optimize_control.repeat(1, suf_length, 1)
                    c, _ = self.lstm_b(optimize_control, (h_n, c_n))
                    optimize_result = self.linear_a(c)
                    for param in self.lstm_a.parameters():
                        param.requires_grad = True
                    for param in self.lstm_b.parameters():
                        param.requires_grad = True
                    for param in self.linear_a.parameters():
                        param.requires_grad = True

                    predict_loss = torch.mean(self.w_to_loss * (predict- output_batch).pow(2))
                    optimize_loss = torch.mean(self.w_to_loss * optimize_result.pow(2))
                    energy_loss = (torch.mean(optimize_control.pow(2))) * energy_loss_k * (predict_loss * 10000)
                    loss = predict_loss + optimize_loss + energy_loss
                    #print("%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f"%(input_batch[-1][0][0],optimize_control[-1][0][0],optimize_result[-1][0][0], predict[-1][0][0], output_batch[-1][0][0],predict_loss, optimize_loss, loss))
                    print("t:%6.2f  ix:%7.2f  ox:%8.4f%%  iy:%7.2f  oy:%8.4f  pl:%8.4f%%  ol:%8.4f%%  el:%8.4f l:%8.4f"%(prior_batch[-1][0][2] * 50, input_batch[-1][0][0],output_batch[-1][0][0]*100,input_batch[-1][0][1],output_batch[-1][0][1]*100, (predict_loss*10000)**0.5, (optimize_loss*10000)**0.5, energy_loss * 10000, loss))
                    wandb.log({'input_x':input_batch[-1][0][0],
                               'input_y':input_batch[-1][0][1],
                               'o_x':output_batch[-1][0][0].pow(2),
                               'o_y':output_batch[-1][0][1].pow(2),
                               'pl':(predict_loss*10000)**0.5,
                               'ol':(optimize_loss*10000)**0.5,
                               'el':energy_loss * 10000,
                              })

#                    print('loss:',(predict[-1]-output_batch[-1])* 100)
#                    print('predict:',predict[-1] * 100)
#                    print('real:',output_batch[-1] * 100)
#                    print('op_control:',optimize_control[-1])
#                    print('op_result:',optimize_result[-1]*100)
                    return loss, predict_loss, optimize_loss, energy_loss

                else:
                    with torch.no_grad():
                        a, (h_n, c_n) = self.lstm_a(prior_batch)
                        b = self.linear_b(c_n[-1]) 
                        optimize_control = b.view(-1, 1, output_width)#.fill_(0.5)
                        optimize_control = optimize_control.repeat(1, suf_length, 1)

                    #c, _ = self.lstm_b(optimize_control, (h_n, c_n))
                    #optimize_result = self.linear_a(c)[0]

                    return optimize_control.detach().cpu().numpy()[0]
                
        def train(self, prior_batch, input_batch, output_batch):
            loss, predict_loss, optimize_loss, energy_loss = self.model(prior_batch,input_batch,output_batch)
            self.optimizer.zero_grad()
            loss.backward()
            with self.lock:
               self.optimizer.step()
            return loss.item(), predict_loss.item(), optimize_loss.item()
        
        def get_optimized_control(self, prior_batch):
            with self.lock:
                optimize_control = self.model(prior_batch)
            return optimize_control

