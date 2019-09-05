import torch
import numpy as np
import torch.nn as nn
import torch.nn.init
import torch.optim as optim
import random

lstm_width = 50
input_width = 2
internal_width = 5
output_width = 2
pre_length = 15
suf_length = 5
cuda = 0 
control_optimized_times = 20
database_size = 100000
ai_affluence = 10
internal_params_name = {'time':0, 'object':1, 
                        'imu_speed_x':2, 'imu_speed_y':3, 'imu_speed':4}
train_batch_size = 50
optimize_loss_k = 0.1


class AiController():
    def __init__(self):
        self.database = self.Database()
        self.ai = self.Ai()
        pass

    def train(self): #return l == (loss, predict_loss, optimize_loss)
        a = self.database.get_train_batch()
        l = self.ai.train(a['prior'], a['input'], a['output'])
        return l
        
    def get_optimized_output(self, previous): #return c == next_control
        a = self.database.get_train_batch()
        c = self.ai.train(a['prior'])
        return c[0]

    def append_internal_data(self,data):
        self.database.append_internal_data(data)
    
    def append_output_data(self,data):
        self.database.append_output_data(data)

    def append_input_data(self,data):
        self.database.append_input_data(data)

    
    class Database():
        def __init__(self):
            self.input_array = np.zeros((database_size, input_width))
            self.internal_array = np.zeros((database_size, internal_width))
            self.output_array = np.zeros((database_size, output_width))
            self.merged_array = np.zeros((database_size,input_width + internal_width + output_width))
            self.array_pointer = -1
            self.input_batch = np.zeros((database_size, suf_length, input_width))
            self.prior_batch = np.zeros((database_size, pre_length, input_width + internal_width + output_width))
            self.output_batch = np.zeros((database_size, suf_length, output_width))
            self.batch_pointer = 0


        def append_input_data(self, data):
            if self.array_pointer >= 0: 
                self.input_array[self.array_pointer] = data
                self.merged_array[self.array_pointer,:input_width] = data
        
        def append_internal_data(self, data):
            if self.array_pointer >= 0: 
                self.internal_array[self.array_pointer] = data
                self.merged_array[self.array_pointer,input_width : input_width + internal_width] = data

        def append_output_data(self, data):
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
            if self.batch_pointer > train_batch_size * 2:
                index = random.sample(range(self.batch_pointer - train_batch_size), train_batch_size)
                index += [self.batch_pointer - train_batch_size + x for x in range(train_batch_size)]
            else:
                index = range(self.batch_pointer)
            batch_data = {'prior':self.prior_batch[index], 'input':self.input_batch[index], 'output':self.output_batch[index]}
            return batch_data

        def get_prior_info(self):
            data = self.prior_batch[self.batch_pointer - 1]

    class Ai():
        def __init__(self):
            self.model = self.Model() 
            self.optimizer = optim.Adam(self.model.parameters())
            if cuda:
                self.model.cuda()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                w = input_width + internal_width + output_width
                self.lstm_a = nn.LSTM(w, lstm_width, num_layers = 1, batch_first = True)
                self.lstm_b = nn.LSTM(w, lstm_width, num_layers = 1, batch_first = True)
                self.linear_a = nn.Linear(lstm_width, output_width)
                self.linear_b = nn.Linear(lstm_width, output_width * suf_length)
                self._init_params(self.lstm_a)
                self._init_params(self.lstm_b)
                self._init_params(self.linear_a)
                self._init_params(self.linear_b)
            
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
                    b, _ = self.lstm_b(control_batch, (h_n, c_n))
                    predict = self.linear_a(b)[:,:,0,:]
                    predict_loss = torch.mean((input_batch - output_batch).abs())

                    self.lstm_a.requires_grad = False
                    self.lstm_b.requires_grad = False
                    self.linear_a.requires_grad = False 
                    a, (h_n, c_n) = self.lstm_a(prior_batch)
                    b = self.linear_b(c_n[0])
                    b = torch.nn.functional.tanh(b) * ai_affluence
                    optimize_control = b.view(-1, output_width, suf_length)
                    c, _ = self.lstm_b(optimize_control, (h_n, c_n))
                    optimize_result = self.linear_a(c)[:,:,0,:]
                    self.linear_a.requires_grad =  optimize_loss = torch.mean(optimize_result.abs())
                    self.lstm_a.requires_grad = True
                    self.lstm_b.requires_grad = True
                    self.linear_a.requires_grad = True 

                    loss = predict_loss - optimize_loss * optimize_loss_k
                    return loss, predict_loss, optimize_loss

                else:
                    a, (h_n, c_n) = self.lstm_a(prior_batch)
                    b = self.linear_b(c_n[0]) 
                    b = torch.nn.functional.tanh(b) * ai_affluence
                    optimize_control = b.view(-1, output_width, suf_length)
                    return optimize_control.numpy()
                
        def train(self, prior_batch, input_batch, output_batch):
            loss, predict_loss, optimize_loss = self.model(prior_batch,input_batch,output_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.item(), predict_loss.item(), optimize_loss.item()
        
        def get_optimized_control(self, prior_batch):
            optimize_control = self.model(prior_batch)
            return optimize_control
