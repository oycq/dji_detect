import torch
import numpy as np
import torch.nn as nn
import torch.nn.init
import torch.optim as optim

lstm_width = 50
input_width = 2
internal_width = 6
output_width = 2
pre_length = 15
suf_length = 5
cuda = 0 
control_optimized_times = 20
database_size = 100000
ai_affluence = 10
internal_params_name = {'time':0, 'object':1, 'ai_affluence':2,
                        'imu_speed_x':3, 'imu_speed_y':4, 'imu_speed':5}
train_batch_size = 100


class AiController():
    def __init__(self):
        self.database = self.Database()
        self.ai = self.Ai()
        pass

    def train(self):
        batch_data = self.data.get_train_batch()
        prior_batch = batch_data['prior']
        input_batch = batch_data['input']
        real_output_batch = batch_data['output']
        predict_batch = self.ai.model(prior_batch, input_batch)
        loss = self.ai.model.loss_function(predict_batch, real_output_batch)
        self.ai.model_optimizer.zero_grad() 
        loss.backward()
        self.ai.model_optimizer.step()

    def get_optimized_output(self, previous):
        self.ai.to_be_optimized_control.fill_(0)
        for i in range(control_optimized_times):
            predict = self.ai.model(prior_batch, to_be_optimized_control)
            loss = torch.mean(predict)
            self.ai.control_optimizer.zero_grad()
            loss.backward()
            self.ai.control_optimizer.step()
        return self.ai.to_be_optimized_control

    def append_input_data(self,data):
        self.database.append_input_data(data)

    def append_internal_data(self,data):
        self.database.append_internal_data(data)
    
    def append_output_data(self,data):
        self.database.append_output_data(data)
    
    class Database():
        def __init__(self):
            self.input_array = np.zeros((database_size, input_width))
            self.internal_array = np.zeros((database_size, internal_width))
            self.output_array = np.zeros((database_size, output_width))
            self.merged_array = np.zeros((database_size,input_width+internal_width+output_width))
            self.array_pointer = -1
            self.input_batch = np.zeros((database_size, suf_length, input_width))
            self.prior_batch = np.zeros((database_size, pre_length, input_width+internal_width))
            self.output_batch = np.zeros((database_size, suf_length, output_width))
            self.batch_pointer = 0


        def append_input_data(self, data):
            if self.pointer >= 0: 
                self.input_array[self.array_pointer] = data
                self.merged_array[self.array_pointer,:input_width] = data
        
        def append_internal_data(self, data):
            if self.array_pointer >= 0: 
                self.internal_array[self.array_pointer] = data
                self.merged_array[self.array_pointer,input_width:input_width+input_width] = data

        def append_output_data(self, data):
            if self.array_pointer >= 0: 
                self.output_array[self.array_pointer] = data
                self.merged_array[self.array_pointer,-output_width:] = data
            if self._check_can_append_to_batch():
                self.input_batch[self.batch_pointer] = self.input_array[self.array_pointer - suf_length + 1:, :]
                self.output_batch[self.batch_pointer] = self.output_array[self.array_pointer - suf_length + 1:, :]
                self.prior_batch[self.batch_pointer] = self.merged_array[self.array_pointer - suf_length - pre_length + 1 : self.array_pointer - suf_length + 1, :]
            self.array_pointer += 1
            self.batch_pointer += 1
        
        def _check_can_append_to_batch(self):
            bar = internal_array[self.array_pointer - suf_length + 1 :, internal_params_name['object']]
            if np.sum(bar) == suf_length:
                return True
            else:
                return False

        def get_train_batch(self):
            if self.train_batch_size < 2 * self.batch_pointer:
                index = random.sample(range(self.batch_pointer - self.train_batch_size), self.train_batch_size)
                index += [self.batch_pointer - x - 1 for x in range(self.train_batch_size)]
            else:
                index = [self.batch_pointer - x - 1 for x in range(2 * self.train_batch_size)]
            batch_data = {'prior':self.prior_batch[index], 'input':self.input_batch[index], 'output':self.output_batch[index]}
            return batch_data

        def get_prior_info(self):
            data = self.prior_batch[self.batch_pointer - 1]

    class Ai():
        def __init__(self):
            self.model = self.Model() 
            self.model_optimizer = optim.Adam(self.model.parameters())
            self.to_be_optimized_control = torch.zeros((1, suf_length, output_width))
            self.to_be_optimized_control.requires_grad = True
            self.control_optimizer= optim.Adam([self.to_be_optimized_control])
            if cuda:
                self.model.cuda()
                self.to_be_optimized_control.cuda()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                w = input_width + internal_width + output_width
                self.lstm_a = nn.LSTM(w, lstm_width, num_layers = 1, batch_first = True)
                self.lstm_b = nn.LSTM(w, lstm_width, num_layers = 1, batch_first = True)
                self.linear = nn.Linear(lstm_width, output_width)
                self._init_params(self.lstm_a)
                self._init_params(self.lstm_b)
                self._init_params(self.linear)
            
            def _init_params(self, layer):
                if isinstance(layer, nn.LSTM):
                    for param in layer.parameters():
                        if len(param.shape) >= 2:
                            nn.init.orthogonal_(param.data)
                        else:
                            nn.init.normal_(param.data)
                    print('init done 0 (2)')
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight.data)
                    nn.init.normal_(layer.bias.data)
                    print('init done 1')
            
            def forward(self, prior_batch, control_batch):
                a, (h_n, c_n) = self.lstm_a(prior_batch)
                b, _ = self.lstm_b(control_batch, (h_n, c_n))
                c = self.linear(b)[:,:,0,:]
                d = nn.functional.tanh(c) * ai_affluence
                print('the shape of forward output is %d'%d.shape)
                return d
            
            def loss_function(self, predict, real):
                loss = torch.mean((predict - real).abs())
                return loss



