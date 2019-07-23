import torch
import my_model
import my_dataloader
import glob
import matplotlib.pyplot as plt
import matplotlib
import time
import torch.nn as nn
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, State, Output#, Event
import numpy as np

#generate = True
generate = False
history_folder = '../data/history/2019-07-11 23:54:58.574675'# no slash\

def arange_weight_histroy_file():
    model = my_model.Model()
    model.train().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    model_list = glob.glob(history_folder + '/*.model')
    def sort_func(model_path_name):
        batch_num,i_num = model_path_name.split('/')[-1].split('.')[0].split(':')
        return int(batch_num) * 9999999 + int(i_num)
    model_list.sort(key = sort_func)
    name_list = []
    parameters_list = []
    grad_list = []
    model.load_state_dict(torch.load(model_list[0]))#,map_location={'cuda:0': 'cpu'}))
    for name, parameters in model.named_parameters():
        name_list.append(name)
        parameters_list.append([])
        grad_list.append([])
    for i, model_path in enumerate(model_list):
        model.load_state_dict(torch.load(model_path,map_location={'cuda:0': 'cpu'}))
        for j, (input_batch, label_batch) in enumerate(my_dataloader.test_loader):
            input_batch = input_batch.cuda()
            input_batch = input_batch.float()
            input_batch = input_batch / 255
            label_batch = label_batch.cuda().squeeze()
            dense, outputs = model(input_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            if j == 3:
                break
        for j,parameters in enumerate(model.parameters()):
            parameters.grad = parameters.grad / 4 
            parameters_list[j].append(parameters.cpu().clone().detach())
            grad_list[j].append(parameters.grad.cpu().clone())

        print('load model parameters:%5.2f%%'%(i / len(model_list) * 100), end = '\r')
    print('\n')
    for i in range(len(parameters_list)):
        parameters_list[i] = torch.stack(parameters_list[i]).numpy()
        grad_list[i] = torch.stack(grad_list[i]).numpy()
    torch.save([name_list,parameters_list,grad_list], history_folder + '/weight_history.wh')


if generate == True: 
    arange_weight_histroy_file()

name_list, parameters_list , grad_list = torch.load(history_folder + '/weight_history.wh')

ods = [0, 0, 0, 0, 0]
ids = [0, 0, 0, 0, 0]
display_string = ''
x = np.array(range(len(parameters_list[0]))) 
y_w = None
y_grad = None
weight_history_figure = {
    'data':
    [{'x' : None, 'y' : None, 'type': 'lines+markers'}],
    'layout':
    {'title': 'Weights'}
}
grad_history_figure = {
    'data':
    [{'x' : None, 'y' : None, 'type': 'lines+markers'}],
    'layout':
    {'title': 'Gradients'}
    }

def after_enter_key(key): 
    global ids,display_string,x,y_w,y_grad 
    if key not in 'qwertasdfg':
        return
    key_dict = {'q':10,'a':20,'w':11,'s':21,'e':12,'d':22,'r':13,'f':23,'t':14,'g':24}
    Id = key_dict[key] % 10
    add_sub = 1
    if key_dict[key] // 10 == 1:
        add_sub = -1
    if Id ==0:
        ods[0] += add_sub
        ids[0] = ods[0] % len(name_list)
        dimension = len(parameters_list[ids[0]].shape) - 1
        for i in range(dimension):
            ids[i+1] = ods[i+1] % parameters_list[ids[0]].shape[i+1]
    dimension = len(parameters_list[ids[0]].shape) - 1
    if Id <= dimension and Id != 0:
        ods[Id] += add_sub
        ids[Id] = ods[Id] % parameters_list[ids[0]].shape[Id]
    display_string = '%-25s' % name_list[ids[0] % len(name_list)] 
    for i in range(4):
        if i < dimension:
            display_string += '%6d' % (ids[i+1] + 1)
        else:
            display_string += '%6d' % (-1)
    if dimension == 1:
        y_w = parameters_list[ids[0]][:,ids[1]]
        y_grad = grad_list[ids[0]][:,ids[1]]
    if dimension == 2:
        y_w = parameters_list[ids[0]][:,ids[1],ids[2]]
        y_grad = grad_list[ids[0]][:,ids[1],ids[2]]
    if dimension == 3:
        y_w = parameters_list[ids[0]][:,ids[1],ids[2],ids[3]]
        y_grad = grad_list[ids[0]][:,ids[1],ids[2],ids[3]]
    if dimension == 4:
        y_w = parameters_list[ids[0]][:,ids[1],ids[2],ids[3],ids[4]]
        y_grad = grad_list[ids[0]][:,ids[1],ids[2],ids[3],ids[4]]
    weight_history_figure['data'][0]['x'] = x
    weight_history_figure['data'][0]['y'] = y_w
    grad_history_figure['data'][0]['x'] = x
    grad_history_figure['data'][0]['y'] = y_grad

after_enter_key('q')
after_enter_key('a')

app = dash.Dash()
external_stylesheets = ['template.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

app.layout = html.Div(children = [
    dcc.Input(id= 'keyboard',
        type = 'text', 
        size = '4' ,
        value = '',
    ),
    html.H1(
        children = 'Weight visualization',
        style = {'textAlign': 'center'}
    ),
    html.H3(
        id = 'ids',
        children = display_string,
        style = {'textAlign': 'center','whiteSpace': 'pre'}
    ),
    dcc.Graph(
        id = 'w_figure',
        figure = weight_history_figure
    ),
    dcc.Graph(
        id = 'grad_figure',
        figure = grad_history_figure
    )
])

@app.callback([Output('ids', 'children'),
                Output('w_figure', 'figure'),
                Output('grad_figure', 'figure')],
              [Input('keyboard', 'value')])
def key_press_recall(string):
    if len(string) > 0:
        after_enter_key(string[-1])
    return display_string, weight_history_figure, grad_history_figure

if __name__ == '__main__':
    app.run_server(port = 8080, debug=True)
