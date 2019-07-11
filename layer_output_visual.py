import cv2
import torch
import torchvision
import glob
import cv2
import time
import torch.nn as nn
import tkinter as tk
from torchvision.models import vgg
import threading
import imagenet1000
import numpy as np 
import glob
import my_model
#from my_model import Model as FMD


config1 = {
        'path' : '../../Downloads/test1/*',
        #'path' : '0',
        'page_rows' : 16,
        'page_cols' : 16,
        'box_size' : [0, 0],
        'image_size' : (112, 112),
        'input_size' : (224, 224),
        'enlarged_size' : (896,896),
        'pix_k': 42,
        }
config2 = {
#        'path' : '../data/mp4/GH010083.MP4',
#        'path' : '../data/mp4/GH010083.MP4',
        'path' : '../data/mp4/GH010083.MP4',
        'page_rows' : 4,
        'page_cols' : 4,
        'box_size' : [0, 0],
        'image_size' : (600, 400),
        'input_size' : (1920, 1280),
        'enlarged_size' : (960,640),
        'pix_k': 50,
        }
config = config2
#model = torchvision.models.vgg16_bn()
#model.load_state_dict(torch.load('../data/vgg16_bn-6c64b313.pth'))
#model = torch.load('../data/history/2019-06-30 10:16:25.401992/7:4000')
#model = torch.load('../data/history/2019-07-05 11:57:55.955671/0:9000')
model = my_model.Model()
model.cuda().half()
model.load_state_dict(torch.load('../data/history/2019-07-10 19:56:03.207301/10:25000.model'))
#model.eval()
#model.features[25].train()
model.train()

for i in range(2):
    if config['box_size'][i] == 0: 
        config['box_size'][i] = int(config['image_size'][i] * 1.1)

module_id = 0
module_name = '' 
channel_id = 0
page_id = 0
modules_output = []
modules_name = []

page_size = (config['page_rows'] * config['box_size'][1],
             config['page_cols'] * config['box_size'][0])
page = np.zeros(page_size, dtype = 'uint8')
cv2.namedWindow('frame')
cv2.namedWindow('output')
cv2.namedWindow('page')
cv2.moveWindow('frame',0,0)
cv2.moveWindow('output',0,1000)
cv2.moveWindow('page',1300,0)
cv2.imshow('page', page)


for i in range(len(model.features)):
    modules_name.append(model.features[i].__class__.__name__)

module_name = '[' +str(module_id) + ']' + modules_name[module_id]

class imageFeeder():
    def __init__(self, config):
        self.config = config
        if config['path'][-1] == '*':
            self.image_list = glob.glob(config['path'])
            self.type = 'images'
            self.image_pointer = 0
        else:
            self.type = 'video'
            if len(config['path']) == 1:
                self.cap = cv2.VideoCapture(int(config['path']))
            else:
                self.cap = cv2.VideoCapture(config['path'])


    def read(self):
        if self.type == 'video':
            while(1):
                status, image = self.cap.read()
                if status == True:
                    break
            if len(self.config['path']) == 1:
                image = cv2.flip(image[16:464,96:544], 1)
            image = cv2.resize(image, self.config['input_size'])
            return image
        if self.type == 'images':
            image = cv2.imread(self.image_list[self.image_pointer])
            image = cv2.resize(image, config['input_size'])
            return image
   
    def next(self):
        if self.type == 'images':
            self.image_pointer = (self.image_pointer + 1) % len(self.image_list)


def change_module(plus):
    global module_id, module_name, channel_id
    if plus == 1:
        module_id = (module_id + 1) % len(modules_output)
    else:
        module_id = (module_id - 1) % len(modules_output)
    channel_id = channel_id % len(modules_output[module_id])
    module_name = '[' +str(module_id) + ']' + modules_name[module_id]\
            + '(' + str(len(modules_output[module_id])) + ')'
    
def change_channel(plus):
    global channel_id
    channels_per_page = config['page_rows'] * config['page_cols']
    if plus == 1:
        channel_id = (channel_id + 1) % channels_per_page
    else:
        channel_id = (channel_id - 1) % channels_per_page

def change_page(plus):
    global page_id
    channels_per_page = config['page_rows'] * config['page_cols']
    if plus == 1:
        page_id = (page_id + 1) % (len(modules_output[module_id]) // channels_per_page)
    else:
        page_id = (page_id - 1) % (len(modules_output[module_id]) // channels_per_page)

def show_page():
    global page
    page[:,:] = 255

    if 'ReLU' in modules_name[module_id] or 'Norm' in modules_name[module_id]:
        temp = modules_output[module_id] * config['pix_k'] * 2
    else:
        temp = modules_output[module_id] * config['pix_k'] + 128
    temp[temp > 255] = 255
    temp[temp < 0] = 0
    image_list_np = temp.cpu().detach().numpy().astype('uint8')

    for i in range(config['page_rows'] * config['page_cols']):
        row = i // config['page_cols']
        col = i % config['page_cols'] 
        box_h = config['box_size'][0]
        box_w = config['box_size'][1]
        image_h = config['image_size'][0]
        image_w = config['image_size'][1]
        bias_channel = i + page_id * config['page_cols'] * config['page_rows']
        if i == channel_id:
            page[row * box_w : row * box_w + box_w, col * box_h : col * box_h + box_h] = 0
        if bias_channel >= len(image_list_np):
            continue
        page[row * box_w : row * box_w +image_w, col * box_h : col * box_h + image_h] = \
                cv2.resize(image_list_np[bias_channel], (image_h, image_w),
                        interpolation= cv2.INTER_NEAREST)
    select_image_id = channel_id + page_id * config['page_cols'] * config['page_rows']
    if select_image_id < len(image_list_np):
        cv2.imshow('output', cv2.resize(image_list_np[select_image_id], config['enlarged_size'],
         interpolation= cv2.INTER_NEAREST)) 
    cv2.imshow('page',page)
        

feeder = imageFeeder(config)
while(1):
    t_start = time.time()*1000
    image = feeder.read()
    cv2.imshow('frame',cv2.resize(image,config['enlarged_size']))
    image = torch.tensor(image.transpose((2,0,1)), dtype = torch.float).cuda()#....
    image = image / 255
    image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])(image)
    x = image.unsqueeze(0)
    x = x.half() 
    modules_output = []
    with torch.no_grad():
        for i in range(len(model.features)):
            x = model.features[i](x)
            modules_output.append(x[0])
        x = x.view(x.size(0), -1)
        x = model.classifier(x)
        _, predicted = torch.max(x, 1)
    show_page()

    t_end = time.time()*1000
    print("                                                                                                ",end = '\r')
    class_label = imagenet1000.d[predicted.item()]
    print("%7.2f %-20s %5d  %-10s"%
            (t_end - t_start, module_name, 
                channel_id + page_id * config['page_rows'] * config['page_cols'],
                class_label[:40]), end = '\r')

    key = cv2.waitKey(10)
    if key == ord('q'):
        break
    if key == ord('j'):
        change_module(1)
    if key == ord('k'):
        change_module(0)
    if key == ord('h'):
        change_channel(0)
    if key == ord('l'):
        change_channel(1)
    if key == ord('u'):
        change_page(0)
    if key == ord('d'):
        change_page(1)
    if key == ord(' '):
        feeder.next()


    #print("%40s %5d"%(module_name,channel_id))
