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

#path = "../../Downloads/test1/*"
path = 'webcam0'

model = torchvision.models.vgg16_bn()
model.load_state_dict(torch.load('../data/vgg16_bn-6c64b313.pth'))
model.cuda().half()
model.eval()

module_id = 0
module_name = '' 
channel_id = 0
page_id = 0
modules_output = []
modules_name = []

page = np.zeros((1920,2000), dtype = 'uint8')
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
    def __init__(self, path = 'webcam'):
        if path[0:6] == 'webcam':
            self.cap = cv2.VideoCapture(int(path[-1]))
            print('ok')
            self.type = 'webcam'
        else:
            self.image_list = glob.glob("../../Downloads/test1/*")
            self.type = 'images'
            self.image_pointer = 0

    def read(self):
        if self.type == 'webcam':
            _, image = self.cap.read()
            image = cv2.flip(image[16:464,96:544],1)
            image = cv2.resize(image, (224,224))
            return image
        if self.type == 'images':
            image = cv2.imread(self.image_list[self.image_pointer])
            image = cv2.resize(image, (224,224))
            return image
    
    def next(self):
        if self.type == 'images':
            self.image_pointer = (self.image_pointer + 1) % len(self.image_list)
feeder = imageFeeder(path)

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
    if plus == 1:
        channel_id = (channel_id + 1) % 64 
    else:
        channel_id = (channel_id - 1) % 64

def change_page(plus):
    global page_id
    if plus == 1:
        page_id = (page_id + 1) % (len(modules_output[module_id]) // 64)
    else:
        page_id = (page_id - 1) % (len(modules_output[module_id]) // 64)

def show_page():
    global page
    page[:,:] = 255

    if 'ReLU' in modules_name[module_id] or 'Norm' in modules_name[module_id]:
        temp = modules_output[module_id] * 85
    else:
        temp = modules_output[module_id] * 42 + 128
    image_list_np = temp.cpu().detach().numpy().astype('uint8')

    for i in range(64):
        row = i // 8 
        col = i % 8  
        if i == channel_id:
            page[row * 240 : row * 240 +240, col * 250 : col * 250 + 250] = 0
        channel = i + page_id * 64
        page[row * 240 : row * 240 +224, col * 250 : col * 250 + 224] = \
                cv2.resize(image_list_np[channel], (224, 224))
    cv2.imshow('output', cv2.resize(image_list_np[channel_id + page_id * 64], (896, 896), 
        interpolation= cv2.INTER_NEAREST)) 
    cv2.imshow('page',page)
        
while(1):
    t_start = time.time()*1000
    image = feeder.read()
    cv2.imshow('frame',cv2.resize(image,(896,896)))
    image = torch.tensor(image.transpose((2,0,1)), dtype = torch.float).cuda()#....
    image = image / 255
    image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])(image)
    x = image.unsqueeze(0)
    x = x.half() 
    modules_output = []
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
    print("%7.2f %-20s %5d  %4d -%4d   %-10s"%
            (t_end - t_start, module_name, channel_id + page_id * 64,
            page_id * 64, page_id * 64 + 64, class_label[:40]), end = '\r')

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
