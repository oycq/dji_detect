import torch
import torchvision
import glob
import cv2
import time
import progressbar
from summary import *
import torch.nn as nn
import my_dataloader
import torch.optim as optim
import tkinter as tk
from my_model import Model as FMD
from torchvision.models import vgg
import threading

model = torchvision.models.vgg16_bn()
model.load_state_dict(torch.load('../data/vgg16_bn-6c64b313.pth'))
model.cuda().half()
model.eval()
print(model)
#summary(model, (3, 1920, 1280))

#image_list = glob.glob("../data/test/1/GH010085/*")
image_list = glob.glob("../../Downloads/test1/*")
image_id = 0
module_id = 0
channel_id = 0
modules_output = []
modules_name = []

cv2.namedWindow('ori')
cv2.namedWindow('output')
cv2.moveWindow('ori',0,0)
cv2.moveWindow('output',1920,0)

def foward(file_name):
    global modules_output,modules_name
    modules_output = []
    modules_name = []
    image = cv2.imread(file_name)
    image = cv2.resize(image, (224,224))
    cv2.imshow('ori', image)
    cv2.waitKey(10)
    image = torch.tensor(image.transpose((2,0,1)),dtype = torch.float) / 255
    image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])(image)

    x = image.unsqueeze(0)
    x = x.half().cuda()
    for i in range(len(model.features)):
        modules_output.append([])
        modules_name.append(model.features[i].__class__.__name__)
        x = model.features[i](x)
        for j in range(x.shape[1]):
            reframe = (x[0][j]*255).float().cpu().detach().numpy().astype('uint8')
            modules_output[-1].append(reframe)
    x = x.view(x.size(0), -1)
    x = model.classifier(x)
    #print(x[0].float().cpu().detach().numpy())

def update_visualization_window():
    image = modules_output[module_id][channel_id]
    image = cv2.resize(image, (1792,1792), interpolation= cv2.INTER_NEAREST)
    cv2.imshow('output', image)
    cv2.waitKey(20)

def change_image(plus):
    global image_id, image_name
    if plus == 1:
        image_id = (image_id + 1) % len(image_list)
    else:
        image_id = (image_id - 1) % len(image_list)
    image_name.set(image_list[image_id].split('/')[-3:])
    foward(image_list[image_id])
    update_visualization_window()

def change_module(plus):
    global module_id, module_name, channel_id, channel_name
    if plus == 1:
        module_id = (module_id + 1) % len(modules_output)
    else:
        module_id = (module_id - 1) % len(modules_output)
    channel_id = channel_id % len(modules_output[module_id])
    channel_name.set(str(channel_id))
    module_name.set('[' +str(module_id) + ']' + modules_name[module_id])
    update_visualization_window()

    
def change_channel(plus):
    global channel_id, channel_name
    if plus == 1:
        channel_id = (channel_id + 1) % len(modules_output[module_id])
    else:
        channel_id = (channel_id - 1) % len(modules_output[module_id])
    channel_name.set(str(channel_id))
    update_visualization_window()

def analysis_image():
    while(1):
        if (cv2.waitKey(20) == ord('a')):
            break

foward(image_list[image_id])
update_visualization_window()
root = tk.Tk()
root.geometry("500x500+3300+1500")
module_name = tk.StringVar(value = '[' +str(module_id) + ']' + modules_name[0])
image_name = tk.StringVar(value = image_list[1].split('/')[-3:])
channel_name = tk.StringVar(value = str(channel_id))

classifier_value = tk.StringVar(value = str(channel_id))
root.title('Control Panel')
tk.Label(root, textvariable = image_name, font=("Helvetica", 14)).pack()
tk.Label(root, textvariable = module_name, font=("Helvetica", 14)).pack()
tk.Label(root, textvariable = channel_name, font=("Helvetica", 14)).pack()
tk.Label(root, textvariable = classifier_value, font=("Helvetica", 14)).pack()
root.bind('<Up>', lambda event: change_module(0))
root.bind('<Down>', lambda event: change_module(1))
root.bind('<j>', lambda event: change_image(0))
root.bind('<k>', lambda event: change_image(1))
root.bind('<Left>', lambda event: change_channel(0))
root.bind('<Right>', lambda event: change_channel(1))
root.bind('<a>', lambda event: analysis_image())
root.mainloop()
