import cv2
import torch
import os
import time 
import 

def resize_frame(frame):
    return frame[80:1360]

def video_to_disk(input_path,vidoe_name,output_path):
    cap = cv2.VideoCapture(input_path+'/'+video_name)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(frame_count):
        status, frame = cap.read()
        if status != 1:
            print(i,0)
            continue
        frame = resize_frame(frame)
        f16_tensor = torch.tensor(frame.transpose((2,0,1)) / 255.0,dtype = torch.float16)
        torch.save(f16_tensor,output_path+"/"+video_name.split('.')[0]+'_'+str(i)+'.pt')
        print(i) 
    cap.release()

def tensor_to_mat(tensor):
    reframe = tensor.data.numpy().astype('float32')*255
    reframe = reframe.astype('uint8')
    reframe = reframe.transpose(1,2,0)
    return reframe


if __name__ == "__main__":#load 100 frame in disk and get its time
#    vidoes_names = ['gh010083.mp4' , 'gh010085.mp4' , 'gh010086.mp4' , 'gh010087.mp4']
#    video_path = '../data'
#    output_path = '../data/disk'
#    items_name = os.listdir(output_path)
#    k = 0
#    time_last = time.time()
#    for item in items_name:
#        tensor = torch.load(output_path + '/' + item)
#        k = k+1
#        if k % 10 == 0:
#            print(k,time.time()-time_last)
#            time_last = time.time()


