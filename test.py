import cv2
import torch

vidoes_names = ['GH010083.MP4' , 'GH010085.MP4' , 'GH010086.MP4' , 'GH010087.MP4']
video_path = '../data'

def tensor_to_mat(tensor):
    reframe = tensor.data.numpy().astype('float32')*255
    reframe = reframe.astype('uint8')
    reframe = reframe.transpose(1,2,0)
    return reframe

def video_to_disk(input_path,vidoe_name,output_path):
    cap = cv2.VideoCapture(input_path+'/'+video_name)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(frame_count):
        status, frame = cap.read()
        if status != 1:
            print(i,0)
            continue
        f16_tensor = torch.tensor(frame.transpose((2,0,1)) / 255.0,dtype = torch.float16)
        torch.save(f16_tensor,output_path+"/"+video_name.split('.')[0]+str(i)+'.pt')
        print(i) 
    cap.close()


for video_name in vidoes_names:
    video_to_disk(video_path, video_name, '../data/disk')

#    cap = cv2.VideoCapture(video_path + '/' + video_name)
#    #print(cap.get(cv2.CAP_PROP_FPS)) 29.9
#    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#    print(video_path,frame_count * 1440 * 1920 * 3 * 2.0 / 1024 / 1024 / 1024)
#
#    for i in range(frame_count):
#        status, frame = cap.read()
#        print(i, status)
#        if status != 1:
#            continue
#        #print(frame.shape) 1440*1920*3
#        #print(type(frame),frame.dtype)
#        #cv2.imshow('frame',frame)
#        return_key = cv2.waitKey(0) & 0xff
#        data = torch.tensor(frame.transpose((2,0,1)) / 255.0, dtype = torch.float16)
#        reframe = tensor_to_mat(data)
#        cv2.imshow('reframe',reframe)
#
#         
#        print(data.shape)
#        if return_key == ord(' '):
#            pass
#        if return_key == ord('q'):
#            break



