import gimbal
import cv2
import camera
import math
import time
import datetime
import b as model
import numpy as np
import torch
import torch.optim as optim
 

cam = camera.Camera()
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
currentDT = str(datetime.datetime.now())
log_file = open('.data/' + currentDT + '.log', 'x')
gimbal.speed_control(0,0,0)
model = model.Model()
model.load_state_dict(torch.load('../data/lstm_history/2019-08-29 10:45:01.081341/1090:7.model'))
prior_fifo = torch.zeros((1,50, 9))
preheat = 0
aaa = torch.zeros([1,2]).cuda()


while(1):
    preheat += 1
    time_ori = time.time() * 1000
    gimbal.request_datas()
    raw = cam.read()
    high_light = raw.max()
    image = raw.copy()
    image.fill(0)
    image[raw < 150] = 255 
    a = image.shape[0]
    b = image.shape[1]
    max_pos = raw.argmax()
    x = (max_pos // b) - a // 2
    y = (max_pos % b) - b // 2
    image[a // 2 - 2 : a // 2 + 2, :].fill(0)
    image[: , b // 2 - 2 : b // 2 + 2].fill(0)
    cv2.imshow('image',image)
    key = cv2.waitKey(2)
    imu_angle,imu_speed,imu_acc,joint_angle = gimbal.get_datas()
    if key == ord('q'):
        gimbal.speed_control(0,0,0)
        del cam
        break
    if key == ord('8'):
        gimbal.speed_control(-25,0,0)
        time.sleep(0.02)
        continue
    if key == ord('2'):
        gimbal.speed_control(25,0,0)
        time.sleep(0.02)
        continue
    if key == ord('4'):
        gimbal.speed_control(0,0,-25)
        time.sleep(0.02)
        continue
    if key == ord('6'):
        gimbal.speed_control(0,0,25)
        time.sleep(0.02)
        continue
    if high_light < 150 or abs(x) > 590 or abs(y) > 950:
        gimbal.speed_control(0,0,0)
#        print("*********  %5.2f ms"%(time.time() * 1000 - time_ori),end = '\r')
        log_file.write("%f %f %f %f %f %f %f %f %f\n" %
                 (time.time() * 1000 - time_ori, 1, 0, 0, 0, 0,
                imu_speed[0], imu_speed[1], imu_speed[2]))
        prior_fifo[-1] = torch.tensor([(time.time() * 1000 - time_ori)/50, 1, 0, 0, 0, 0,
                imu_speed[0]/40, imu_speed[1]/10, imu_speed[2]/40])
        prior_fifo[0,:-1] = prior_fifo[0,1:]
        continue
    else:
#        print("%5d  %5d   %5.2f ms"%(x, y, time.time() * 1000 - time_ori),end = '\r')   
        u = v = 0
        if abs(x / 1200 * 30.3) > 0:
            u = x / 3
        if abs(y / 1200 * 30.3) > 0:
            v = y / 3
            print(x,y)
            print("good  %20.3f %20.3f"%(u,v),end = '\n')
#            gimbal.speed_control(u,0,v)
        prior_fifo[0,-1,:4] = torch.tensor([(time.time() * 1000 - time_ori)/50, 0, x/600, y/960])
        prior_fifo[0,-1,6:9] = torch.tensor([imu_speed[0]/40, imu_speed[1]/10, imu_speed[2]/40])
        
        if preheat > 30:
            control = torch.zeros((1,50, 2))
            control.requires_grad = True
            sadiao = torch.zeros((1,50, 2))
            optimizer = optim.Adam([control])
            for j in range(50):
                loss,predict = model([prior_fifo, control,sadiao])
                optimizer.zero_grad() 
                loss.backward()
                optimizer.step()
            uu = predict[0,0,0]
            vv = predict[0,0,1]
            print("bad   %20.3f %20.3f\n"%(uu*200,vv*200),end = '\n')
#            print(prior_fifo[0,-1])
#            gimbal.speed_control(uu * 200,0,vv *200)

        prior_fifo[0,:-1] = prior_fifo[0,1:]
        prior_fifo[0,-1,4:6] = torch.tensor([u/200, v/200])

        log_file.write("%f %f %f %f %f %f %f %f %f\n" %
                 (time.time() * 1000 - time_ori,0 , x, y, u, v,
                imu_speed[0], imu_speed[1], imu_speed[2]))
        continue

       
