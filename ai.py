import gimbal
import cv2
import camera
import math
import time
import datetime
import supper
import random
import numpy as np
 
controller = supper.AiController()
cam = camera.Camera()
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
currentDT = str(datetime.datetime.now())
log_file = open('.data/' + currentDT + '.log', 'x')
gimbal.speed_control(0,0,0)
iii = 0
while(1):
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
        internal_data = np.array([(time.time() * 1000 - time_ori) / 50, 0, imu_speed[0]/40, imu_speed[1]/40, imu_speed[2]/40])
        output_data = np.array([x/600, y/960])
        input_data = np.array([0,0])
        controller.append_internal_data(internal_data)
        controller.append_output_data(output_data)
        controller.append_input_data(input_data)
        gimbal.speed_control(0,0,0)
        print("*********  %5.2f ms"%(time.time() * 1000 - time_ori),end = '\n')
        log_file.write("%f %f %f %f %f %f %f %f %f\n" %
                 (time.time() * 1000 - time_ori, 1, 0, 0, 0, 0,
                imu_speed[0], imu_speed[1], imu_speed[2]))
        continue
    else:
        iii += 1
        internal_data = np.array([(time.time() * 1000 - time_ori) / 50, 1, imu_speed[0]/40, imu_speed[1]/40, imu_speed[2]/40])
        output_data = np.array([x/600, y/960])

        controller.append_internal_data(internal_data)
        controller.append_output_data(output_data)

        if iii > 24:
            controller.train()
            sss = controller.get_optimized_control()
            sx = sss[0]
            sy = sss[1]
        else:
            sx = random.gauss(0, 5)
            sy = random.gauss(0, 5)

#        print("%5d  %5d   %5.2f ms"%(x, y, time.time() * 1000 - time_ori),end = '\n')   
        u = v = 0
        u = x / 3
        v = y / 3
        if sx < -20:
            sx = - random.gauss(10, 5)
        if sx > 20:
            sx = random.gauss(10, 5)
        if sy < -20:
            sy = -random.gauss(10, 5)
        if sy > 20:
            sy = random.gauss(10, 5)
        if random.gauss(0, 1) < - iii / 1000:
            sx = random.gauss(0, 5)
            sy = random.gauss(0, 5)


        input_data = np.array([sx,sy])
        controller.append_input_data(input_data)
        gimbal.speed_control(u+sx,0,v+sy)

        controller.append_input_data(input_data)

        log_file.write("%f %f %f %f %f %f %f %f %f\n" %
                 (time.time() * 1000 - time_ori,0 , x, y, u, v,
                imu_speed[0], imu_speed[1], imu_speed[2]))
        continue

       
