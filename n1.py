import gimbal
import cv2
import camera
import math
import time
import datetime
 

cam = camera.Camera()
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
currentDT = str(datetime.datetime.now())
log_file = open('.data/' + currentDT + '.log', 'x')
gimbal.speed_control(0,0,0)
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
        gimbal.speed_control(0,0,0)
        print("*********  %5.2f ms"%(time.time() * 1000 - time_ori),end = '\r')
        log_file.write("%f %f %f %f %f %f %f %f %f\n" %
                 (time.time() * 1000 - time_ori, 1, 0, 0, 0, 0,
                imu_speed[0], imu_speed[1], imu_speed[2]))
        continue
    else:
        print("%5d  %5d   %5.2f ms"%(x, y, time.time() * 1000 - time_ori),end = '\r')   
        u = v = 0
        if abs(x / 1200 * 30.3) > 0:
            u = x / 3
        if abs(y / 1200 * 30.3) > 0:
            v = y / 3
            gimbal.speed_control(u,0,v)
        log_file.write("%f %f %f %f %f %f %f %f %f\n" %
                 (time.time() * 1000 - time_ori,0 , x, y, u, v,
                imu_speed[0], imu_speed[1], imu_speed[2]))
        continue

       
