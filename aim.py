import gimbal
import cv2
import camera
import math
import time

cam = camera.Camera()
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
while(1):
    time_ori = time.time() * 1000
    raw = cam.read()
#    cv2.imshow('raw',raw)
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
    key = cv2.waitKey(5)
    if key == ord('q'):
        gimbal.angle_control(0,0,0,0,0,0)
        del cam
        break
    if key == ord('8'):
        gimbal.angle_control(0,0,0,-15,0,0)
        continue
    if key == ord('2'):
        gimbal.angle_control(0,0,0,15,0,0)
        continue
    if key == ord('4'):
        gimbal.angle_control(0,0,0,0,0,-15)
        continue
    if key == ord('6'):
        gimbal.angle_control(0,0,0,0,0,15)
        continue
    if high_light < 150 or abs(x) > 590 or abs(y) > 950:
        print("*********  %5.2f ms"%(time.time() * 1000 - time_ori),end = '\r')
        gimbal.angle_control(0,0,0,0,0,0)
        continue
    else:
        print("%5d  %5d   %5.2f ms"%(x, y, time.time() * 1000 - time_ori),end = '\r')   
        u = v = 0
        if abs(x / 1200 * 30.3) > 0:
            u = x * abs(x) / 1.5 / 600
        if abs(y / 1200 * 30.3) > 0:
            v = y * abs(y) / 1.5 / 960
            gimbal.angle_control(0,0,0,u,0,v)
        continue
    gimbal.angle_control(0,0,0,0,0,0)

       
