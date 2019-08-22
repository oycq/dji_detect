import gimbal
import cv2
import camera

cam = camera.Camera()
u = 0 
v = 0 
while(1):
    raw = cam.read()
#    cv2.imshow('raw',raw)
    high_light = raw.max()
    image = raw.copy()
    image.fill(0)
    image[raw < 40] = 255 
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
        del cam
        break
    if key == ord('8'):
        u -= 0.5
    if key == ord('2'):
        u += 0.5
    if key == ord('4'):
        v -= 0.5
    if key == ord('6'):
        v += 0.5
    if high_light < 100 or abs(x) > 590 or abs(y) > 950:
        print("            u : %5.2f  v : %5.2f"%(u,v),end = '\r')
        gimbal.angle_control(u,0,v)
        continue
    else:
        print("%5d %5d u : %5.2f  v : %5.2f"%(x,y,u,v),end = '\r')
    if key == ord('5'):
        d_u = x / 1200 * 30.3
        if abs(d_u) > 1:
            if abs(d_u) > 2:
                d_u = 3 * d_u / abs(d_u)
            u = u + d_u 
        d_v = y / 1200 * 30.3
        if abs(d_v) > 1:
            if abs(d_v) > 2:
                d_v = 3* d_v / abs(d_v)
            v = v + d_v 
    gimbal.angle_control(u,0,v)
