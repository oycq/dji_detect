import cv2
import glob
import math
import numpy as np
    
def sort_func(image_name):
    video_name = int.from_bytes(image_name.split('/')[-2].encode(), 'little')
    num = int(image_name.split('/')[-1].split('.')[-2])
    return video_name + num
image_name_list1 = glob.glob('../data/train/1/*/*.jpg')
image_name_list1.sort(key = sort_func)
image_name_list2 = glob.glob('../data/train/2/*/*.jpg')
image_name_list2.sort(key = sort_func)
image_name_list0 = glob.glob('../data/train/0/*/*.jpg')
image_name_list0.sort(key = sort_func)
image_name_list = image_name_list1 + image_name_list2 + image_name_list0

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
image = cv2.imread(image_name_list[0])
image_size = image.shape
r = 125

mouse_x = 0
mouse_y = 0
point_a = (0, 0)
point_b = (0, 0)

def mouse_callback(event, x, y, flag, param):
    global mouse_x, mouse_y, point_a, point_b
    mouse_x = x
    mouse_y = y
    point_a = (max(x - r, 0), max(y - r, 0))
    point_b = (min(x + r, image_size[1]), min(y + r, image_size[0]))

cv2.setMouseCallback('image', mouse_callback)
i = 0 

#data = np.zeros((len(image_name_list), 4), dtype ='int')
#np.save("../data/mark.npy", data)
data = np.load("../data/mark.npy")

while(1):
    i = i % len(image_name_list)
    image_name = image_name_list[i]
    image = cv2.imread(image_name)
    if data[i].sum() != 0:
        p_a = (data[i][0], data[i][1])
        p_b = (data[i][2], data[i][3])
        image = cv2.rectangle(image, p_a, p_b, [237, 128, 255] , 5)
    image = cv2.rectangle(image, point_a, point_b, [128, 128, 0] , 5)
    cv2.imshow('image',image)
    key = cv2.waitKey(2)
    if key == ord('q'):
        break
    if key == ord('s'):
        np.save("../data/mark.npy", data)
        np.save('../data/image_name_list.npy',np.array(image_name_list))
        continue
    if key == ord('2'):
        data[i][0] = point_a[0]
        data[i][1] = point_a[1]
        data[i][2] = point_b[0]
        data[i][3] = point_b[1]
        i += 1
        continue
    if key == ord('1'):
        i = i - 1
        continue
    if key == ord('3'):
        i = i + 1
        continue
    if key == ord('7'):
        i = i - 100
        continue
    if key == ord('9'):
        i = i + 100
    if key == ord('3'):
        i = i + 10
        continue
    if key == ord('4'):
        i = i - 10
    if key == ord('6'):
        i = i + 10
        continue

