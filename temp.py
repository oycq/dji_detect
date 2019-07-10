import matplotlib.pyplot as plt
import numpy as np
import cv2
plt.ion() ## Note this correction
fig=plt.figure()
plt.axis([0,1000,0,1])

i=0
x=list()
y=list()
cap = cv2.VideoCapture('../data/mp4/GH010083.MP4')
while i <1000:
    _ , image = cap.read()
    if _ == False:
        continue
    temp_y=np.random.random();
    x.append(i);
    y.append(temp_y);
    plt.scatter(i,temp_y);
    i+=1;
    plt.show()
    plt.pause(0.0001) #Note this correction
    #cv2.imshow('hasakey',image)
    #cv2.waitKey(20)
    break
