import cv2
import time
import progressbar
import os

missions =[
#        ['../data/mp4/GH010083.MP4'         ,'../data/train/1'       ],
#        ['../data/mp4/GH010085.MP4'         ,'../data/test/1'       ],
#        ['../data/mp4/GH010086.MP4'         ,'../data/train/0'       ],
#        ['../data/mp4/GH010087.MP4'         ,'../data/test/0'       ],
#        ['../data/mp4/GH010092.MP4'         ,'../data/train/1'       ],
#        ['../data/mp4/GH010093.MP4'         ,'../data/train/0'       ],
#        ['../data/mp4/GH010097.MP4'         ,'../data/train/2'       ],
#        ['../data/mp4/GH010113.MP4'         ,'../data/train/2'       ],
#        ['../data/mp4/GH010114.MP4'         ,'../data/train/2'       ],
#        ['../data/mp4/GH010115.MP4'         ,'../data/train/2'       ],
#        ['../data/mp4/GH010116.MP4'         ,'../data/train/2'       ],
#        ['../data/mp4/GH010123.MP4'         ,'../data/train/2'       ],
#        ['../data/mp4/GH010124.MP4'         ,'../data/train/0'       ],
#        ['../data/mp4/GH010125.MP4'         ,'../data/train/1'       ],
        ['../data/mp4/GH010127.MP4'         ,'../data/train/1'       ],
        ['../data/mp4/GH010128.MP4'         ,'../data/train/0'       ],
        ['../data/mp4/GH010129.MP4'         ,'../data/train/2'       ],
          ]

for mission in missions:
    quality = 90
    video_name = mission[0]
    output_path = mission[1]+'/'+mission[0].split('/')[-1].split('.')[0]
    try:
        os.makedirs(output_path)
    except FileExistsError:
        print(output_path+' exists')
    cap = cv2.VideoCapture(video_name)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(video_name,frame_count,output_path)
    my_progressbar = progressbar.ProgressBar(maxval=frame_count, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    my_progressbar.start()
    for i in range(frame_count):
        status, image = cap.read()
        if status == 0:
            continue
        image = image[80:1360]
        cv2.imshow('image',image)
        cv2.imwrite(output_path+'/%d.jpg'%i, image, (cv2.IMWRITE_JPEG_QUALITY, quality))
        #print(output_path+'/%d.jpg'%i)
        return_key = cv2.waitKey(1)
        if return_key == ord('q'):
            break
        my_progressbar.update(i) 
    my_progressbar.finish()
    cap.release()
cv2.destroyWindow('image')
