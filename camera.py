import os
import PySpin
import numpy as np
import cv2
import serial#sudo pip3 install pyserial
import time

#2-3frame latency, 3-4ms latency
setting_list = [
    ['TriggerMode'           , 'Off'      , 'General'],
    ['TriggerSource'           , 'Software'      , 'General'],
    ['TriggerMode'           , 'On'      , 'General'],
    ['StreamBufferHandlingMode'  , 'NewestOnly'      , 'Stream' ],
    ['AcquisitionMode'           , 'Continuous'      , 'General'],
                ]

class Camera():
    def config_camera(self):
        cam = self.cam
        setting_list = self.setting_list
        for i in range(len(setting_list)):
            if setting_list[i][2] == 'General':
                nodemap = cam.GetNodeMap()
            if setting_list[i][2] == 'Stream':
                nodemap = cam.GetTLStreamNodeMap()
            attribute = setting_list[i][0]
            value =  setting_list[i][1]
            node =  PySpin.CEnumerationPtr(nodemap.GetNode(attribute))
            if not PySpin.IsAvailable(node) or not PySpin.IsWritable(node):
                print('Unable to set attribute : %s'%attribute)
                continue
            node_new_value = node.GetEntryByName(value)
            if not PySpin.IsAvailable(node_new_value) or not PySpin.IsReadable(node_new_value):
                print('Cant set %s --> %s'%(value,attribute))
                continue
            node.SetIntValue(node_new_value.GetValue())


    def __init__(self):
        self.setting_list = setting_list
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        num_cameras = self.cam_list.GetSize()
        if num_cameras == 0:
            self.cam_list.Clear()
            self.system.ReleaseInstance()
            print('Not enough cameras!')
            os._exit(0)
        self.cam = self.cam_list[0]
        self.cam.Init()
        self.config_camera()
        nodemap = self.cam.GetNodeMap()
        self.trigger_node = PySpin.CCommandPtr(nodemap.GetNode('TriggerSoftware'))
        self.cam.BeginAcquisition()

    def __del__(self):
        self.cam.EndAcquisition()
        self.cam.DeInit()
        del self.cam
        self.cam_list.Clear()
        self.system.ReleaseInstance()

    def read(self):
        self.trigger_node.Execute()
        image_pt = self.cam.GetNextImage()
        if image_pt.IsIncomplete():
            print('Image incomplete with image status %d ...' % image_pt.GetImageStatus())
            os._exit(0) 
        image= image_pt.GetNDArray()
        image_pt.Release()
        return image

if __name__ == '__main__':
    cam = Camera()
    while(1):
        image = cam.read()
        cv2.imshow('a',image)
        key = cv2.waitKey(20)
        if key == ord('q'):
            del cam
            break
