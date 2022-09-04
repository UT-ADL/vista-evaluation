import bagpy
from bagpy import bagreader
import pandas
import seaborn as sea
import matplotlib.pyplot as plt
import numpy as np
from cv_bridge import CvBridge
import rosbag
import cv2
import rospy 

import json

config = open('rtv_config.json')

data = json.load(config)

import shutil, os, subprocess, cv2
import mitdeeplearning as mdl

# Create a simple helper class that will assist us in storing videos of the render
class VideoStream():
    def __init__(self):
        self.tmp = "./tmp_imgs"
        if os.path.exists(self.tmp) and os.path.isdir(self.tmp):
            shutil.rmtree(self.tmp)
        os.mkdir(self.tmp)
    def write(self, image, index):
        cv2.imwrite(os.path.join(self.tmp, f"{index:04}.png"), image)
    def save(self, fname):
        cmd = f"/usr/bin/ffmpeg -f image2 -i {self.tmp}/%04d.png -crf 0 -y {fname}"
        subprocess.call(cmd, shell=True)


stream = VideoStream()

i_step = 0

bag = rosbag.Bag(data['rosbag'], 'r')
msgs = bag.read_messages(topics=[data['camera']])

for m in msgs:
    msg = m.message

    vis_img = cv2.imdecode( np.asarray(bytearray(msg.data)), cv2.IMREAD_ANYCOLOR )

    stream.write(vis_img, index=i_step)
    i_step += 1

stream.save(data['rosbag'] + ".avi")