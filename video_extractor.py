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

from video_stream_rec import VideoStream
import mitdeeplearning as mdl

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