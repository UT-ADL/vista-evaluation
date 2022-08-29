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

root_path = '/data/Bolt/bagfiles/'
big_bag = root_path + '2021-10-20-11-50-26_e2e_rec_vahi_06_spring2.bag' # '2021-11-17-12-29-22_e2e_rec_elva_camera.bag'

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

bag = rosbag.Bag(big_bag, 'r')
msgs = bag.read_messages(topics=['/interfacea/link2/image/compressed'])

for m in msgs:
    msg = m.message

    fl = open( 'tmp.jpeg', 'wb' )
    fl.write( bytearray(msg.data) )
    fl.close()

    vis_img = cv2.imread( 'tmp.jpeg' )

    stream.write(vis_img, index=i_step)
    i_step += 1

stream.save("from_rosbag_vahi_link3.avi")