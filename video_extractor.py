import shutil, os, subprocess
import time
import socket

import rosbag
import cv2
import numpy as np

import uuid

HOSTNAME = socket.gethostname()
FFMPEG_BIN = '/usr/local/bin/ffmpeg' if HOSTNAME == 'neuron' else 'ffmpeg'
CAMERA_TOPIC = '/interfacea/link2/image/compressed'

class VideoStream:
    '''Class to write images to a video stream.
    '''
    def __init__(self, fps=30):
        self.tmp = os.path.join('.', str(uuid.uuid4().hex)[:8])
        self.fps = fps
        if os.path.exists(self.tmp) and os.path.isdir(self.tmp):
            shutil.rmtree(self.tmp)
        os.mkdir(self.tmp)
    def write(self, image, index):
        cv2.imwrite(os.path.join(self.tmp, f"{index:04}.png"), image)
    def save(self, fname):
        cmd = f"{FFMPEG_BIN} -f image2 -framerate {self.fps} -i {self.tmp}/%04d.png -codec copy -y {fname}"
        subprocess.call(cmd, shell=True)
        shutil.rmtree(self.tmp)

def compressed_imgmsg_to_cv2(cmprs_img_msg):
    str_msg = cmprs_img_msg.data
    buf = np.ndarray(shape=(1, len(str_msg)),
                        dtype=np.uint8, buffer=str_msg)
    im = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    return im

def compressed_imgmsg_to_cv2_old(cmprs_img_msg):
    fl = open( 'tmp.jpeg', 'wb' )
    fl.write( bytearray(cmprs_img_msg.data) )
    fl.close()
    vis_img = cv2.imread( 'tmp.jpeg' )
    return vis_img

def bag_to_video(bag_path, output_dir):
    stream = VideoStream()
    output_path = os.path.join(output_dir, 'camera_front.avi')

    print('Reading bag file: {}'.format(bag_path))
    start_time = time.perf_counter()
    bag = rosbag.Bag(bag_path, 'r')
    msgs = bag.read_messages(topics=[CAMERA_TOPIC])

    print('Saving video to: {}'.format(output_path))
    i_step = 0
    for m in msgs:
        vis_img = compressed_imgmsg_to_cv2(m.message)
        stream.write(vis_img, index=i_step)
        i_step += 1
    end_time = time.perf_counter()

    stream.save(output_path)
    print(f"Time to extract & save: {end_time - start_time:.2f}s")