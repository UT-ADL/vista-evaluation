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
NVIDIA_ACCEL = True if HOSTNAME == 'neuron' else False


class VideoStream:
    '''Class to write images to a raw video stream.
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


class VideoStreamCompressed:
    '''Class to write images to a video stream. Uses GPU if running on neuron (ffmpeg is not built with CUDA on HPC).

    Some useful links:
    - choosing best options for h264_nvenc: 
        https://superuser.com/questions/1296374/best-settings-for-ffmpeg-with-nvenc
    - h264_nvenc encoding options (note the lossless compression "-preset 10"): 
        https://gist.github.com/nico-lab/e1ba48c33bf2c7e1d9ffdd9c1b8d0493
    '''
    def __init__(self, root_dir='.', fps=30, suffix='', no_encoding=False):
        self.no_encoding = no_encoding
        self.tmp = os.path.join(root_dir, 'tmp' + suffix)
        self.fps = fps
        os.makedirs(self.tmp)
        self.index = 0

    def write(self, image):
        cv2.imwrite(os.path.join(self.tmp, f"{self.index:04}.png"), image)
        self.index += 1

    def save(self, fname):
        encoding = '-c:v mpeg4 -q:v 10'
        if NVIDIA_ACCEL:
            # -preset 10 is lossless compression
            # -cq 0 is best quality given compression
            encoding = '-c:v h264_nvenc -preset hq -profile:v high -rc-lookahead 8 -bf 2 -rc vbr -cq 15 -b:v 0 -maxrate 120M -bufsize 240M'

        if self.no_encoding:
            encoding = '-codec copy'
        cmd = f"{FFMPEG_BIN} -f image2 -framerate {self.fps} -i {self.tmp}/%04d.png {encoding} -y {fname}"
        subprocess.call(cmd, shell=True)
        shutil.rmtree(self.tmp)