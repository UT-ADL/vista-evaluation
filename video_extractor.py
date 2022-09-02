import shutil, os, subprocess
import time

import rosbag
import cv2
import numpy as np


root_path = '/data/Bolt/bagfiles/'
big_bag = root_path + '2021-10-20-11-50-26_e2e_rec_vahi_06_spring2.bag' # '2021-11-17-12-29-22_e2e_rec_elva_camera.bag'


# Create a simple helper class that will assist us in storing videos of the render
class VideoStream():
    def __init__(self):
        self.tmp = "./tmp"
        if os.path.exists(self.tmp) and os.path.isdir(self.tmp):
            shutil.rmtree(self.tmp)
        os.mkdir(self.tmp)
    def write(self, image, index):
        cv2.imwrite(os.path.join(self.tmp, f"{index:04}.png"), image)
    def save(self, fname):
        cmd = f"ffmpeg -f image2 -i {self.tmp}/%04d.png -crf 0 -y {fname}"
        subprocess.call(cmd, shell=True)

def compressed_imgmsg_to_cv2(cmprs_img_msg):
    str_msg = cmprs_img_msg.data
    buf = np.ndarray(shape=(1, len(str_msg)),
                        dtype=np.uint8, buffer=cmprs_img_msg.data)
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
    i_step = 0

    output_path = os.path.join(output_dir, 'camera_front.avi')

    print('Starting reading bag file: {}'.format(bag_path))
    bag = rosbag.Bag(bag_path, 'r')
    msgs = bag.read_messages(topics=['/interfacea/link2/image/compressed'])

    print('Starting to save video to: {}'.format(output_path))
    start_time = time.perf_counter()
    for m in msgs[:100]:
        vis_img = compressed_imgmsg_to_cv2(m.message)
        stream.write(vis_img, index=i_step)
        i_step += 1
    end_time = time.perf_counter()

    print(f"Time to extract: {end_time - start_time}")

    stream.save(output_path)