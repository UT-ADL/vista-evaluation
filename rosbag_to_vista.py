import os
import argparse
import socket
import time

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import rosbag
from collections import defaultdict
from tqdm import tqdm

from video_extractor import VideoStream, compressed_imgmsg_to_cv2

BOLT_DIR = '/data/Bolt' if socket.gethostname() == 'neuron' else '/gpfs/space/projects/Bolt'
BAGS_DIR = os.path.join(BOLT_DIR, 'bagfiles')

CAMERA_TOPIC_30HZ = '/interfacea/link2/image/compressed'
SPEED_TOPIC_30HZ = '/ssc/velocity_accel_cov'
CURVATURE_TOPIC_30HZ = '/ssc/curvature_feedback'



def read_bag(path_to_bag, topics, max_duration=0):
    bag = rosbag.Bag(path_to_bag, 'r')
    msg_count = bag.get_message_count(topics)
    progress = tqdm(total=msg_count)

    video_stream = VideoStream()
    frame_idx = 0

    camera_dict = defaultdict(list)
    speed_dict = defaultdict(list)
    curvature_dict = defaultdict(list)

    bag_start_time = None

    for topic, msg, ts in bag.read_messages(topics):
        msg_timestamp = msg.header.stamp.to_sec()

        if bag_start_time is None:
            bag_start_time = msg_timestamp

        seconds_since_start = (msg_timestamp - bag_start_time)
        if max_duration > 0 and seconds_since_start > max_duration:
            break

        if topic == CAMERA_TOPIC_30HZ:
            camera_dict['time'].append(msg_timestamp)
            vis_img = compressed_imgmsg_to_cv2(msg)
            video_stream.write(vis_img, index=frame_idx)
            frame_idx += 1
        elif topic == SPEED_TOPIC_30HZ:
            speed_dict['time'].append(msg_timestamp)
            speed_dict['speed'].append(msg.velocity)
        elif topic == CURVATURE_TOPIC_30HZ:
            curvature_dict['time'].append(msg_timestamp)
            curvature_dict['curvature'].append(msg.curvature)

        if max_duration > 0:
            progress.set_description_str(f'Processed {seconds_since_start:.2f}s/{max_duration:.2f}s')
        progress.update(1)
        
    bag.close()

    return video_stream, camera_dict, speed_dict, curvature_dict

def create_timestamp_index(df, timestamp_col='time', index_col='index'):
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df.set_index([timestamp_col], inplace=True)
    df.index.rename(index_col, inplace=True)

def save_video(stream, output_dir):
    output_path = os.path.join(output_dir, 'camera_front.avi')
    stream.save(output_path)

def save_video_timestamps(camera_dict, output_dir):
    camera_df = pd.DataFrame(data=camera_dict)
    df = pd.DataFrame({'ros_time': camera_df['time']})
    df.index.name = '#frame_num'
    df.to_csv(os.path.join(output_dir, 'camera_front.csv'))

def save_speed(speed_dict, output_dir):
    speed_df = pd.DataFrame(data=speed_dict, columns=['time', 'speed'])
    speed_df.to_csv(os.path.join(output_dir, 'speed.csv'), index=False)
    return speed_df

def save_yaw_rate(curvature_dict, speed_df, output_dir):
    curvature_df = pd.DataFrame(data=curvature_dict, columns=['time', 'curvature'])

    # assume that the frequency is similar for both topics
    min_len = min( len(curvature_df), len(speed_df) )
    curvs = curvature_df[:min_len]
    spdss = speed_df[:min_len]
    f_speed = interp1d(spdss['time'].values.astype(np.float64), spdss['speed'], fill_value='extrapolate')
    timestamps = curvs['time']
    yaw_rate = curvs['curvature'] * np.maximum(f_speed(timestamps), 1e-10)
    zeros = [0]*len(spdss)
    imu_df = pd.DataFrame( {'time': curvs['time'], 'ax': zeros, 'ay': zeros, 'az': zeros, 
                                                   'rx': zeros, 'ry': zeros, 'rz': yaw_rate, # <----
                                                   'qx': zeros, 'qy': zeros, 'qz': zeros, 'qw': zeros})
    imu_df.set_index('time')
    imu_df.to_csv(os.path.join(output_dir, 'imu.csv'), index=False)

def create_params_file(output_dir, template_file='./params.xml'):
    with open(template_file, 'r') as f:
        params = f.read()
    with open(os.path.join(output_dir, 'params.xml'), 'w') as f:
        f.write(params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a rosbag to a vista file')
    parser.add_argument('--root', default=BAGS_DIR, help='Path to the bags directory')
    parser.add_argument('--bag', required=True, help='Name of the bag file') # e.g. '2022-06-29-10-46-40_e2e_elva__forward_steering2.bag'
    parser.add_argument('--output-base', default=os.path.join(BOLT_DIR, 'end-to-end/vista'), help='Path to the output base directory.')
    parser.add_argument('--max-duration', type=float, default=0, help='Maximum duration of the bag file to process (in seconds).')
    args = parser.parse_args()

    bag_name = os.path.basename(args.bag).split('.')[0]
    output_dir = os.path.join(args.output_base, bag_name)
    os.makedirs(output_dir, exist_ok=True)
    path_to_bag = os.path.join(args.root, args.bag) 

    topics = [CAMERA_TOPIC_30HZ, SPEED_TOPIC_30HZ, CURVATURE_TOPIC_30HZ]
    
    bag_read_start = time.perf_counter()
    video_stream, camera_dict, speed_dict, curvature_dict = read_bag(path_to_bag, topics, max_duration=args.max_duration)
    print('Done reading!')
    bag_read_end = time.perf_counter()

    save_start = time.perf_counter()
    save_video(video_stream, output_dir)
    save_video_timestamps(camera_dict, output_dir)
    speed_df = save_speed(speed_dict, output_dir)
    save_yaw_rate(curvature_dict, speed_df, output_dir)
    create_params_file(output_dir)
    save_end = time.perf_counter()
    print('Done saving!')

    bag_read_time = bag_read_end - bag_read_start
    save_time = save_end - save_start
    total_time = bag_read_time + save_time

    print(f'\nTime spent: {total_time:.2f} seconds (reading: {bag_read_time:.2f}, saving: {save_time:.2f})')
