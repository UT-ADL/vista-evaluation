import os
import argparse
import time
import csv

import cv2
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
import rosbag

from src.video import VideoStream
from src.preprocessing import resize_before_crop

CAMERA_TOPIC_30HZ = '/interfacea/link2/image/compressed'
SPEED_TOPIC_30HZ = '/ssc/velocity_accel_cov'
CURVATURE_TOPIC_30HZ = '/ssc/curvature_feedback'
TURN_SIGNAL_TOPIC_30HZ = '/pacmod/parsed_tx/turn_rpt'


def process_bag(output_dir, path_to_bag, topics, resize_mode, max_duration=0):
    bag = rosbag.Bag(path_to_bag, 'r')
    msg_count = bag.get_message_count(topics)
    progress = tqdm(total=msg_count)

    video_stream = VideoStream(os.path.join(output_dir, 'camera_front.avi'), fps=30, lossless=True)

    camera_csv = open(os.path.join(output_dir, 'camera_front.csv'), 'w')
    speed_csv = open(os.path.join(output_dir, 'speed.csv'), 'w')
    curvature_csv = open(os.path.join(output_dir, 'curvature.csv'), 'w')
    turn_signal_csv = open(os.path.join(output_dir, 'turn_signal.csv'), 'w')

    camera_csv_writer = csv.DictWriter(camera_csv, fieldnames=['#frame_num', 'ros_time'])
    speed_csv_writer = csv.DictWriter(speed_csv, fieldnames=['time', 'speed'])
    curvature_csv_writer = csv.DictWriter(curvature_csv, fieldnames=['time', 'curvature'])
    turn_signal_csv_writer = csv.DictWriter(turn_signal_csv, fieldnames=['time', 'turn_signal'])

    camera_csv_writer.writeheader()
    speed_csv_writer.writeheader()
    curvature_csv_writer.writeheader()

    bag_start_time = None
    frame_idx = 0

    for topic, msg, ts in bag.read_messages(topics):
        msg_timestamp = msg.header.stamp.to_sec()

        if bag_start_time is None:
            bag_start_time = msg_timestamp

        seconds_since_start = (msg_timestamp - bag_start_time)
        if max_duration > 0 and seconds_since_start > max_duration:
            break

        if topic == CAMERA_TOPIC_30HZ:
            camera_csv_writer.writerow({'#frame_num': frame_idx, 'ros_time': msg_timestamp})
            vis_img = compressed_imgmsg_to_cv2(msg)

            if resize_mode == 'full_res':
                # original, in-distribution but slow
                pass
            elif resize_mode == 'resize': 
                # needs intrinsics scaling and cropping during inference
                vis_img = resize_before_crop(vis_img)
            else: 
                raise NotImplementedError(f'No such resize mode: "{resize_mode}"')

            video_stream.write(vis_img)
            frame_idx += 1
        elif topic == SPEED_TOPIC_30HZ:
            speed_csv_writer.writerow({'time': msg_timestamp, 'speed': msg.velocity})
        elif topic == CURVATURE_TOPIC_30HZ:
            curvature_csv_writer.writerow({'time': msg_timestamp, 'curvature': msg.curvature})
        elif topic == TURN_SIGNAL_TOPIC_30HZ:
            turn_signal_csv_writer.writerow({'time': msg_timestamp, 'turn_signal': int(msg.output)})

        if max_duration > 0:
            progress.set_description_str(f'Processed {seconds_since_start:.2f}s/{max_duration:.2f}s')
        progress.update(1)
        
    video_stream.save()
    bag.close()
    camera_csv.close()
    speed_csv.close()
    curvature_csv.close()

    save_yaw_rate(output_dir)
    os.remove(os.path.join(output_dir, 'curvature.csv'))
    create_params_file(output_dir, template_file=f'./calibration/params-{resize_mode}.xml')

def compressed_imgmsg_to_cv2(cmprs_img_msg):
    str_msg = cmprs_img_msg.data
    buf = np.ndarray(shape=(1, len(str_msg)),
                        dtype=np.uint8, buffer=str_msg)
    im = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    return im

def save_yaw_rate(output_dir):
    with open(os.path.join(output_dir, 'speed.csv'), 'r') as f:
        speed_reader = csv.DictReader(f)
        speeds = np.array([[float(row['time']), float(row['speed'])] for row in speed_reader]).reshape(-1, 2)
        speed_times = speeds[:, 0]
        speed_values = speeds[:, 1]
    with open(os.path.join(output_dir, 'curvature.csv'), 'r') as f:
        curvatures = csv.DictReader(f)
        curvatures = np.array([[float(row['time']), float(row['curvature'])] for row in curvatures]).reshape(-1, 2)
        curvature_times = curvatures[:, 0]
        curvature_values = curvatures[:, 1]
    
    # NOTE: here we assume that the frequency is similar for both topics
    min_len = min( len(curvature_times), len(speed_times) )
    speed_times = speed_times[:min_len]
    speed_values = speed_values[:min_len]
    curvature_times = curvature_times[:min_len]
    curvature_values = curvature_values[:min_len]
    f_speed = interp1d(speed_times.astype(np.float64), speed_values, fill_value='extrapolate')

    timestamps = curvature_times
    yaw_rate = curvature_values * np.maximum(f_speed(timestamps), 1e-10)
    imu_csv_writer = csv.DictWriter(open(os.path.join(output_dir, 'imu.csv'), 'w'), fieldnames=['time', 'ax', 'ay', 'az', 
                                                                                                        'rx', 'ry', 'rz',
                                                                                                        'qx', 'qy', 'qz', 'qw'])

    imu_csv_writer.writeheader()
    for i in range(len(timestamps)):
        imu_csv_writer.writerow({'time': timestamps[i], 'ax': 0, 'ay': 0, 'az': 0, 
                                                        'rx': 0, 'ry': 0, 'rz': yaw_rate[i],
                                                        'qx': 0, 'qy': 0, 'qz': 0, 'qw': 0})

def create_params_file(output_dir, template_file='./calibration/params.xml'):
    with open(template_file, 'r') as f:
        params = f.read()
    with open(os.path.join(output_dir, 'params.xml'), 'w') as f:
        f.write(params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a rosbag to a vista file')
    parser.add_argument('--bag', required=True, help='Path to the bag file')
    parser.add_argument('--force', default=False, help='Override existing vista trace for given bag if the trace already exists.')
    parser.add_argument('--max-duration', type=float, default=0, help='Maximum duration of the bag file to process (in seconds).')
    parser.add_argument('--output-root', required=True, help='Path to the directory where traces live.')
    parser.add_argument('--resize-mode', required=False, default='resize', choices=['full_res', 'resize'], help='How to resize the images.')
    args = parser.parse_args()

    bag_name = os.path.basename(args.bag).split('.')[0]
    output_dir = os.path.join(args.output_root, bag_name + '-' + args.resize_mode)
    os.makedirs(output_dir, exist_ok=args.force)

    topics = [CAMERA_TOPIC_30HZ, SPEED_TOPIC_30HZ, CURVATURE_TOPIC_30HZ, TURN_SIGNAL_TOPIC_30HZ]
    
    start_time = time.perf_counter()
    process_bag(output_dir, args.bag, topics, args.resize_mode, max_duration=args.max_duration)
    end_time = time.perf_counter()
    print('Done creating a trace!')

    total_time = end_time - start_time

    print(f'\nTime spent: {total_time:.2f} seconds')
