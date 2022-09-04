import bagpy
from bagpy import bagreader
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
import numpy as np

import bagpy
from bagpy import bagreader
import pandas
import seaborn as sea
import matplotlib.pyplot as plt
import numpy as np
import rosbag
import cv2
import rospy 

import shutil, os, subprocess, cv2
import mitdeeplearning as mdl

import json

config = open('rtv_config.json')

data = json.load(config)

b = bagreader( data['rosbag'] )

pd.set_option('display.max_columns', None)

def topic_to_dataframe( topic ):  
    ros_message = b.message_by_topic(topic)
    return pd.read_csv(ros_message)

## Processing camera information
df_camera = topic_to_dataframe( data['camera'] )

df = pd.DataFrame({'ros_time': df_camera['Time']})
df.index.name = '#frame_num'
df.to_csv('camera_front.csv')

## Processing speed information
spd_name, spd_col = data['speed'].split('.')
df_speed = topic_to_dataframe( spd_name )
df = pd.DataFrame({'time': df_speed['Time'], 'speed': df_speed[spd_col] })
df.set_index('time')
df.to_csv('speed.csv', index=False)

## Processing curvature information
curv_name, curv_col = data['curvature'].split('.')
df_curvature = topic_to_dataframe( curv_name )

yaw_rate = df_curvature[curv_col] * np.maximum(df_speed[spd_col], 1e-10)

zeros = [0]*len(df_speed['Time'])
df = pd.DataFrame( {'time': df_speed['Time'], 'ax': zeros, 'ay': zeros, 'az': zeros, 'rx': zeros, 'ry': zeros, 'rz': yaw_rate, 'qx': zeros, 'qy': zeros, 'qz': zeros, 'qw': zeros})
df.set_index('time')
df.to_csv('imu.csv', index=False)