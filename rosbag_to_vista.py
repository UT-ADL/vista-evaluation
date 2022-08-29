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

root_path = '/gpfs/space/projects/Bolt/bagfiles/'
big_bag = root_path + '2022-06-29-10-46-40_e2e_elva__forward_steering2.bag'
small_bag = root_path + '2022-06-29-13-39-57_e2e_elva__backward_steering1.bag'
b = bagreader(big_bag) 

# for topic in b.topic_table['Topics']:
#     print( topic )
    
# LASER_MSG = b.message_by_topic('/interfacea/link2/image/compressed')
# pd.set_option('display.max_columns', None)
# df_laser = pd.read_csv(LASER_MSG)
# df_laser['Time']

# df = pd.DataFrame({'ros_time': df_laser['Time']})
# df.index.name = '#frame_num'
# df.to_csv('camera_front.csv')

# times_count = len(df_laser['Time'])
# df0 = pd.DataFrame({'timestamp': df_laser['Time'], 'camera_front': range(times_count) })
# df0.set_index('timestamp')
# df0.to_csv('master_clock.csv', index=False)



### ===================================================
# import math
# LASER_MSG = b.message_by_topic('/current_velocity')
# pd.set_option('display.max_columns', None)
# df_yaw = pd.read_csv(LASER_MSG)

# LASER_MSG = b.message_by_topic('/novatel/oem7/inspva')
# pd.set_option('display.max_columns', None)
# df_yaw = pd.read_csv(LASER_MSG)

# yaw_rates = []
# prev_yaw = 0.0
# prev_time = 0.0
# for yaw in df_yaw:
#     yaw_rad = math.radians(yaw['azimuth'])
#     yaw_rate = ( yaw_rad - prev_yaw ) / ( yaw['Time'] - prev_time )
#     yaw_rates.append( yaw_rate )
#     prev_yaw = yaw_rad
#     prev_time = yaw['Time']

# zeros = [0]*len(df_yaw['Time'])
# df3 = pd.DataFrame( {'time': df_yaw['Time'], 'ax': zeros, 'ay': zeros, 'az': zeros, 'rx': zeros, 'ry': zeros, 'rz': df_yaw['twist.angular.z'], 'qx': zeros, 'qy': zeros, 'qz': zeros, 'qw': zeros})

# df3.set_index('time')
# df3.to_csv('imu.csv', index=False)

### ===================================================

# LASER_MSG = b.message_by_topic('/novatel/oem7/corrimu')
# pd.set_option('display.max_columns', None)
# df_yaw_rate = pd.read_csv(LASER_MSG)

# yaw_rate = df_yaw_rate['yaw_rate'] * 100

# i = 0
# print( yaw_rate[i:i+5], np.mean( yaw_rate[i:i+5] ) )


# smooth_yawrate = []
# for i in range( len(yaw_rate) - 5 ):
#     smooth_yawrate.append( np.mean( yaw_rate[i:i+5] ) )

# last_yaw = smooth_yawrate[-1]
# for _ in range(5):
#     smooth_yawrate.append(last_yaw)

### ===================================================

LASER_MSG = b.message_by_topic('/ssc/curvature_feedback')
pd.set_option('display.max_columns', None)
df_curvature = pd.read_csv(LASER_MSG)


LASER_MSG = b.message_by_topic('/current_velocity')
pd.set_option('display.max_columns', None)
df_speed = pd.read_csv(LASER_MSG)

df2 = pd.DataFrame({'time': df_speed['Time'], 'speed': df_speed['twist.linear.x'] })
df2.set_index('time')
df2.to_csv('speed.csv', index=False)

min_len = min( len(df_curvature), len(df_speed) )

curvs = df_curvature[:min_len]
spdss = df_speed[:min_len]


yaw_rate = curvs['curvature'] * np.maximum(spdss['twist.linear.x'], 1e-10)

# indxs = range( len(smooth_yawrate) )

# plt.plot( indxs, smooth_yawrate )
# indxs = range( len(yaw_rate) )
# plt.plot( indxs, yaw_rate )

# plt.savefig('yaw_rate.png')


zeros = [0]*len(spdss['Time'])
#print( f'zeros={len(zeros)} yaw={len(smooth_yawrate)}' )
df3 = pd.DataFrame( {'time': spdss['Time'], 'ax': zeros, 'ay': zeros, 'az': zeros, 'rx': zeros, 'ry': zeros, 'rz': yaw_rate, 'qx': zeros, 'qy': zeros, 'qz': zeros, 'qw': zeros})

df3.set_index('time')
df3.to_csv('imu.csv', index=False)


# LASER_MSG = b.message_by_topic('/pacmod/parsed_tx/yaw_rate_rpt') #/novatel/oem7/corrimu
# pd.set_option('display.max_columns', None)
# df_laser = pd.read_csv(LASER_MSG)
# zeros = [0]*len(df_laser['Time'])
# df3 = pd.DataFrame( {'time': df_laser['Time'], 'ax': zeros, 'ay': zeros, 'az': zeros, 'rx': zeros, 'ry': zeros, 'rz': df_laser['yaw_rate'], 'qx': zeros, 'qy': zeros, 'qz': zeros, 'qw': zeros})

# df3.set_index('time')
# df3.to_csv('imu.csv', index=False)

# Create a simple helper class that will assist us in storing videos of the render
# class VideoStream():
#     def __init__(self):
#         self.tmp = "./tmp_imgs"
#         if os.path.exists(self.tmp) and os.path.isdir(self.tmp):
#             shutil.rmtree(self.tmp)
#         os.mkdir(self.tmp)
#     def write(self, image, index):
#         cv2.imwrite(os.path.join(self.tmp, f"{index:04}.png"), image)
#     def save(self, fname):
#         cmd = f"/usr/bin/ffmpeg -f image2 -i {self.tmp}/%04d.png -crf 0 -y {fname}"
#         subprocess.call(cmd, shell=True)


#stream = VideoStream()

#i_step = 0

#bag = rosbag.Bag(big_bag, 'r')
#msgs = bag.read_messages(topics=['/interfacea/link2/image/compressed'])

#for m in msgs:
#    msg = m.message

#   fl = open( 'tmp.jpeg', 'wb' )
#    fl.write( bytearray(msg.data) )
#    fl.close()

#    vis_img = cv2.imread( 'tmp.jpeg' )

#    stream.write(vis_img, index=i_step)
#    i_step += 1

#stream.save("from_rosbag.avi")









