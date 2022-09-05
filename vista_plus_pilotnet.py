import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
import cv2
import base64, io, os, time, gym
import IPython, functools
import time
import tensorflow_probability as tfp
import torch
import mitdeeplearning as mdl
from pilotnet import PilotNet
import tensorflow as tf
import math

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

from memory import Memory

import vista
from vista.utils import logging
logging.setLevel(logging.ERROR)

# define traces 
trace_root = "../traces/"
trace_path = [
    "data"
]

trace_path = [os.path.join(trace_root, p) for p in trace_path]

# Create a virtual world with VISTA, the world is defined by a series of data traces
world = vista.World(trace_path, trace_config={'road_width': 4})

# Create a car in our virtual world. The car will be able to step and take different 
#   control actions. As the car moves, its sensors will simulate any changes it environment
car = world.spawn_agent(
    config={
        'length': 5.,
        'width': 2.,
        'wheel_base': 2.78,
        'steering_ratio': 14.7,
        'lookahead_road': True
    })

# Create a camera on the car for synthesizing the sensor data that we can use to train with! 
camera = car.spawn_camera(config={'size': (640, 920)}) #(200, 320)

# Define a rendering display so we can visualize the simulated car camera stream and also 
#   get see its physical location with respect to the road in its environment. 
display = vista.Display(world, display_config={"gui_scale": 2, "vis_full_frame": False})

def vista_reset():
    world.reset()
    display.reset()

# First we define a step function, to allow our virtual agent to step 
# with a given control command through the environment 
# agent can act with a desired curvature (turning radius, like steering angle)
# and desired speed. if either is not provided then this step function will 
# use whatever the human executed at that time in the real data.
def vista_step(curvature=None, speed=None):
    # Arguments:
    #   curvature: curvature to step with
    #   speed: speed to step with
    if curvature is None: 
        curvature = car.trace.f_curvature(car.timestamp)
    if speed is None: 
        speed = car.trace.f_speed(car.timestamp)    

    car.step_dynamics(action=np.array([curvature, speed]), dt=1/25.)
    car.step_sensors()

def get_curvature():
    return car.trace.f_curvature(car.timestamp)


## Define terminal states and crashing conditions ##

def check_out_of_lane(car):
    distance_from_center = np.abs(car.relative_state.x)
    road_width = car.trace.road_width 
    half_road_width = road_width / 2
    return distance_from_center > half_road_width

def check_exceed_max_rot(car):
    maximal_rotation = np.pi / 10.
    current_rotation = np.abs(car.relative_state.yaw)
    return current_rotation > maximal_rotation

def check_crash(car): 
    return check_out_of_lane(car) or check_exceed_max_rot(car) or car.done

## Nvidia preprocessing stuff
xmin = 300
xmax = 1620
ymin = 520
ymax = 864

def resize(img):
    scale = 0.2
    height = ymax - ymin
    width = xmax - xmin

    scaled_width = int(width * scale)
    scaled_height = int(height * scale)

    return cv2.resize(img, dsize=(scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)

def normalise(img):
    return (img / 255.0)

def crop(img):
    return img[ymin:ymax, xmin:xmax, :]


def preprocess(full_obs):

    img = crop( full_obs )
    img = resize( img )
    img = normalise( img )

    return img

def grab_and_preprocess_obs(car):
    full_obs = car.observations[camera.name]
    obs = preprocess(full_obs)
    return obs


# Model class must be defined somewhere
nvidia_model = PilotNet()
nvidia_model.load_state_dict(torch.load('nvidia-v3.pt'))
print( nvidia_model.eval() )

from video_stream_rec import VideoStream
        
stream = VideoStream()
i_step = 0
wheel_base = 2.79
steering_ratio = 14.8

vista_reset()
observation = grab_and_preprocess_obs(car)

def steering_angle_2_curvature( steering_angle ):
    frontal_wheel_angle = steering_angle / steering_ratio    
    curvature_action = math.tan( frontal_wheel_angle ) / wheel_base
    return curvature_action

def nvidia_inference( observation, model ):

    new_obs = np.moveaxis(observation, -1, 0)    
    inputs = np.array( [new_obs] )
    
    inpt_tens = torch.Tensor( inputs )
    
    predictions = model(inpt_tens)
    steering_angle = predictions.detach().numpy()[0,0]
    
    return steering_angle_2_curvature( steering_angle ), steering_angle

# Iterate until first crash
while not check_crash(car):
    
    curvature_action, steering_angle = nvidia_inference( observation, nvidia_model )

    print( f'steering_angle={steering_angle} true_curvature={get_curvature()} nvidia_curvature={curvature_action}' )
    vista_step( curvature_action )

    observation = grab_and_preprocess_obs(car)
    vis_img = display.render()

    stream.write(vis_img[:, :, ::-1], index=i_step)

    i_step += 1

print("Saving trajectory with trained policy...")
stream.save("trained_policy.avi")