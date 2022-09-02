import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import vista
from vista.utils import logging
logging.setLevel(logging.ERROR)

trace_root = "./vista_traces"
trace_path = [
    #"e2e_rec_vahi/",
    #"e2e_rec_link3/"
    #"custom_trace/",
    #"custom_trace2/",
    #"custom_trace_reverse/",
    "20210726-154641_lexus_devens_center", 
    #"20210726-155941_lexus_devens_center_reverse", 
    #"20210726-184624_lexus_devens_center", 
    #"20210726-184956_lexus_devens_center_reverse", 
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
camera = car.spawn_camera(config={'size': (1208, 1928)}) #(200, 320)

#print(f'CAMERA={camera.name}')
#print(f'CAMERA={camera.camera_param.get_roi()}')

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

def get_curvature():
    return car.trace.f_curvature(car.timestamp)

def vista_step(curvature=None, speed=None):
    # Arguments:
    #   curvature: curvature to step with
    #   speed: speed to step with
    if curvature is None: 
        curvature = car.trace.f_curvature(car.timestamp)
    if speed is None: 
        speed = car.trace.f_speed(car.timestamp)
    
    print( f'curvature={curvature} speed={speed}' )

    car.step_dynamics(action=np.array([curvature, speed]), dt=1/25.)
    car.step_sensors()


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


## Data preprocessing functions ##

xmin = 300
xmax = 1620
ymin = 520
ymax = 864

def resize(img):
    scale = 0.2
    height = ymax - ymin
    width = xmax - xmin

    #width = 264
    #height = 64
    scaled_width = int(width * scale)
    scaled_height = int(height * scale)

    return cv2.resize(img, dsize=(scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)

def normalise(img):
    return (img / 255.0)

def crop(img):
    return img[ymin:ymax, xmin:xmax, :]

def preprocess(full_obs):
    print('full obs shape:', full_obs.shape)
    img = crop( full_obs )
    img = resize( img )
    img = normalise( img )
    return img

def grab_and_preprocess_obs(car):
    full_obs = car.observations[camera.name]
    obs = preprocess(full_obs)
    return obs

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class PilotNet(nn.Module):
    """
    Network from 'End to End Learning for Self-Driving Cars' paper:
    https://arxiv.org/abs/1604.07316
    """

    def __init__(self, n_input_channels=3, n_outputs=1):
        super(PilotNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(n_input_channels, 24, 5, stride=2),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.BatchNorm2d(36),
            nn.LeakyReLU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(),
            nn.Conv2d(48, 64, 3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Flatten()
        )

        self.regressor = nn.Sequential(
            nn.Linear(1664, 100),
            nn.BatchNorm1d(100),
            nn.LeakyReLU(),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.LeakyReLU(),
            nn.Linear(50, 10),
            nn.LeakyReLU(),
            nn.Linear(10, n_outputs),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

# Model class must be defined somewhere
nvidia_model = PilotNet()
nvidia_model.load_state_dict(torch.load('mae-v2.pt', map_location=torch.device('cpu')))
nvidia_model.eval()

import shutil, os, subprocess, cv2

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
        

stream = VideoStream()
i_step = 0

import matplotlib.pyplot as plt
import math
fig = plt.figure(figsize=(3, 6))

max_reward = float('-inf') # keep track of the maximum reward acheived during training
if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists

for i_episode in range(1):

    # Restart the environment
    vista_reset()
    observation = grab_and_preprocess_obs(car)
    
    wheel_base = 2.79
    steering_ratio = 14.8

    total_reward = 0.0
    curvature_actions = []
    while True:
        #curvature_dist = run_driving_model(observation)
        #curvature_action = curvature_dist.sample()[0,0]

        #print( f'curvature_action={curvature_action}' )
#curvature_action
        #vista_step()

        new_obs = np.moveaxis(observation, -1, 0) 
        inputs = np.array( [new_obs] )

        print(f'input:', inputs.shape, inputs.dtype, 'max:', np.max(inputs), 'min:', np.min(inputs), 'mean:', np.mean(inputs))
        
        inpt_tens = torch.Tensor( inputs )
        

        predictions = nvidia_model(inpt_tens)
        steering_angle = predictions.detach().numpy()[0,0]
        #curvature_action = wheel_base / math.tan( steering_angle )
        
        frontal_wheel_angle = steering_angle / steering_ratio
        #turning_radius = wheel_base / math.tan(frontal_wheel_angle)
        #curvature_action = 1 / turning_radius

        curvature_action = math.tan( frontal_wheel_angle ) / wheel_base

        print( f'steering_angle={steering_angle} true_curvature={get_curvature()} curvature_action={curvature_action}' )
        vista_step( curvature_action )


        observation = grab_and_preprocess_obs(car)
        vis_img = display.render()

        stream.write(vis_img[:, :, ::-1], index=i_step)

        i_step += 1

        reward = 1.0 if not check_crash(car) else 0.0

        # is the episode over? did you crash or do so well that you're done?
        if reward == 0.0:
            break
        
    if total_reward >= 800:
        break

print("Saving trajectory...")
stream.save("trained_policy.avi")
#mdl.lab3.play_video("trained_policy.mp4")