import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import math

import vista
from vista.entities.agents.Dynamics import steering2curvature

from rosbag_to_vista import BOLT_DIR
from video_extractor import VideoStream

LEXUS_LENGTH = 4.89
LEXUS_WIDTH = 1.895
WHEEL_BASE = 2.79
STEERING_RATIO = 14.7
LOG_FREQUENCY_SEC = 1
FPS = 13


trace_root = os.path.join(BOLT_DIR, 'end-to-end', 'vista')
trace_path = [
    #"e2e_rec_vahi/",
    #"e2e_rec_link3/"
    #"custom_trace/",
    #"custom_trace2/",
    #"custom_trace_reverse/",
    # "20210726-154641_lexus_devens_center", 
    #"20210726-155941_lexus_devens_center_reverse", 
    #"20210726-184624_lexus_devens_center", 
    #"20210726-184956_lexus_devens_center_reverse", 
    "2022-08-31-15-37-37_elva_ebm_512_front"
]
trace_path = [os.path.join(trace_root, p) for p in trace_path]

# Create a virtual world with VISTA, the world is defined by a series of data traces
world = vista.World(trace_path, trace_config={'road_width': 4})

# Create a car in our virtual world. The car will be able to step and take different 
#   control actions. As the car moves, its sensors will simulate any changes it environment
car = world.spawn_agent(
    config={
        'length': LEXUS_LENGTH,
        'width': LEXUS_WIDTH,
        'wheel_base': WHEEL_BASE,
        'steering_ratio': STEERING_RATIO,
        'lookahead_road': False
    })

# Create a camera on the car for synthesizing the sensor data that we can use to train with! 
camera = car.spawn_camera(config={'name': 'camera_front', 
                                  'size': (1208, 1928),
                                  })

# Define a rendering display so we can visualize the simulated car camera stream and also 
#   get see its physical location with respect to the road in its environment. 
display = vista.Display(world, display_config={"gui_scale": 2, "vis_full_frame": True, })

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
    if curvature is None: 
        curvature = car.trace.f_curvature(car.timestamp)
    if speed is None: 
        speed = car.trace.f_speed(car.timestamp)
    
    car.step_dynamics(action=np.array([curvature, speed]), dt=1/FPS)
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

def is_done_or_crashed(car): 
    # return check_out_of_lane(car) or check_exceed_max_rot(car) or car.done
    return check_out_of_lane(car) or car.done


## Data preprocessing functions ##

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nvidia_model = PilotNet()
nvidia_model.load_state_dict(torch.load('models/mae-v2.pt', map_location=torch.device('cpu')))
nvidia_model.to(device)
nvidia_model.eval()

stream = VideoStream(FPS)
i_step = 0

vista_reset()
observation = grab_and_preprocess_obs(car)

while True:

    inference_start = time.perf_counter()
    new_obs = np.moveaxis(observation, -1, 0) 
    new_obs = torch.from_numpy(new_obs).float().unsqueeze(0).to(device)
    predictions = nvidia_model(new_obs)
    steering_angle = predictions.detach().cpu().numpy()[0,0]
    inference_time = time.perf_counter() - inference_start

    curvature = steering2curvature(math.degrees(steering_angle), WHEEL_BASE, STEERING_RATIO)

    step_start = time.perf_counter()
    vista_step(curvature)
    step_time = time.perf_counter() - step_start

    observation = grab_and_preprocess_obs(car)

    vis_start = time.perf_counter()
    vis_img = display.render()
    stream.write(vis_img[:, :, ::-1], index=i_step)
    vis_time = time.perf_counter() - vis_start

    print( f'dynamics step: {step_time:.2f}s | inference: {inference_time:.4f}s | visualization: {vis_time:.2f}s' )

    i_step += 1
    if i_step % (FPS * LOG_FREQUENCY_SEC) == 0:
        print(f'Step {i_step} ({i_step / FPS:.0f}s) - Still going...')

    if is_done_or_crashed(car):
        print(f'Crashed or Done at step {i_step} ({i_step / FPS:.0f}s)!')
        break
        
print("Saving trajectory...")
stream.save(f"model_run.avi")

print(f'\nReached {i_step} steps ({i_step / FPS:.0f}s)!')
