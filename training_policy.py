import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
import matplotlib, cv2
import matplotlib.pyplot as plt
import base64, io, os, time, gym
import IPython, functools
import time
from tqdm import tqdm
import tensorflow_probability as tfp

import mitdeeplearning as mdl

import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

from memory import Memory

import vista
from vista.utils import logging
logging.setLevel(logging.ERROR)

### Reward function ###

# Helper function that normalizes an np.array x
def normalize(x):
    x -= np.mean(x)
    x /= np.std(x)
    return x.astype(np.float32)

# Compute normalized, discounted, cumulative rewards (i.e., return)
# Arguments:
#   rewards: reward at timesteps in episode
#   gamma: discounting factor
# Returns:
#   normalized discounted reward
def discount_rewards(rewards, gamma=0.95): 
    discounted_rewards = np.zeros_like(rewards)
    R = 0
    for t in reversed(range(0, len(rewards))):
        # update the total discounted reward
        R = R * gamma + rewards[t]
        discounted_rewards[t] = R
      
    return normalize(discounted_rewards)

def train_step(model, loss_function, optimizer, observations, actions, discounted_rewards, custom_fwd_fn=None):
    with tf.GradientTape() as tape:
        # Forward propagate through the agent network
        if custom_fwd_fn is not None:
            prediction = custom_fwd_fn(observations)
        else: 
            prediction = model(observations)

        '''TODO: call the compute_loss function to compute the loss'''
        loss = loss_function(prediction, actions, discounted_rewards) # TODO
        # loss = loss_function('''TODO''', '''TODO''', '''TODO''')

    '''TODO: run backpropagation to minimize the loss using the tape.gradient method. 
             Unlike supervised learning, RL is *extremely* noisy, so you will benefit 
             from additionally clipping your gradients to avoid falling into 
             dangerous local minima. After computing your gradients try also clipping
             by a global normalizer. Try different clipping values, usually clipping 
             between 0.5 and 5 provides reasonable results. '''
    grads = tape.gradient(loss, model.trainable_variables) # TODO
    # grads = tape.gradient('''TODO''', '''TODO''')
    grads, _ = tf.clip_by_global_norm(grads, 4)
    # grads, _ = tf.clip_by_global_norm(grads, '''TODO''')
    optimizer.apply_gradients(zip(grads, model.trainable_variables))



trace_root = "../traces/" #"./vista_traces"
trace_path = [
    #"e2e_rec_vahi/",
    #"e2e_rec_link3/"
    #"custom_trace/",
    #"custom_trace2/",
    #"custom_trace_reverse/",
    "data"
    #"20210726-154641_lexus_devens_center", 
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
camera = car.spawn_camera(config={'size': (640, 920)}) #(200, 320)

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

def vista_step(curvature=None, speed=None):
    # Arguments:
    #   curvature: curvature to step with
    #   speed: speed to step with
    if curvature is None: 
        curvature = car.trace.f_curvature(car.timestamp)
    if speed is None: 
        speed = car.trace.f_speed(car.timestamp)
    
    print( f'curvature={curvature} speed={speed}' )

    car.step_dynamics(action=np.array([curvature, speed]), dt=1/15.)
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

def preprocess(full_obs):
    # Extract ROI
    i1, j1, i2, j2 = camera.camera_param.get_roi()
    obs = full_obs[i1:i2, j1:j2]
    
    # Rescale to [0, 1]
    obs = obs / 255.
    return obs

def grab_and_preprocess_obs(car):
    full_obs = car.observations[camera.name]
    obs = preprocess(full_obs)
    return obs


### Define the self-driving agent ###
# Note: we start with a template CNN architecture -- experiment away as you 
#   try to optimize your agent!

# Functionally define layers for convenience
# All convolutional layers will have ReLu activation
act = tf.keras.activations.swish
Conv2D = functools.partial(tf.keras.layers.Conv2D, padding='valid', activation=act)
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense

# Defines a CNN for the self-driving agent
def create_driving_model():
    model = tf.keras.models.Sequential([
        # Convolutional layers
        # First, 32 5x5 filters and 2x2 stride
        Conv2D(filters=32, kernel_size=5, strides=2),

        # TODO: define convolutional layers with 48 5x5 filters and 2x2 stride
        Conv2D(filters=48, kernel_size=5, strides=2), # TODO
        # Conv2D('''TODO'''),

        # TODO: define two convolutional layers with 64 3x3 filters and 2x2 stride
        Conv2D(filters=64, kernel_size=3, strides=2), # TODO
        Conv2D(filters=64, kernel_size=3, strides=2), # TODO
        # Conv2D('''TODO'''),

        Flatten(),

        # Fully connected layer and output
        Dense(units=128, activation=act),
        # TODO: define the output dimension of the last Dense layer. 
        #    Pay attention to the space the agent needs to act in.
        #    Remember that this model is outputing a distribution of *continuous* 
        #    actions, which take a different shape than discrete actions.
        #    How many outputs should there be to define a distribution?'''
        Dense(units=2, activation=None) # TODO
        # Dense('''TODO''')

    ])
    return model

driving_model = create_driving_model()


## The self-driving learning algorithm ##

# hyperparameters
max_curvature = 1/8
max_std = 0.1

def run_driving_model(image):
    # Arguments:
    #   image: an input image
    # Returns:
    #   pred_dist: predicted distribution of control actions 
    single_image_input = tf.rank(image) == 3  # missing 4th batch dimension
    if single_image_input:
        image = tf.expand_dims(image, axis=0)

    distribution = driving_model(image) # TODO

    mu, logsigma = tf.split(distribution, 2, axis=1)
    mu = max_curvature * tf.tanh(mu) # conversion
    sigma = max_std * tf.sigmoid(logsigma) + 0.005 # conversion
    
    #print( f'mu={mu}, sigma={sigma}' )

    pred_dist = tfp.distributions.Normal(mu, sigma) # TODO
    return pred_dist


def compute_driving_loss(dist, actions, rewards):
    # Arguments:
    #   logits: network's predictions for actions to take
    #   actions: the actions the agent took in an episode
    #   rewards: the rewards the agent received in an episode
    # Returns:
    #   loss
    neg_logprob = -1 * dist.log_prob(actions)

    loss = tf.reduce_mean( neg_logprob * rewards ) # TODO
    return loss

## Training parameters and initialization ##
## Re-run this cell to restart training from scratch ##

''' TODO: Learning rate and optimizer '''
learning_rate = 5e-10
# learning_rate = '''TODO'''
optimizer = tf.keras.optimizers.Adam(learning_rate)
# optimizer = '''TODO'''

# instantiate driving agent
vista_reset()
driving_model = create_driving_model()
# NOTE: the variable driving_model will be used in run_driving_model execution

# to track our progress
smoothed_reward = mdl.util.LossHistory(smoothing_factor=0.9)
plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Rewards')

# instantiate Memory buffer
memory = Memory()

## Evaluation block!##

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
        cmd = f"/usr/bin/ffmpeg -f image2 -i {self.tmp}/%04d.png -crf 0 -y {fname}"
        subprocess.call(cmd, shell=True)
        

stream = VideoStream()
i_step = 0

### ### ###  ###
### TRAINING ###
### ### ###  ###

## Driving training! Main training block. ##
## Note: stopping and restarting this cell will pick up training where you
#        left off. To restart training you need to rerun the cell above as 
#        well (to re-initialize the model and optimizer)
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(3, 6))

max_batch_size = 300
max_reward = float('-inf') # keep track of the maximum reward acheived during training
if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists

for i_episode in range(2):

    plotter.plot(smoothed_reward.get())
    # Restart the environment
    vista_reset()
    memory.clear()
    observation = grab_and_preprocess_obs(car)
    
    total_reward = 0.0
    curvature_actions = []
    while True:
        curvature_dist = run_driving_model(observation)
        curvature_action = curvature_dist.sample()[0,0]

        #print( f'curvature_action={curvature_action}' )
#curvature_action
        vista_step()
        #vista_step( curvature_action )
        observation = grab_and_preprocess_obs(car)
        vis_img = display.render()
        stream.write(vis_img[:, :, ::-1], index=i_step)
        i_step += 1

        reward = 1.0 if not check_crash(car) else 0.0

        # add to memory
        memory.add_to_memory(observation, curvature_action, reward)
        curvature_actions.append( curvature_action )

        print( f'total_reward={total_reward}' )
        
        if total_reward >= 1200:
            break

        # is the episode over? did you crash or do so well that you're done?
        if reward == 0.0:
            # determine total reward and keep a record of this
            total_reward = sum(memory.rewards)

            print( f'total_reward={total_reward}' )
            smoothed_reward.append(total_reward)

            #print( f'min_act={min(curvature_actions)}, max_act={max(curvature_actions)}' )

            # execute training step - remember we don't know anything about how the 
            #   agent is doing until it has crashed! if the training step is too large 
            #   we need to sample a mini-batch for this step.
            batch_size = min(len(memory), max_batch_size)
            i = np.random.choice(len(memory), batch_size, replace=False)
            train_step(driving_model, compute_driving_loss, optimizer, 
                               observations=np.array(memory.observations)[i],
                               actions=np.array(memory.actions)[i],
                               discounted_rewards = discount_rewards(memory.rewards)[i], 
                               custom_fwd_fn=run_driving_model)   

            plt.plot([i_episode], [total_reward] )
            plt.show()

            fig.savefig('progress.png', dpi=fig.dpi)

            # reset the memory
            memory.clear()
            break
        
    if total_reward >= 800:
        break





### ### ### ### ### ### ### ###
### FINAL RESULT VALIDATION ###
### ### ### ### ### ### ### ###


# i_step = 0
# num_episodes = 5
# num_reset = 5
# stream = VideoStream()
# for i_episode in range(num_episodes):
    
#     # Restart the environment
#     vista_reset()
#     observation = grab_and_preprocess_obs(car)
    
#     print("rolling out in env")
#     episode_step = 0
#     while True:
#         # using our observation, choose an action and take it in the environment
#         curvature_dist = run_driving_model(observation)
#         curvature = curvature_dist.mean()[0,0]

#         print(f'curvature={curvature}')
#         # Step the simulated car with the same action
#         vista_step(curvature)
#         observation = grab_and_preprocess_obs(car)

#         vis_img = display.render()
#         stream.write(vis_img[:, :, ::-1], index=i_step)
#         i_step += 1
#         episode_step += 1
    
#         if check_crash(car) or episode_step >= 800:
#             break
#     break
#     for _ in range(num_reset):
#         stream.write(np.zeros_like(vis_img), index=i_step)
#         i_step += 1
        
print("Saving trajectory with trained policy...")
stream.save("trained_policy.avi")
#mdl.lab3.play_video("trained_policy.mp4")