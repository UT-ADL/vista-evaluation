# Copyright 2022 MIT 6.S191 Introduction to Deep Learning. All Rights Reserved.
# 
# Licensed under the MIT License. You may not use this file except in compliance
# with the License. Use and/or modification of this code outside of 6.S191 must
# reference:
#
# © MIT 6.S191: Introduction to Deep Learning
# http://introtodeeplearning.com
#
# Laboratory 3: Reinforcement Learning

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['EGL_DEVICE_ID'] = os.environ.get('CUDA_AVAILABLE_DEVICES', '0')

import functools
import shutil, subprocess, cv2

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tqdm import tqdm
import wandb
import vista
from vista.utils import logging

logging.setLevel(logging.ERROR)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


class Memory:
    def __init__(self): 
        self.clear()

    def clear(self): 
        self.observations = []
        self.actions = []
        self.rewards = []

    def add_to_memory(self, new_observation, new_action, new_reward): 
        self.observations.append(new_observation)
        self.actions.append(new_action)
        self.rewards.append(new_reward)

    def __len__(self):
        return len(self.actions)


def normalize(x):
    x -= np.mean(x)
    x /= np.std(x)
    return x.astype(np.float32)

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

        loss = loss_function(prediction, actions, discounted_rewards)

    grads = tape.gradient(loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 4)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

def vista_reset():
    world.reset()
    display.reset()

def vista_step(curvature=None, speed=None):
    if curvature is None: 
        curvature = car.trace.f_curvature(car.timestamp)
    if speed is None: 
        speed = car.trace.f_speed(car.timestamp)
    
    car.step_dynamics(action=np.array([curvature, speed]), dt=1/10.)
    car.step_sensors()


class VideoStream():
    def __init__(self):
        self.tmp = "./tmp"
        if os.path.exists(self.tmp) and os.path.isdir(self.tmp):
            shutil.rmtree(self.tmp)
        os.mkdir(self.tmp)
        self.index = 0
    def write(self, image, index=None):
        if index is None: index = self.index
        cv2.imwrite(os.path.join(self.tmp, f"{self.index:04}.png"), image)
        self.index = index + 1
    def save(self, fname):
        cmd = f"ffmpeg -f image2 -framerate 10 -i {self.tmp}/%04d.png -crf 0 -y {fname}"
        subprocess.call(cmd, shell=True)

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
    # return check_out_of_lane(car) or car.done

def preprocess(full_obs):
    # Extract ROI
    i1, j1, i2, j2 = camera.camera_param.get_roi()
    obs = full_obs[i1:i2, j1:j2]
    
    # Rescale to [0, 1]
    obs = obs / 255.
    obs = obs[:, :, ::-1] # BGR -> RGB
    return obs

def grab_and_preprocess_obs(car):
    full_obs = car.observations[camera.name]
    obs = preprocess(full_obs)
    return obs

def create_driving_model():
    act = tf.keras.activations.swish
    Conv2D = functools.partial(tf.keras.layers.Conv2D, padding='valid', activation=act)
    Flatten = tf.keras.layers.Flatten
    Dense = tf.keras.layers.Dense

    model = tf.keras.models.Sequential([
        Conv2D(filters=32, kernel_size=5, strides=2),
        Conv2D(filters=48, kernel_size=5, strides=2),
        Conv2D(filters=64, kernel_size=3, strides=2),
        Conv2D(filters=64, kernel_size=3, strides=2),
        Flatten(),
        Dense(units=128, activation=act),
        Dense(units=2, activation=None)
    ])
    return model

def run_driving_model(image, max_curvature = 1/8., max_std = 0.1):
    single_image_input = tf.rank(image) == 3  # missing 4th batch dimension
    if single_image_input:
        image = tf.expand_dims(image, axis=0)

    distribution = driving_model(image)


    mu, logsigma = tf.split(distribution, 2, axis=1)
    print('mu before tanh(mu) * max_curvature :', mu)

    mu = max_curvature * tf.tanh(mu) # conversion
    sigma = max_std * tf.sigmoid(logsigma) + 0.005 # conversion
    
    pred_dist = tfp.distributions.Normal(mu, sigma)
    return pred_dist


def compute_driving_loss(dist, actions, rewards):
    neg_logprob = -1 * dist.log_prob(actions)
    loss = tf.reduce_mean( neg_logprob * rewards )
    return loss


if __name__ == '__main__':

    wandb.init(project="ebm-driving", entity="nikebless", job_type="vista-policy-learning", config={})

    trace_root = "./traces"
    trace_path = [
        "20210726-154641_lexus_devens_center", 
        "20210726-155941_lexus_devens_center_reverse", 
        "20210726-184624_lexus_devens_center", 
        "20210726-184956_lexus_devens_center_reverse", 
    ]
    trace_path = [os.path.join(trace_root, p) for p in trace_path]

    world = vista.World(trace_path, trace_config={'road_width': 4})
    car = world.spawn_agent(
        config={
            'length': 5.,
            'width': 2.,
            'wheel_base': 2.78,
            'steering_ratio': 14.7,
            'lookahead_road': True
        })
    camera = car.spawn_camera(config={'size': (600, 960)})
    display = vista.Display(world, display_config={"gui_scale": 2, "vis_full_frame": False})

    driving_model = create_driving_model()
    learning_rate = 5e-4 # -6=rich-silence-767, -7=comfy-monkey-768, -3=clear-smoke-769
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    vista_reset()
    driving_model = create_driving_model()
    memory = Memory()
    # stream = VideoStream()

    max_batch_size = 300
    num_episodes = 1000
    max_reward = float('-inf') # keep track of the maximum reward acheived during training
    if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists
    for i_episode in tqdm(range(num_episodes)):

        # Restart the environment
        vista_reset()
        memory.clear()
        observation = grab_and_preprocess_obs(car)

        last_curvature_mean = 0.
        alpha = 0.01

        while True:
            curvature_dist = run_driving_model(observation)
            curvature_action = curvature_dist.sample()[0,0]

            if len(memory) % 10 == 0:
                curvature_dist_mean = curvature_dist.mean()[0,0]
                last_curvature_mean = alpha * curvature_dist_mean + (1 - alpha) * last_curvature_mean
                print('avg pred curvature *mean*:', last_curvature_mean)
            
            vista_step(curvature_action)
            observation = grab_and_preprocess_obs(car)
            reward = 1.0 if not check_crash(car) else 0.0
            
            memory.add_to_memory(observation, curvature_action, reward)
            # stream.write(observation * 255)               
            
            if reward == 0.0:
                total_reward = sum(memory.rewards)
                wandb.log({"train_episode_return": total_reward, "train_episode_length": len(memory)})
                
                #   if the training step is too large 
                #   we need to sample a mini-batch for this step.
                batch_size = min(len(memory), max_batch_size)
                i = np.random.choice(len(memory), batch_size, replace=False)
                train_step(driving_model, compute_driving_loss, optimizer, 
                                observations=np.array(memory.observations)[i],
                                actions=np.array(memory.actions)[i],
                                discounted_rewards = discount_rewards(memory.rewards)[i], 
                                custom_fwd_fn=run_driving_model)            
                memory.clear()
                # stream.save("policy_obs_during_training.mp4")
                break

        if total_reward >= 800:
            print('Reached reward threshold, stopping training prematurely.')
            break


    ## Evaluation block ##
    i_step = 0
    num_episodes = 5
    num_reset = 5
    stream = VideoStream()
    for i_episode in tqdm(range(num_episodes)):
        
        # Restart the environment
        vista_reset()
        observation = grab_and_preprocess_obs(car)
        
        print("rolling out in env")
        episode_step = 0
        while not check_crash(car) and episode_step < 100:
            curvature_dist = run_driving_model(observation)
            curvature = curvature_dist.mean()[0,0]

            vista_step(curvature)
            observation = grab_and_preprocess_obs(car)

            vis_img = display.render()
            stream.write(vis_img[:, :, ::-1], index=i_step)
            i_step += 1
            episode_step += 1

        for _ in range(num_reset):
            stream.write(np.zeros_like(vis_img), index=i_step)
            i_step += 1
            
    print(f"Average reward: {(i_step - (num_reset*num_episodes)) / num_episodes}")

    print("Saving trajectory with trained policy...")
    stream.save("trained_policy.mp4")

    wandb.finish()

    # * different model architectures, for example recurrent models or Transformers with self-attention;
    # * data augmentation and improved pre-processing;
    # * different data modalities from different sensor types. VISTA also supports [LiDAR](https://driving.ca/car-culture/auto-tech/what-is-lidar-and-how-is-it-used-in-cars) and [event-based camera](https://arxiv.org/pdf/1804.01310.pdf) data, with a [new VISTA paper](https://arxiv.org/pdf/2111.12083.pdf) describing this. If you are interested in this, please contact Alexander Amini (amini@mit.edu)!
    # * improved reinforcement learning algorithms, such as [PPO](https://paperswithcode.com/paper/proximal-policy-optimization-algorithms), [TRPO](https://paperswithcode.com/method/trpo), or [A3C](https://paperswithcode.com/method/a3c);
    # * different reward functions for reinforcemnt learning, for example penalizing the car's distance to the lane center rather than just whether or not it crashed;
    # * [Guided Policy Learning (GPL)](https://rll.berkeley.edu/gps/). Not reinforcement learning, but a powerful algorithm to leverage human data to provide additional supervision the learning task.

