import os
import sys

import dotenv
dotenv.load_dotenv()
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['EGL_DEVICE_ID'] = os.environ.get('CUDA_AVAILABLE_DEVICES', '0')

import uuid
import time
from datetime import datetime
from collections import defaultdict
import argparse

import wandb
import numpy as np
import math
import tqdm

import vista
from vista.entities.agents.Dynamics import steering2curvature
from vista.entities.sensors.camera_utils.ViewSynthesis import DepthModes

from src.preprocessing import grab_and_preprocess_obs, get_camera_size
from src.metrics import calculate_whiteness
from src.video import VideoStream
from src.dynamics_model import OnnxDynamicsModel, ExpMovingAverageDynamicsModel
from src.steering_model import ConditionalSteeringModel, ConditionalWaypointsModel, SteeringModel
from src.car_constants import LEXUS_LENGTH, LEXUS_WIDTH, LEXUS_WHEEL_BASE, LEXUS_STEERING_RATIO

PATH_TO_LEARNED_DYNAMICS_MODEL = os.environ.get('PATH_TO_LEARNED_DYNAMICS_MODEL', os.path.join(os.path.dirname(__file__), 'models', 'dynamics_model_v6_10hz.onnx'))
OUTPUT_DIR = 'out'

LOG_FREQUENCY_SEC = 1
SRC_FPS = 30
FPS = 10
SECONDS_SKIP_AFTER_CRASH = 2
FRAME_RESET_OFFSET = 1

os.makedirs(OUTPUT_DIR, exist_ok=True)

def vista_step(car, curvature=None, speed=None):
    if curvature is None: 
        curvature = car.trace.f_curvature(car.timestamp)
    if speed is None: 
        speed = car.trace.f_speed(car.timestamp)
    
    car.step_dynamics(action=np.array([curvature, speed]), dt=1/FPS)
    car.step_sensors()

def check_out_of_lane(car):
    distance_from_center = np.abs(car.relative_state.x)
    road_width = car.trace.road_width 
    half_road_width = road_width / 2
    return distance_from_center > half_road_width


def run_evaluation_episode(trace_name, model, world, camera, car, display, logging, video_dir, save_video=False, resize_mode='resize', dynamics_model=None):
    if save_video:
        stream = VideoStream(os.path.join(video_dir, 'full.avi'), FPS, lossless=False)
        stream_cropped = VideoStream(os.path.join(video_dir, 'cropped.avi'), FPS, lossless=True)

    i_step = 0
    i_segment = 0
    last_driven_frame_idx = 0
    crash_times = []

    world.reset()
    car.reset(0, i_segment, FRAME_RESET_OFFSET)
    display.reset()
    observation = grab_and_preprocess_obs(car, camera, resize_mode)

    car_timestamp_start = car._timestamp
    n_segments = len(car.trace.good_timestamps[car.trace._multi_sensor.master_sensor])
    segments_lengths = [len(seg) for seg in car.trace.good_timestamps[car.trace._multi_sensor.master_sensor]]
    total_frames = sum(segments_lengths)

    timestamps = []
    cmd_steering_angle_history = []
    eff_steering_angle_history = []

    progress = tqdm.tqdm(total=total_frames, desc=trace_name, unit='frames')

    while True:

        inference_start = time.perf_counter()
        model_input = np.moveaxis(observation, -1, 0)
        model_input = np.expand_dims(model_input, axis=0)
        steering_angle, speed = model.predict(model_input, car)

        timestamps.append(car.timestamp)
        cmd_steering_angle_history.append(math.degrees(steering_angle))
        eff_steering_angle_history.append(math.degrees(car._ego_dynamics.steering))

        if dynamics_model is not None:
            steering_angle = dynamics_model.predict(steering_angle)
        inference_time = time.perf_counter() - inference_start

        curvature = steering2curvature(math.degrees(steering_angle), LEXUS_WHEEL_BASE, LEXUS_STEERING_RATIO)

        step_start = time.perf_counter()
        vista_step(car, curvature, speed)
        step_time = time.perf_counter() - step_start

        observation = grab_and_preprocess_obs(car, camera, resize_mode)
        i_step += 1

        vis_start = time.perf_counter()
        if save_video:
            vis_img = display.render()
            stream.write(vis_img[:, :, ::-1])
            stream_cropped.write(observation[:, :, ::-1] * 255.)
        vis_time = time.perf_counter() - vis_start

        cmd_whiteness = calculate_whiteness(cmd_steering_angle_history, timestamps)
        eff_whiteness = calculate_whiteness(eff_steering_angle_history, timestamps)

        logging.debug( f'\nStep {i_step} ({car._timestamp - car_timestamp_start:.0f}s, segment: {i_segment}, frame: {last_driven_frame_idx}) env step: {step_time:.2f}s | inference: {inference_time:.4f}s | visualization: {vis_time:.2f}s | crashes: {len(crash_times)} | whiteness_cmd: {cmd_whiteness:.2f} | whiteness_eff: {eff_whiteness:.2f}')

        if check_out_of_lane(car):
            crash_times.append(car._timestamp - car_timestamp_start)
            restart_at_frame = last_driven_frame_idx + SECONDS_SKIP_AFTER_CRASH*SRC_FPS
            logging.debug(f'Crashed at step {i_step} (frame={last_driven_frame_idx}) ({car._timestamp - car_timestamp_start:.0f}s). Re-engaging at frame {restart_at_frame}!')

            try:
                car.reset(0, i_segment, restart_at_frame)
            except IndexError:
                if i_segment == n_segments - 1:
                    logging.debug(f'Finished trace at step {i_step} ({car._timestamp - car_timestamp_start:.0f}s).')
                    break
                logging.debug(f'Finished segment {i_segment} at step ({i_step}) ({car._timestamp - car_timestamp_start:.0f}s).')
                i_segment += 1
                car.reset(0, i_segment, FRAME_RESET_OFFSET)

            display.reset()
            if dynamics_model is not None: dynamics_model.reset()
            observation = grab_and_preprocess_obs(car, camera, resize_mode)

        if car.done:
            if i_segment < n_segments - 1:
                logging.debug(f'Finished segment {i_segment} at step ({i_step}) ({car._timestamp - car_timestamp_start:.0f}s).')
                i_segment += 1
                car.reset(0, i_segment, FRAME_RESET_OFFSET)
                display.reset()
                if dynamics_model is not None: dynamics_model.reset()
                observation = grab_and_preprocess_obs(car, camera, resize_mode)
            else:
                logging.debug(f'Finished trace at step {i_step} ({car._timestamp - car_timestamp_start:.0f}s).')
                break

        last_driven_frame_idx = car.frame_index
        inc = last_driven_frame_idx - progress.n
        progress.update(n=inc)

    cmd_whiteness = calculate_whiteness(cmd_steering_angle_history, timestamps)
    eff_whiteness = calculate_whiteness(eff_steering_angle_history, timestamps)

    run_metrics = {
        'crash_times': crash_times,
        'cmd_whiteness': cmd_whiteness,
        'eff_whiteness': eff_whiteness,
    }

    logging.debug(f'\nLast trace crashes: {len(crash_times)}')

    if save_video:
        logging.debug('Saving trace videos to:', video_dir)
        stream.save()
        stream_cropped.save()

    return run_metrics


def create_steering_model(args):
    if args.model_type == "steering":
        model = SteeringModel(args.model)
    elif args.model_type == "conditional-steering":
        model = ConditionalSteeringModel(args.model, args.speed_model)
    elif args.model_type == "conditional-waypoints":
        model = ConditionalWaypointsModel(args.model, args.speed_model)
    else:
        print(f"Uknown model type {args.model_type}")
        sys.exit()

    return model


if __name__ == '__main__':

    run_start_time = int(time.time())

    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb-project', type=str, help='Weights and Biases project for logging.')
    parser.add_argument('--save-video', action='store_true', help='Save video of model run.')
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model to predict vehicle steering.')
    parser.add_argument('--speed-model', type=str, required=False, help='Path to ONNX model to predict vehicle speed.')
    parser.add_argument('--model-type', type=str, required=False, default="steering", help="Model type to use for making predictions.")
    parser.add_argument('--resize-mode', default='resize', choices=['full', 'resize'], help='Resize mode of the input images (bags pre-processed for Vista).')
    parser.add_argument('--dynamics', default='learned', choices=['learned', 'exp-mov-avg', 'none'], help='Which vehicle dynamics model to use. Defaults to a learned 10hz GRU model.')
    parser.add_argument('--road-width', type=float, default=4.0, help='Vista road width in meters.')
    parser.add_argument('--comment', type=str, default=None, help='W&B run description.')
    parser.add_argument('--tags', type=str, nargs='+', default=[], help='W&B run tags.')
    parser.add_argument('--depth-mode', type=str, default='monodepth', choices=['fixed_plane', 'monodepth'], help='''Depth approximation mode. Monodepth uses a neural network to estimate depth from a single image, 
                                                                                                                     resulting in fewer artifacts in synthesized images. Fixed plane uses a fixed plane at a fixed distance from the camera.''')
    parser.add_argument('--turn-signals', action='store_true', help="Use turn signals for conditioning driving directions.")
    parser.add_argument('--traces', type=str, nargs='+', default=None, required=True, help='Traces to evaluate on.')
    parser.add_argument('--traces-root', type=str, default='./traces', help='Root directory of traces. Defaults to `./traces`.')
    parser.add_argument('--verbose', action='store_true', help='Print debug messages.')
    args = parser.parse_args()
    print(vars(args))

    import logging
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    # aquire GPU early (helpful for distributing runs across GPUs on a single machine)
    model = create_steering_model(args)

    dynamics_model = None
    if args.dynamics == 'learned':
        dynamics_model = OnnxDynamicsModel(PATH_TO_LEARNED_DYNAMICS_MODEL)
    elif args.dynamics == 'exp-mov-avg':
        dynamics_model = ExpMovingAverageDynamicsModel()

    if args.wandb_project is not None:
        config = {
            'model_path': args.model,
            'trace_paths': args.traces,
            'dynamics': args.dynamics,
            'dynamics_model': PATH_TO_LEARNED_DYNAMICS_MODEL,
            'resize_mode': args.resize_mode,
            'save_video': args.save_video,
            'road_width': args.road_width,
            'depth_mode': args.depth_mode,
            'turn_signals': args.turn_signals
        }
        wandb.init(project=args.wandb_project, config=config, job_type='vista-evaluation', notes=args.comment, tags=args.tags)

    trace_paths = [os.path.join(args.traces_root, track_path) for track_path in args.traces]
    total_n_crashes = 0
    metrics_by_trace = defaultdict(lambda: defaultdict(float))

    model_name = os.path.basename(args.model).replace('.onnx', '')
    date_time_str = datetime.fromtimestamp(run_start_time).replace(microsecond=0).isoformat()
    unique_chars = uuid.uuid4().hex[:3] # uuid4 is robust to same-node time collisions
    run_dir = os.path.join(OUTPUT_DIR, f'{date_time_str}_{unique_chars}_{model_name}')

    for trace in trace_paths:
        run_trace_dir = os.path.join(run_dir, os.path.basename(trace))
        os.makedirs(run_trace_dir) # will fail if the directory already exists

        world = vista.World([trace], trace_config={'road_width': args.road_width, 'turn_signals': args.turn_signals})
        car = world.spawn_agent(
            config={
                'length': LEXUS_LENGTH,
                'width': LEXUS_WIDTH,
                'wheel_base': LEXUS_WHEEL_BASE,
                'steering_ratio': LEXUS_STEERING_RATIO,
                'lookahead_road': False
            })

        camera_size = get_camera_size(args.resize_mode)
        camera = car.spawn_camera(config={'name': 'camera_front', 'size': camera_size, 'depth_mode': DepthModes.MONODEPTH})
        display = vista.Display(world, display_config={'gui_scale': 2, 'vis_full_frame': True })

        metrics = run_evaluation_episode(os.path.basename(trace), model, world, camera, car, display, logging,
                                                           save_video=args.save_video, 
                                                           video_dir=run_trace_dir,
                                                           resize_mode=args.resize_mode,
                                                           dynamics_model=dynamics_model)
        
        metrics_by_trace[trace] = metrics

        # cleanup
        del camera._view_synthesis._renderer
        del camera
        del car
        del display
        del world

    print(f'\nMetrics by trace:')
    for trace, metrics in metrics_by_trace.items():
        crash_times = metrics['crash_times']
        cmd_whiteness = metrics['cmd_whiteness']
        eff_whiteness = metrics['eff_whiteness']
        print(f'{trace}: {len(crash_times)} crashes. whiteness_cmd: {cmd_whiteness:.2f}, whiteness_eff: {eff_whiteness:.2f}')
        for crash_time in crash_times:
            print(f'  > {crash_time:.0f}s')

    total_crashes = sum([len(metrics['crash_times']) for metrics in metrics_by_trace.values()])
    avg_cmd_whiteness = np.mean([metrics['cmd_whiteness'] for metrics in metrics_by_trace.values()])
    avg_eff_whiteness = np.mean([metrics['eff_whiteness'] for metrics in metrics_by_trace.values()])

    print(f'\nTotal crashes: {total_crashes}')
    print(f'Average command whiteness: {avg_cmd_whiteness:.2f}')
    print(f'Average effective whiteness: {avg_eff_whiteness:.2f}')
    print(f'Time spent: {time.time() - run_start_time:.0f}s ({(time.time() - run_start_time) / 60:.2f}min)')

    if args.wandb_project is not None:
        wandb.log({'crash_count': total_crashes, 'whiteness_cmd': avg_cmd_whiteness, 'whiteness_eff': avg_eff_whiteness})
        wandb.finish()
