# Evaluating Steering Models using VISTA Driving Simulator

This is a road-following benchmark for camera-based end-to-end steering models that uses the [VISTA Driving Simulator](https://github.com/vista-simulator/vista). 

VISTA takes a recording of a real-world drive and allows replaying it interactively with deviations from the original trajectory by reprojecting the view-point as desired. Thus, a simulator can be used for on-policy, closed-loop evaluation, allowing fast and reproducible model evaluation (as in Carla-like simulators) while staying visually in-distribution with real-world data.

The [Rally Estonia End-to-End Driving](https://github.com/UT-ADL/e2e-rally-estonia) provides a dataset and codebase for training baseline models.

## Task

The evaluation is done on a VISTA-simulated 4.3 km section of the WRC Rally Estonia 2022 [SS10+14 Elva](https://www.rally-maps.com/Rally-Estonia-2022/Elva) track, driven in both ways (8.6 km in total). The track was chosen to be challenging for humans, and includes elevations, curves and narrow rural roads. The speed in the original recording is around 35-45 km/h, chosen to be comfortable by the driver.

The only inputs available to the model are RGB image frames from the front-facing camera. The model is expected to output a steering angle in radians, where positive angles are left turns. The longitudinal (speed) control is not evaluated and is taken from the ground truth speed at the moment of the frame capture.

The key **evaluation metric** is the number of crashes, where a crash is defined as diverging from the human-driven center-line by more than 2 meters. When a crash occurs, the car is restarted two seconds further than the place of the crash.

![Screenshot from one of the recordings](extra/recording_shot.png)
![Screenshot from a VISTA run](extra/vista_shot.png)

## Download Traces

The VISTA simulator is run on *traces*, i.e. drive recordings in a VISTA-specific format. 

Download the official benchmark [traces](https://owncloud.ut.ee/owncloud/s/cRj2teJCLpYpMmz) from the University of Tartu ownCloud. There are two archives, representing two drives through the track in different directions: 
- `ebm-paper-mae-s2-forward_2022-09-23-10-31-24-resize` (4.3km section of SS10+14 Elva)
- `ebm-paper-mae-s2-backward_2022-09-21-12-34-47-resize` (4.3km section of SS10+14 Elva, reversed)

Unzip the archives and put the resulting two folders into the `traces` folder in the root of the repository.

> If you'd like extra, unofficial evaluation traces, you can:
> - use VISTA-provided [sample traces](https://www.dropbox.com/s/62pao4mipyzk3xu/vista_traces.zip?dl=1)
> - create your own trace. An example script creating a trace from a ROS bag is included.
> - (only members of the Autonomous Driving Lab at University of Tartu) use additional created traces from Oct 2021 and Sep 2022 Elva bags, found in `<shared-directory>/end-to-end/vista` on HPC. Use only the ones with `-resize` in the end of the name.

## Requirements

1. Linux (tested on Ubuntu 18.04 and CentOS 7.9)
2. An Nvidia GPU with at least 2GB of memory. More memory — more parallel evaluations.

## Install

0. Ensure to have VISTA's system dependencies:

```bash
sudo apt-get update
sudo apt-get install -y freeglut3-dev
sudo apt-get install -y libglib2.0-0
sudo apt-get install -y ffmpeg
```

1. Create a [conda](https://docs.conda.io/en/latest/miniconda.html) environment with Python 3.8 and activate it.

```bash
conda create -n vista python=3.8
conda activate vista
```

2. Install core python dependencies:

```bash
pip install -r requirements.txt
```

If you want to train your own vehicle dynamics model, install a more complete dependencies list:

```bash
pip install -r requirements-training.txt
```

3. (optional) Install ROS dependencies (only necessary for creating custom VISTA traces from your bag files):

```bash
pip install --extra-index-url https://rospypi.github.io/simple/ rospy rosbag roslz4 cv-bridge
```

4. Install our fork of VISTA:
```bash
pip install git+https://github.com/UT-ADL/vista.git
```

Our [fork](https://github.com/UT-ADL/vista) has the following improvements:
- added monocular depth estimation using [monodepth2](https://github.com/nianticlabs/monodepth2) for artifact-free view reprojection
    - the original [VISTA paper](https://ieeexplore.ieee.org/document/8957584) reported using monocular depth estimation in their experiments, citing [monodepth](https://github.com/mrharicot/monodepth), but the authors did not include it in the codebase
- fixed a bug that in rare cases created zero-length segments which made the simulator crash

## Usage

Scripts:

> Use the `--help` flag when calling a script to see the full list of arguments.

- evaluate.py - run a model on a set of VISTA traces and calculate the number of crashes. Main args:
    - `--model` - path to the steering model ONNX file to evaluate, **required**
    - `--traces` - names of traces in the `--trace-root` directory to run evaluation on, **required**
    - `--traces-root` - path to the folder containing the traces to evaluate on (`./traces` by default),
    - `--wandb-project` - if provided, results are logged to Weights & Biases.
    - `--model-type` - type of the model used for predicting steering (and optionally speed) (`steering_model` by default)
    - `--turn-signal` - if provided, turn signal information from the traces are used to decide driving direction on junctions 
- create_trace.py - an example script to create a VISTA trace from a ROS bag. Uses topics:
    - RGB camera: `/interfacea/link2/image/compressed`
    - speed: `/ssc/velocity_accel_cov`
    - curvature: `/ssc/curvature_feedback`

Environment variables:
- `PATH_TO_LEARNED_DYNAMICS_MODEL` - path to the learned dynamics model to use for smoothing the steering angle. Defaults to `./models/dynamics_model_v6_10hz.onnx`.
- `CUDA_VISIBLE_DEVICES` - a comma-separated list of GPU indices to use for evaluation. Defaults to `0`.
