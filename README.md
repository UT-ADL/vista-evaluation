# Scripts using the Vista Simulator

This repo uses the [Vista Simulator](https://vista.csail.mit.edu/index.html) to:
- evaluate models in the simulator
- train models on-policy in the simulator
## Install

0. Ensure to have system dependencies:

```bash
sudo apt-get update
sudo apt-get install -y freeglut3-dev
sudo apt-get install -y libglib2.0-0
sudo apt-get install -y ffmpeg
```

1. Create a conda environment with Python 3.8 (unless you have Python3.8 environment already) and activate it.

```bash
conda create -n vista python=3.8
conda activate vista
```

2. Install main python dependencies:

```bash
pip install -r requirements.txt
```

3. Install ROS dependencies:

```bash
pip install --extra-index-url https://rospypi.github.io/simple/ rospy rosbag roslz4 cv-bridge
```

4. Install our fork of Vista (with bug fixes and monocular depth estimation):
```bash
pip install git+https://gitlab.cs.ut.ee/autonomous-driving-lab/e2e/vista.git
```

It will ask you for your username and password to the CS UT Gitlab.

## Get Traces

Vista simulation is run on *traces*, i.e. recordings of a drive in a Vista-specific format. You have several options:

- use Vista-provided [sample traces](https://www.dropbox.com/s/62pao4mipyzk3xu/vista_traces.zip?dl=1)
- use some of the already created traces from Oct 2021 and Sep 2022 Elva bags (found in `<shared-Bolt-directory>/end-to-end/vista`)
- create a trace from a new bag

## Usage

Scripts:
- evaluate.py - run a model on a set of Vista traces and calculate the number of crashes. Can upload the results to wandb.
    - If no traces are provided, we use two traces (forward and backward) from model-driven bags driven through a section of the Elva track, used in the EBM paper.
- create_trace.py - create a Vista trace from a ROS bag. Uses topics:
    - wide camera: `/interfacea/link2/image/compressed`
    - speed: `/ssc/velocity_accel_cov`
    - curvature: `/ssc/curvature_feedback`
- train_rl.py - train a reinforcement learning agent on a set of Vista traces. Doesn't work yet.

Use the `--help` flag to see the full list of arguments for each script.
