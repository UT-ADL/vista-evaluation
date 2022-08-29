# Data:


[data](https://www.dropbox.com/s/62pao4mipyzk3xu/vista_traces.zip?dl=1)

# Files:

pilotnet.py - for using nvidia model for inference. Model need to downloaded separately.

sim_control.py - for manual driving.

training_policy.py - training with reinforcement learning.

rosbag_to_vista.py - converting rosbag to vista format. At this point is messy.

# Installation:

https://vista.csail.mit.edu/getting_started/installation.html

```
pip install vista mitdeeplearning numpy opencv-python pandas matplotlib tensorflow_probability tqdm tensorflow bagpy
```

