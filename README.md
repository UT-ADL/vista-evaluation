# Data:


[data](https://www.dropbox.com/s/62pao4mipyzk3xu/vista_traces.zip?dl=1)

# Files:

**vista_plus_pilotnet.py** - Use Nvidia model to drive in vista simulation. NOTE: Model need to downloaded separately.

**pilotnet.py** - containes implementation of Nvidia PilotNet model

**memory.py** - vista's implementation for tracking observation and reward.

**sim_control.py** - allow manual driving with arrow keys in vista simulator.

**training_policy.py** - this is vista script for learning control policy with reinforcement learning.

**rosbag_to_vista.py** - converts rosbag to vista format. In config file **rtv_config.json** you need to specify topic where corresponding information should be taken. For field *speed* and *curvature* additionally you have to specify column name. The format is following **<topic_name>.<column_name>**.

# Installation:

https://vista.csail.mit.edu/getting_started/installation.html

```
pip install vista mitdeeplearning numpy opencv-python pandas matplotlib tensorflow_probability tqdm tensorflow bagpy
```

