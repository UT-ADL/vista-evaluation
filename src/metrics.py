import numpy as np

def calculate_whiteness(steering_angles, timestamps):
    steering_angles = np.array(steering_angles)
    timestamps = np.array(timestamps)

    current_angles = steering_angles[:-1]
    next_angles = steering_angles[1:]
    current_timestamps = timestamps[:-1]
    next_timestamps = timestamps[1:]

    delta_angles = next_angles - current_angles
    delta_timestamps = (next_timestamps - current_timestamps).astype(np.float32)
    whiteness = np.sqrt(((delta_angles / delta_timestamps) ** 2).mean())

    return whiteness
