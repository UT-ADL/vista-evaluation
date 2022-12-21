import math

import numpy as np
from scipy import optimize


# https://scipy.github.io/old-wiki/pages/Cookbook/Least_Squares_Circle.html
def fit_circle(x, y):
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)

    def calc_R(xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def f_2(c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = x_m, y_m
    center_2, ier = optimize.leastsq(f_2, center_estimate)

    xc_2, yc_2 = center_2
    Ri_2 = calc_R(*center_2)
    R_2 = Ri_2.mean()
    return xc_2, yc_2, R_2


def calculate_steering_angle(waypoints, num_waypoints, ref_distance, use_vehicle_pos, latitudinal_correction):

    # use specified number of waypoints
    waypoints = waypoints[:2*num_waypoints]

    # add current vehicle position to the trajectory
    if use_vehicle_pos:
        waypoints = np.hstack(([0.0, 0.0], waypoints))

    wp_x = waypoints[::2]
    wp_y = waypoints[1::2] + latitudinal_correction
    x_center, y_center, radius = fit_circle(wp_x, wp_y)

    current_pos_theta = math.atan2(0 - y_center, 0 - x_center)
    circumference = 2 * np.pi * radius
    if current_pos_theta < 0:
        next_pos_theta = current_pos_theta + (2 * np.pi / circumference) * ref_distance
    else:
        next_pos_theta = current_pos_theta - (2 * np.pi / circumference) * ref_distance

    next_x = x_center + radius * np.cos(next_pos_theta)
    next_y = y_center + radius * np.sin(next_pos_theta)

    return np.arctan(next_y / next_x) * 14.7
