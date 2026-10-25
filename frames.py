"""
frames.py = 2D frame conversions using 3x3 matrices.

Frames:
    world = the fixed map; landmark positions are known here
    robot = origin at the rover's centre, x-axis points forward, y-axis left
    sensor = origin at a sensor, x-axis points where the sensor looks (diff for each sensor)
"""

import numpy as np

def make_transform(x, y, theta):
    # make transform matrix frame b into frame a.
    # parameters are offset of b from a relative to a
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c,  -s,  x],
                     [s,   c,  y],  
                     [0.0, 0.0, 1.0]])

def transform_point(T_a_b, p_b):
    # convert point from frame b into frame a (T_a_b = make_transform)
    p_b_h = np.array([p_b[0], p_b[1], 1.0]) 
    p_a_h = T_a_b @ p_b_h
    return p_a_h[:2]                       

def pose_of(T_a_b):
    # convert matrix to pose for readability
    x = T_a_b[0, 2]
    y = T_a_b[1, 2]
    theta = np.arctan2(T_a_b[1, 0], T_a_b[0, 0])
    return x, y, theta

