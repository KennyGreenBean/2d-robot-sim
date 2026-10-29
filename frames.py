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
    # parameters are offset of 'b' from 'a' relative to 'a'
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

def inverse(T_a_b):
    # reverse a conversion: inverse(T_a_b) = T_b_a
    # solve p_a = R p_b + t for p_b  ->  p_b = R^T p_a - R^T t
    # (R^-1 = R^T for a rotation: rotating back by -theta undoes theta)
    R = T_a_b[:2, :2]
    t = T_a_b[:2, 2]
    T_b_a = np.eye(3)
    T_b_a[:2, :2] = R.T
    T_b_a[:2, 2] = -R.T @ t
    return T_b_a

def wrap_angle(theta):
    # keep  angle in [-pi, pi) eg. 350 deg -> -10 deg
    return (theta + np.pi) % (2.0 * np.pi) - np.pi

