"""
ekf.py = improve localization data using EKF

guess = (x, y, theta), the robot pose on the map   
uncertainty = 3x3 covariance: how unsure we are about the guess
Works with sensors that measure distance, angle, or both.
"""

import numpy as np

from frames import wrap_angle
from sensors import SENSORS
from simulate import LANDMARKS, expected_reading

def measurement_jacobian(pose, landmark, sensor):
    # H = how (distance, angle) change when (x, y, theta) change a little
    # row 0 = distance, row 1 = angle
    x, y, theta = pose
    c, s = np.cos(theta), np.sin(theta)
    mx, my = sensor.x, sensor.y

    # sensor position on the map = robot position + mounting offset turned by theta
    sx = x + c * mx - s * my
    sy = y + s * mx + c * my
    # derivatives of sx and sy with respect to theta
    dsx_dth = -s * mx - c * my
    dsy_dth = c * mx - s * my

    # landmark distance relative to the sensor
    dx = landmark[0] - sx
    dy = landmark[1] - sy
    q = dx**2 + dy**2
    d = np.sqrt(q)

    # distance = sqrt(dx^2 + dy^2)
    # angle    = arctan2(dy, dx) - theta - mount angle
    # moving the robot +x makes dx smaller, so signs flip
    return np.array([
        [-dx / d, -dy / d, -(dx * dsx_dth + dy * dsy_dth) / d], # distance
        [ dy / q, -dx / q,  (dy * dsx_dth - dx * dsy_dth) / q - 1], # angle
    ])


def correct(guess, uncertainty, readings):
    # update the pose guess and its uncertainty with every reading
    for i, j, measured_distance, measured_angle in readings:
        sensor = SENSORS[i]
        landmark = LANDMARKS[j]
        expected_distance, expected_angle = expected_reading(guess, landmark, sensor)
        jacobian_full = measurement_jacobian(guess, landmark, sensor)

        # keep only the parts this sensor measures (1 or 2 rows)
        difference, jacobian, noise_std = [], [], []
        if measured_distance is not None:
            difference.append(measured_distance - expected_distance)
            jacobian.append(jacobian_full[0])
            noise_std.append(sensor.range_noise)
        if measured_angle is not None:
            difference.append(wrap_angle(measured_angle - expected_angle)) 
            jacobian.append(jacobian_full[1])
            noise_std.append(sensor.bearing_noise)
        difference = np.array(difference)                   # actual - expected reading 
        jacobian = np.array(jacobian)                       # slopes                     
        sensor_noise = np.diag(np.array(noise_std)**2)      # sensor variance            

        difference_uncertainty = jacobian @ uncertainty @ jacobian.T + sensor_noise
        
        
        gain = uncertainty @ jacobian.T @ np.linalg.inv(difference_uncertainty)

        guess = guess + gain @ difference                   # move the guess
        guess[2] = wrap_angle(guess[2])
        uncertainty = (np.eye(3) - gain @ jacobian) @ uncertainty 
    return guess, uncertainty
