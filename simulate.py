"""
simulate.py = simulate robot localization

The robot stands still at TRUE_POSE. Every sensor in SENSORS takes one set
of noisy range-bearing readings of the landmarks it can see. The robot only
has a rough GUESS of where it is; the EKF will use the readings to fix it.
"""

import numpy as np

from frames import make_transform, transform_point, inverse, wrap_angle
from sensors import SENSORS

# known landmark positions on the map (m)
LANDMARKS = np.array([
    [-9.0,  0.0], [-5.0,  5.0], [-5.0, -5.0], [-2.0,  1.0], [ 0.0,  5.5],
    [ 0.0, -5.5], [ 2.0, -1.0], [ 5.0,  5.0], [ 5.0, -5.0], [ 9.0,  0.0],
])

# where the robot really is (x, y, heading) - hidden from the EKF
TRUE_POSE = np.array([0.0, 3, -.0, 40])


def expected_reading(pose, landmark, sensor):
    # noise-free (distance & angle) of a landmark, as seen by this sensor
    # chain: world -> robot -> sensor
    T_world_robot = make_transform(pose[0], pose[1], pose[2])
    T_world_sensor = T_world_robot @ sensor.T_robot_sensor()
    p = transform_point(inverse(T_world_sensor), landmark)  # landmark in sensor frame
    d = np.hypot(p[0], p[1])        # distance
    angle = np.arctan2(p[1], p[0])      # angle from the sensor's forward axis
    return d, angle


def take_readings(pose, noisy=True):
    # every sensor reads every landmark it can see
    # returns a list of (sensor index, landmark index, distance, angle)
    readings = []
    for i, sensor in enumerate(SENSORS):
        for j, landmark in enumerate(LANDMARKS):
            d, angle = expected_reading(pose, landmark, sensor)
            if d > sensor.max_range or abs(angle) > sensor.fov / 2:
                continue   # out of range or outside fov
            elif noisy:
                d = d + np.random.normal(0, sensor.range_noise)
                angle = wrap_angle(angle + np.random.normal(0, sensor.bearing_noise))
            readings.append((i, j, d, angle))
    return readings


if __name__ == "__main__":

    readings = take_readings(TRUE_POSE)
    print(f"true pose: {TRUE_POSE[:2]}, {np.degrees(TRUE_POSE[2]):.0f} deg")
    print(f"{len(readings)} readings:")
    for i, j, d, angle in readings:
        print(f"  {SENSORS[i].name} sees landmark {j} at {LANDMARKS[j]}: "
              f"distance {d:.2f} m, angle {np.degrees(angle):.1f} deg")
    print("simulate.py: checks passed.")
