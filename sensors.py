"""
sensors.py = sensor definitions and the list of sensors on the robot.
"""

from dataclasses import dataclass

import numpy as np

from frames import make_transform, transform_point


@dataclass
class Sensor:
    name: str
    x: float              # mounting position, forward of robot centre (m)
    y: float              # mounting position, left of robot centre (m)
    theta: float          # direction it points, relative to robot forward (rad)
    range_noise: float    # std dev of distance error (m)
    bearing_noise: float  # std dev of angle error (rad)
    max_range: float      # farthest it can see (m)
    fov: float            # full width of its view cone (rad)

    def T_robot_sensor(self):
        # converts points from this sensor's frame into the robot frame
        return make_transform(self.x, self.y, self.theta)


SENSORS = [
    Sensor("front", x=0.4, y=0.15, theta=0,
           range_noise=0.1, bearing_noise=2,
           max_range=6.0, fov=np.radians(90)),
]


if __name__ == "__main__":
    front = SENSORS[0]

    # test: (1,0) sensor transform onto robot frame
    p_robot = transform_point(front.T_robot_sensor(), [1.0, 0.0])
    a = 0
    if not np.allclose(p_robot, [0.4 + np.cos(a), 0.15 + np.sin(a)]):
        print("not working")

    for s in SENSORS:
        print(f"{s.name}: at ({s.x}, {s.y}) m, pointing {np.degrees(s.theta):.0f} deg, "
              f"range {s.max_range} m, fov {np.degrees(s.fov):.0f} deg")
    print("sensors.py: checks passed.")
