# 2D Robot Sim

A small simulation of a robot standing still on a 2D map. It works out where the
robot is by combining a rough guess with noisy sensor readings of known landmarks,
using an **Extended Kalman Filter (EKF)**.

This is a "snapshot" EKF: it only does the **correct** step (using readings to fix
the guess). There is no **predict** step, so the robot doesn't move over time.

## How it works

1. The map has 10 landmarks at known positions.
2. The robot has a true position and heading that the EKF can't see.
3. Sensors on the robot measure the distance and angle to each landmark they can see,
   with random noise added.
4. The EKF starts from a rough guess and uses each reading to move the guess closer
   to the truth. It also tracks how unsure it still is (the covariance).

## Files

| File | What it does |
|---|---|
| `frames.py` | Converts points between the world, robot and sensor frames using 3x3 matrices |
| `sensors.py` | Sensor definitions (mounting spot, direction, noise, range, field of view) and the list of sensors on the robot |
| `simulate.py` | Landmark positions, the true robot pose, and fake noisy readings |
| `ekf.py` | The EKF correct step and the Jacobian (how readings change when the pose changes) |
| `main.py` | Interactive window for trying it out |

## Setup

Needs Python 3.10 or newer, plus `numpy` and `matplotlib`:

```
pip install numpy matplotlib
```

## Run

```
python main.py
```

In the window you can:

- **Click the map** to move the robot. Use the heading slider to turn it.
- Use the **guess error** sliders to set how wrong the starting guess is.
- Use the **noise** sliders to set how noisy the sensors are.
- Compare the **EKF estimate** (solid blue) with the **sensors-only estimate** (dashed green).
  Both use the same readings, but the sensors-only one ignores the guess.
- Turn the **front** and **rear** sensors on or off.
- Click **New noise** to draw a fresh set of random errors.
- Click **Run 200 trials** to see the average and worst error over 200 runs.

The dotted ellipse is the 95% region: the area where the EKF thinks the robot is,
with 95% confidence.

You can also run `python sensors.py` or `python simulate.py` on their own for quick
checks in the terminal.
