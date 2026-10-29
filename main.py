"""
main.py = run the snapshot: simulate readings, correct the guess, plot it,
and compare the front sensor alone vs every sensor.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Wedge

from frames import make_transform, transform_point, wrap_angle
from sensors import SENSORS
from simulate import LANDMARKS, TRUE_POSE, take_readings
from ekf import correct

# rough first guess = truth + this error (0.5 m, -0.3 m, 10 deg)
GUESS_ERROR = np.array([0.5, -0.3, np.radians(10)])
# how unsure we are about the guess (std dev 0.5 m, 0.5 m, 10 deg)
START_UNCERTAINTY = np.diag([0.5**2, 0.5**2, np.radians(10)**2])

TRUE_COLOR = "#0b0b0b"
GUESS_COLOR = "#eb6834"
ESTIMATE_COLOR = "#2a78d6"
SENSOR_COLOR = "#1baf7a"
MUTED = "#52514e"


def error(pose):
    # (position error in m, heading error in rad) compared to the truth
    position_error = np.hypot(pose[0] - TRUE_POSE[0], pose[1] - TRUE_POSE[1])
    heading_error = abs(wrap_angle(pose[2] - TRUE_POSE[2]))
    return position_error, heading_error


def describe(name, pose):
    position_error, heading_error = error(pose)
    print(f"{name:9s} ({pose[0]:6.3f}, {pose[1]:6.3f}, {np.degrees(pose[2]):6.2f} deg)   "
          f"error {position_error:.3f} m, {np.degrees(heading_error):.2f} deg")


def draw_robot(ax, pose, color, label):
    # circle = body, line = which way it faces
    x, y, theta = pose
    ax.add_patch(plt.Circle((x, y), 0.3, fill=False, color=color, lw=2, label=label))
    ax.plot([x, x + 0.6 * np.cos(theta)], [y, y + 0.6 * np.sin(theta)], color=color, lw=2)


def draw_sensors(ax, pose, readings):
    # each sensor's spot on the robot, its view cone, and lines to what it saw
    T_world_robot = make_transform(pose[0], pose[1], pose[2])
    sensor_spots = []
    for i, sensor in enumerate(SENSORS):
        sx, sy = transform_point(T_world_robot @ sensor.T_robot_sensor(), [0.0, 0.0])
        sensor_spots.append((sx, sy))
        facing = np.degrees(pose[2] + sensor.theta)
        half_fov = np.degrees(sensor.fov / 2)
        ax.add_patch(Wedge((sx, sy), sensor.max_range, facing - half_fov, facing + half_fov,
                           color=SENSOR_COLOR, alpha=0.08, lw=0))
        ax.plot(sx, sy, "s", color=SENSOR_COLOR, ms=8, label="sensors" if i == 0 else None)
        ax.annotate(sensor.name, (sx, sy), textcoords="offset points", xytext=(8, -14),
                    color=MUTED, fontsize=9)
    for i, j, _, _ in readings:
        sx, sy = sensor_spots[i]
        ax.plot([sx, LANDMARKS[j][0]], [sy, LANDMARKS[j][1]], color=SENSOR_COLOR, lw=1, ls="--")


def draw_uncertainty(ax, pose, uncertainty, color):
    # 95% region for the position: an ellipse stretched along the directions
    # where (x, y) is most and least unsure (eigenvectors of the 2x2 covariance)
    variances, directions = np.linalg.eigh(uncertainty[:2, :2])   # smallest first
    angle = np.degrees(np.arctan2(directions[1, 1], directions[0, 1]))
    scale = 2.448   # 95% of 2D normal samples fall within 2.448 std devs
    width = 2 * scale * np.sqrt(variances[1])
    height = 2 * scale * np.sqrt(variances[0])
    ax.add_patch(Ellipse(pose[:2], width, height, angle=angle, fill=False, color=color,
                         lw=1.5, ls=":", label="estimate 95% region"))


def compare_sensors(n_runs=200):
    # average error over many noise seeds: front sensor alone vs every sensor
    front_only, every_sensor = [], []
    for seed in range(n_runs):
        np.random.seed(seed)
        readings = take_readings(TRUE_POSE)
        front_readings = [r for r in readings if r[0] == 0]
        guess = TRUE_POSE + GUESS_ERROR
        estimate_front, _ = correct(guess, START_UNCERTAINTY, front_readings)
        estimate_all, _ = correct(guess, START_UNCERTAINTY, readings)
        front_only.append(error(estimate_front))
        every_sensor.append(error(estimate_all))
    return np.mean(front_only, axis=0), np.mean(every_sensor, axis=0)


if __name__ == "__main__":
    np.random.seed(0)
    guess = TRUE_POSE + GUESS_ERROR
    readings = take_readings(TRUE_POSE)
    estimate, uncertainty = correct(guess, START_UNCERTAINTY, readings)

    print(f"readings used: {len(readings)}")
    describe("truth", TRUE_POSE)
    describe("guess", guess)
    describe("estimate", estimate)

    front, every = compare_sensors()
    print("\naverage over 200 runs:")
    print(f"  front sensor only: {front[0]:.3f} m, {np.degrees(front[1]):.2f} deg")
    print(f"  all {len(SENSORS)} sensors:     {every[0]:.3f} m, {np.degrees(every[1]):.2f} deg")

    # left: whole map, right: zoomed in on the robot
    fig, (ax_map, ax_zoom) = plt.subplots(1, 2, figsize=(14, 6.5),
                                          gridspec_kw={"width_ratios": [1.5, 1]})
    for ax in (ax_map, ax_zoom):
        ax.plot(LANDMARKS[:, 0], LANDMARKS[:, 1], "^", color=MUTED, ms=9, label="landmarks")
        for j, (lx, ly) in enumerate(LANDMARKS):
            ax.annotate(str(j), (lx, ly), textcoords="offset points", xytext=(6, 4),
                        color=MUTED, fontsize=9)
        draw_sensors(ax, TRUE_POSE, readings)
        draw_robot(ax, TRUE_POSE, TRUE_COLOR, "true pose")
        draw_robot(ax, guess, GUESS_COLOR, "guess")
        draw_robot(ax, estimate, ESTIMATE_COLOR, "EKF estimate")
        draw_uncertainty(ax, estimate, uncertainty, ESTIMATE_COLOR)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)", color=MUTED)
        ax.set_ylabel("y (m)", color=MUTED)
        ax.grid(color="#e5e4e0", lw=0.8)
        ax.set_axisbelow(True)

    ax_map.set_xlim(-10, 10)
    ax_map.set_ylim(-7, 7)
    ax_map.set_title("Whole map", color=TRUE_COLOR)
    ax_zoom.set_xlim(TRUE_POSE[0] - 1.5, TRUE_POSE[0] + 1.5)
    ax_zoom.set_ylim(TRUE_POSE[1] - 1.5, TRUE_POSE[1] + 1.5)
    ax_zoom.set_title("Zoomed in on the robot", color=TRUE_COLOR)
    ax_zoom.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.suptitle("One snapshot: EKF correct step", color=TRUE_COLOR)
    fig.tight_layout()
    fig.savefig("result.png", dpi=150)
    plt.show()
