"""
main.py = interactive window for the snapshot EKF.

Run with:  python3 main.py
  - click the map to move the robot
  - sliders set its heading and how wrong the first guess is
  - tick boxes turn sensors on/off
  - "New noise" draws fresh sensor noise, "Run 200 trials" averages the error
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Wedge
from matplotlib.widgets import Button, CheckButtons, Slider

from frames import make_transform, transform_point, wrap_angle
from sensors import SENSORS
from simulate import LANDMARKS, TRUE_POSE, take_readings
from ekf import correct

# how unsure we are about the first guess (std dev 0.5 m, 0.5 m, 10 deg)
START_UNCERTAINTY = np.diag([0.5**2, 0.5**2, np.radians(10)**2])

TRUE_COLOR = "#0b0b0b"
GUESS_COLOR = "#eb6834"
ESTIMATE_COLOR = "#2a78d6"
SENSOR_COLOR = "#1baf7a"
MUTED = "#52514e"
GRID = "#e5e4e0"


def error(pose, truth):
    # (position error in m, heading error in rad)
    position_error = np.hypot(pose[0] - truth[0], pose[1] - truth[1])
    heading_error = abs(wrap_angle(pose[2] - truth[2]))
    return position_error, heading_error


def run_once(truth, guess_error, enabled, seed):
    # simulate readings, keep the enabled sensors' ones, correct the guess
    np.random.seed(seed)
    readings = [r for r in take_readings(truth) if enabled[r[0]]]
    guess = truth + guess_error
    guess[2] = wrap_angle(guess[2])
    estimate, uncertainty = correct(guess, START_UNCERTAINTY, readings)
    return readings, guess, estimate, uncertainty


def run_trials(truth, guess_error, enabled, runs=200):
    # average error over many noise draws
    errors = []
    for seed in range(runs):
        _, _, estimate, _ = run_once(truth, guess_error, enabled, seed)
        errors.append(error(estimate, truth))
    errors = np.array(errors)
    return (f"{runs} trials (this setup):\n"
            f"  mean  {errors[:, 0].mean():.3f} m, {np.degrees(errors[:, 1].mean()):.2f} deg\n"
            f"  worst {errors[:, 0].max():.3f} m, {np.degrees(errors[:, 1].max()):.2f} deg")


def draw_robot(ax, pose, color, label):
    # circle = body, line = which way it faces
    x, y, theta = pose
    ax.add_patch(plt.Circle((x, y), 0.3, fill=False, color=color, lw=2, label=label))
    ax.plot([x, x + 0.6 * np.cos(theta)], [y, y + 0.6 * np.sin(theta)], color=color, lw=2)


def draw_sensors(ax, pose, readings, enabled):
    # each enabled sensor's spot on the robot, its view cone, and lines to what it saw
    T_world_robot = make_transform(pose[0], pose[1], pose[2])
    sensor_spots = {}
    labelled = False
    for i, sensor in enumerate(SENSORS):
        if not enabled[i]:
            continue
        sx, sy = transform_point(T_world_robot @ sensor.T_robot_sensor(), [0.0, 0.0])
        sensor_spots[i] = (sx, sy)
        facing = np.degrees(pose[2] + sensor.theta)
        half_fov = np.degrees(sensor.fov / 2)
        ax.add_patch(Wedge((sx, sy), sensor.max_range, facing - half_fov, facing + half_fov,
                           color=SENSOR_COLOR, alpha=0.08, lw=0))
        ax.plot(sx, sy, "s", color=SENSOR_COLOR, ms=7, label=None if labelled else "sensors")
        labelled = True
        ax.annotate(sensor.name, (sx, sy), textcoords="offset points", xytext=(8, -14),
                    color=MUTED, fontsize=8)
    for i, j, _, _ in readings:
        sx, sy = sensor_spots[i]
        ax.plot([sx, LANDMARKS[j][0]], [sy, LANDMARKS[j][1]], color=SENSOR_COLOR, lw=1, ls="--")


def draw_uncertainty(ax, pose, uncertainty, color):
    # 95% region for the position: an ellipse stretched along the directions
    # where (x, y) is most and least unsure (eigenvectors of the 2x2 covariance)
    variances, directions = np.linalg.eigh(uncertainty[:2, :2])   # smallest first
    angle = np.degrees(np.arctan2(directions[1, 1], directions[0, 1]))
    scale = 2.448   # 95% of 2D normal samples fall within 2.448 std devs
    ax.add_patch(Ellipse(pose[:2], 2 * scale * np.sqrt(variances[1]),
                         2 * scale * np.sqrt(variances[0]), angle=angle, fill=False,
                         color=color, lw=1.5, ls=":", label="estimate 95% region"))


def draw_scene(ax, truth, guess, estimate, uncertainty, readings, enabled):
    ax.plot(LANDMARKS[:, 0], LANDMARKS[:, 1], "^", color=MUTED, ms=9, label="landmarks")
    for j, (lx, ly) in enumerate(LANDMARKS):
        ax.annotate(str(j), (lx, ly), textcoords="offset points", xytext=(6, 4),
                    color=MUTED, fontsize=8)
    draw_sensors(ax, truth, readings, enabled)
    draw_robot(ax, truth, TRUE_COLOR, "true pose")
    draw_robot(ax, guess, GUESS_COLOR, "guess")
    draw_robot(ax, estimate, ESTIMATE_COLOR, "EKF estimate")
    draw_uncertainty(ax, estimate, uncertainty, ESTIMATE_COLOR)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", color=MUTED)
    ax.set_ylabel("y (m)", color=MUTED)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def build_gui():
    fig = plt.figure(figsize=(15, 8.5))
    fig.suptitle("Snapshot EKF: click the map to move the robot", color=TRUE_COLOR)
    ax_map = fig.add_axes([0.04, 0.33, 0.44, 0.58])
    ax_zoom = fig.add_axes([0.53, 0.33, 0.24, 0.58])
    ax_info = fig.add_axes([0.80, 0.33, 0.19, 0.58])

    # controls along the bottom
    heading = Slider(fig.add_axes([0.17, 0.22, 0.27, 0.03]), "true heading (deg)",
                     -180, 180, valinit=np.degrees(TRUE_POSE[2]), valstep=1)
    guess_dx = Slider(fig.add_axes([0.17, 0.16, 0.27, 0.03]), "guess error x (m)",
                      -1.5, 1.5, valinit=0.5, valstep=0.05)
    guess_dy = Slider(fig.add_axes([0.17, 0.10, 0.27, 0.03]), "guess error y (m)",
                      -1.5, 1.5, valinit=-0.3, valstep=0.05)
    guess_dth = Slider(fig.add_axes([0.17, 0.04, 0.27, 0.03]), "guess error heading (deg)",
                       -45, 45, valinit=10, valstep=1)
    fig.text(0.53, 0.25, "sensors on", color=MUTED)
    sensor_boxes = CheckButtons(fig.add_axes([0.53, 0.04, 0.12, 0.20]),
                                [s.name for s in SENSORS], [True] * len(SENSORS))
    noise_button = Button(fig.add_axes([0.69, 0.15, 0.12, 0.06]), "New noise")
    trials_button = Button(fig.add_axes([0.69, 0.06, 0.12, 0.06]), "Run 200 trials")

    state = {"x": TRUE_POSE[0], "y": TRUE_POSE[1], "seed": 0, "trials": ""}

    def current_setup():
        truth = np.array([state["x"], state["y"], np.radians(heading.val)])
        guess_error = np.array([guess_dx.val, guess_dy.val, np.radians(guess_dth.val)])
        enabled = sensor_boxes.get_status()
        return truth, guess_error, enabled

    def redraw(keep_trials=False):
        if not keep_trials:
            state["trials"] = ""
        truth, guess_error, enabled = current_setup()
        readings, guess, estimate, uncertainty = run_once(truth, guess_error, enabled, state["seed"])

        for ax in (ax_map, ax_zoom):
            ax.clear()
            draw_scene(ax, truth, guess, estimate, uncertainty, readings, enabled)
        ax_map.set_xlim(-10, 10)
        ax_map.set_ylim(-7, 7)
        ax_map.set_title("Whole map", color=TRUE_COLOR)
        ax_zoom.set_xlim(truth[0] - 2, truth[0] + 2)
        ax_zoom.set_ylim(truth[1] - 2, truth[1] + 2)
        ax_zoom.set_title("Zoomed in on the robot", color=TRUE_COLOR)

        guess_pos, guess_head = error(guess, truth)
        est_pos, est_head = error(estimate, truth)
        std = np.sqrt(np.diag(uncertainty))
        info = (f"readings used: {len(readings)}   (noise seed {state['seed']})\n\n"
                f"error      position   heading\n"
                f"guess      {guess_pos:6.3f} m  {np.degrees(guess_head):6.2f} deg\n"
                f"estimate   {est_pos:6.3f} m  {np.degrees(est_head):6.2f} deg\n\n"
                f"estimate std dev:\n"
                f"  x {std[0]:.3f} m, y {std[1]:.3f} m\n"
                f"  heading {np.degrees(std[2]):.2f} deg\n\n"
                f"{state['trials']}")
        ax_info.clear()
        ax_info.axis("off")
        ax_info.text(0, 1, info, va="top", family="monospace", fontsize=9, color=TRUE_COLOR)
        handles, labels = ax_zoom.get_legend_handles_labels()
        ax_info.legend(handles, labels, loc="lower left", frameon=False, fontsize=9)
        fig.canvas.draw_idle()

    def on_click(event):
        # left-click on the map moves the robot there
        if event.inaxes is ax_map and event.button == 1:
            state["x"], state["y"] = event.xdata, event.ydata
            redraw()

    def new_noise(_):
        state["seed"] += 1
        redraw()

    def trials(_):
        state["trials"] = run_trials(*current_setup())
        redraw(keep_trials=True)

    for slider in (heading, guess_dx, guess_dy, guess_dth):
        slider.on_changed(lambda _: redraw())
    sensor_boxes.on_clicked(lambda _: redraw())
    noise_button.on_clicked(new_noise)
    trials_button.on_clicked(trials)
    fig.canvas.mpl_connect("button_press_event", on_click)

    redraw()
    # the widgets must stay referenced, or matplotlib stops responding to them
    widgets = {"heading": heading, "guess_dx": guess_dx, "guess_dy": guess_dy,
               "guess_dth": guess_dth, "sensors": sensor_boxes,
               "noise": noise_button, "trials": trials_button}
    return fig, widgets


if __name__ == "__main__":
    fig, widgets = build_gui()
    plt.show()
