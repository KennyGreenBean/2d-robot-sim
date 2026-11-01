"""
main.py = interactive window comparing the snapshot EKF to a sensors-only estimate.
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
# returned when the readings can't locate the robot (std dev 1000 m, 1000 rad)
CANT_LOCATE = np.diag([1e6, 1e6, 1e6])

TRUE_COLOR = "#0b0b0b"
GUESS_COLOR = "#eb6834"
SENSOR_COLOR = "#8a8984"   # gray: sensors are background, not something to compare
MUTED = "#52514e"
GRID = "#e5e4e0"
# the two estimates being compared: colour + line style, so they differ even in grayscale
METHOD_COLORS = {"sensors only": "#1baf7a", "EKF": "#2a78d6"}
METHOD_STYLES = {"sensors only": "--", "EKF": "-"}


def error(pose, truth):
    # (position error in m, heading error in rad)
    position_error = np.hypot(pose[0] - truth[0], pose[1] - truth[1])
    heading_error = abs(wrap_angle(pose[2] - truth[2]))
    return position_error, heading_error


def pose_from_readings(readings):
    # direct mapping, readings -> pose (the guess is never used)
    # each reading says where its landmark sits in the robot frame
    seen = np.array([transform_point(SENSORS[i].T_robot_sensor(), [d * np.cos(a), d * np.sin(a)])
                     for i, _, d, a in readings])
    on_map = LANDMARKS[[j for _, j, _, _ in readings]]
    # heading = turn that lines up the landmark layout the robot sees with the map
    # (each landmark measured from the middle of the group, then summed)
    seen_c = seen - seen.mean(axis=0)
    map_c = on_map - on_map.mean(axis=0)
    theta = np.arctan2(np.sum(seen_c[:, 0] * map_c[:, 1] - seen_c[:, 1] * map_c[:, 0]),
                       np.sum(seen_c * map_c))
    # position = middle of the group on the map, minus where the robot saw it (turned by theta)
    x, y = on_map.mean(axis=0) - make_transform(0, 0, theta)[:2, :2] @ seen.mean(axis=0)
    return np.array([x, y, theta])


def locate_without_guess(readings, step=1e-6):
    # NOT an EKF: pose straight from the readings, guess ignored
    # needs distance + angle, and at least 2 different landmarks (1 can't fix heading)
    readings = [r for r in readings if r[2] is not None and r[3] is not None]
    if len({j for _, j, _, _ in readings}) < 2:
        return np.zeros(3), CANT_LOCATE
    pose = pose_from_readings(readings)
    # uncertainty: nudge each measured number a little, see how far the pose moves,
    # then scale by that sensor's noise
    slopes, noise_std = [], []
    for k, (i, _, _, _) in enumerate(readings):
        for which, std in ((2, SENSORS[i].range_noise), (3, SENSORS[i].bearing_noise)):
            nudged = [list(r) for r in readings]
            nudged[k][which] += step
            change = pose_from_readings(nudged) - pose
            change[2] = wrap_angle(change[2])   
            slopes.append(change / step)
            noise_std.append(std)
    slopes = np.array(slopes).T                     # 3 x (number of measured numbers)
    uncertainty = slopes @ np.diag(np.array(noise_std)**2) @ slopes.T
    return pose, uncertainty


def pinned_down(uncertainty):
    # did the readings actually locate the robot? (std dev under 5 m and 45 deg)
    std = np.sqrt(np.diag(uncertainty))
    return std[0] < 5 and std[1] < 5 and std[2] < np.radians(45)


def run_once(truth, guess_error, enabled, seed):
    # simulate readings, keep the enabled sensors' ones, then estimate the pose two ways
    # from the SAME readings:
    #   sensors only -> readings alone, guess ignored
    #   EKF          -> guess + readings, each weighted by how much it's trusted
    np.random.seed(seed)
    readings = [r for r in take_readings(truth) if enabled[r[0]]]
    guess = truth + guess_error
    guess[2] = wrap_angle(guess[2])
    estimates = {
        "sensors only": locate_without_guess(readings),
        "EKF": correct(guess, START_UNCERTAINTY, readings),
    }   # each is (estimate, uncertainty)
    return readings, guess, estimates


def run_trials(truth, guess_error, enabled, runs=200):
    # average error over many noise draws (only runs where the robot was located)
    errors = {name: [] for name in METHOD_COLORS}
    for seed in range(runs):
        _, _, estimates = run_once(truth, guess_error, enabled, seed)
        for name, (estimate, uncertainty) in estimates.items():
            if pinned_down(uncertainty):
                errors[name].append(error(estimate, truth))
    text = f"{runs} trials (this setup):\n"
    for name, errs in errors.items():
        text += f"  {name}\n"
        if errs:
            errs = np.array(errs)
            text += (f"    mean  {errs[:, 0].mean():.3f} m, {np.degrees(errs[:, 1].mean()):.2f} deg\n"
                     f"    worst {errs[:, 0].max():.3f} m, {np.degrees(errs[:, 1].max()):.2f} deg\n")
        if len(errs) < runs:
            text += f"    couldn't locate in {runs - len(errs)} runs\n"
    return text


def draw_robot(ax, pose, color, label, ls="-"):
    # circle = body, line = which way it faces
    x, y, theta = pose
    ax.add_patch(plt.Circle((x, y), 0.3, fill=False, color=color, lw=2, ls=ls, label=label))
    ax.plot([x, x + 0.6 * np.cos(theta)], [y, y + 0.6 * np.sin(theta)], color=color, lw=2, ls=ls)


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
                           color=SENSOR_COLOR, alpha=0.10, lw=0))
        ax.plot(sx, sy, "s", color=SENSOR_COLOR, ms=7, label=None if labelled else "sensors")
        labelled = True
        ax.annotate(sensor.name, (sx, sy), textcoords="offset points", xytext=(8, -14),
                    color=MUTED, fontsize=8)
    for i, j, _, _ in readings:
        sx, sy = sensor_spots[i]
        ax.plot([sx, LANDMARKS[j][0]], [sy, LANDMARKS[j][1]], color=SENSOR_COLOR, lw=1, ls=":")


def draw_uncertainty(ax, pose, uncertainty, color, label):
    # 95% region for the position: an ellipse stretched along the directions
    # where (x, y) is most and least unsure (eigenvectors of the 2x2 covariance)
    variances, directions = np.linalg.eigh(uncertainty[:2, :2])   # smallest first
    angle = np.degrees(np.arctan2(directions[1, 1], directions[0, 1]))
    scale = 2.448   # 95% of 2D normal samples fall within 2.448 std devs
    ax.add_patch(Ellipse(pose[:2], 2 * scale * np.sqrt(variances[1]),
                         2 * scale * np.sqrt(variances[0]), angle=angle, fill=False,
                         color=color, lw=1.5, ls=":", label=label))


def draw_scene(ax, truth, guess, estimates, readings, enabled):
    ax.plot(LANDMARKS[:, 0], LANDMARKS[:, 1], "^", color=MUTED, ms=9, label="landmarks")
    for j, (lx, ly) in enumerate(LANDMARKS):
        ax.annotate(str(j), (lx, ly), textcoords="offset points", xytext=(6, 4),
                    color=MUTED, fontsize=8)
    draw_sensors(ax, truth, readings, enabled)
    draw_robot(ax, truth, TRUE_COLOR, "true pose")
    draw_robot(ax, guess, GUESS_COLOR, "guess")
    for name, (estimate, uncertainty) in estimates.items():
        if pinned_down(uncertainty):
            draw_robot(ax, estimate, METHOD_COLORS[name], f"{name} estimate", METHOD_STYLES[name])
            draw_uncertainty(ax, estimate, uncertainty, METHOD_COLORS[name], None)
    # one gray legend entry for both ellipses (draws nothing on the map)
    ax.plot([], [], color=MUTED, lw=1.5, ls=":", label="95% region (dotted)")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", color=MUTED)
    ax.set_ylabel("y (m)", color=MUTED)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def build_gui():
    fig = plt.figure(figsize=(15, 8.5))
    fig.suptitle("Snapshot EKF vs sensors only: click the map to move the robot",
                 color=TRUE_COLOR)
    ax_map = fig.add_axes([0.04, 0.33, 0.44, 0.58])
    ax_zoom = fig.add_axes([0.53, 0.33, 0.24, 0.58])
    ax_info = fig.add_axes([0.80, 0.33, 0.19, 0.58])

    # controls along the bottom
    # column 1: robot and guess
    heading = Slider(fig.add_axes([0.15, 0.22, 0.22, 0.03]), "true heading (deg)",
                     -180, 180, valinit=np.degrees(TRUE_POSE[2]), valstep=1)
    guess_dx = Slider(fig.add_axes([0.15, 0.16, 0.22, 0.03]), "guess error x (m)",
                      -1.5, 1.5, valinit=0.5, valstep=0.05)
    guess_dy = Slider(fig.add_axes([0.15, 0.10, 0.22, 0.03]), "guess error y (m)",
                      -1.5, 1.5, valinit=-0.3, valstep=0.05)
    guess_dth = Slider(fig.add_axes([0.15, 0.04, 0.22, 0.03]), "guess error heading (deg)",
                       -45, 45, valinit=10, valstep=1)
    # column 2: sensor noise (std dev), applied to every sensor, and the buttons
    start_range_noise = next((s.range_noise for s in SENSORS if s.range_noise is not None), 0.1)
    start_bearing_noise = next((s.bearing_noise for s in SENSORS if s.bearing_noise is not None),
                               np.radians(2))
    range_noise = Slider(fig.add_axes([0.55, 0.22, 0.17, 0.03]), "distance noise (m)",
                         0.01, 1.0, valinit=start_range_noise, valstep=0.01)
    bearing_noise = Slider(fig.add_axes([0.55, 0.16, 0.17, 0.03]), "angle noise (deg)",
                           0.1, 20, valinit=np.degrees(start_bearing_noise), valstep=0.1)
    noise_button = Button(fig.add_axes([0.45, 0.04, 0.12, 0.06]), "New noise")
    trials_button = Button(fig.add_axes([0.60, 0.04, 0.12, 0.06]), "Run 200 trials")
    # column 3: which sensors are on
    fig.text(0.80, 0.235, "sensors on", color=MUTED)
    sensor_boxes = CheckButtons(fig.add_axes([0.80, 0.10, 0.15, 0.13]),
                                [s.name for s in SENSORS], [True] * len(SENSORS))

    state = {"x": TRUE_POSE[0], "y": TRUE_POSE[1], "seed": 0, "trials": ""}

    def current_setup():
        # copy the noise sliders onto every sensor (both the simulation and the EKF read these)
        for sensor in SENSORS:
            if sensor.range_noise is not None:
                sensor.range_noise = range_noise.val
            if sensor.bearing_noise is not None:
                sensor.bearing_noise = np.radians(bearing_noise.val)
        truth = np.array([state["x"], state["y"], np.radians(heading.val)])
        guess_error = np.array([guess_dx.val, guess_dy.val, np.radians(guess_dth.val)])
        enabled = sensor_boxes.get_status()
        return truth, guess_error, enabled

    def redraw(keep_trials=False):
        if not keep_trials:
            state["trials"] = ""
        truth, guess_error, enabled = current_setup()
        readings, guess, estimates = run_once(truth, guess_error, enabled, state["seed"])

        for ax in (ax_map, ax_zoom):
            ax.clear()
            draw_scene(ax, truth, guess, estimates, readings, enabled)
        ax_map.set_xlim(-10, 10)
        ax_map.set_ylim(-7, 7)
        ax_map.set_title("Whole map", color=TRUE_COLOR)
        ax_zoom.set_xlim(truth[0] - 2, truth[0] + 2)
        ax_zoom.set_ylim(truth[1] - 2, truth[1] + 2)
        ax_zoom.set_title("Zoomed in on the robot", color=TRUE_COLOR)

        info = f"readings used: {len(readings)}   (noise seed {state['seed']})\n\n"
        info += "error          position   heading\n"
        guess_pos, guess_head = error(guess, truth)
        info += f"guess          {guess_pos:6.3f} m  {np.degrees(guess_head):6.2f} deg\n"
        spread = "\nestimate std dev  x / y (m)  heading\n"
        for name, (estimate, uncertainty) in estimates.items():
            if pinned_down(uncertainty):
                est_pos, est_head = error(estimate, truth)
                std = np.sqrt(np.diag(uncertainty))
                info += f"{name:<14} {est_pos:6.3f} m  {np.degrees(est_head):6.2f} deg\n"
                spread += f"{name:<14} {std[0]:.2f} / {std[1]:.2f}  {np.degrees(std[2]):5.2f} deg\n"
            else:
                info += f"{name:<14} can't locate\n"
        info += spread + "\n" + state["trials"]
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

    for slider in (heading, guess_dx, guess_dy, guess_dth, range_noise, bearing_noise):
        slider.on_changed(lambda _: redraw())
    sensor_boxes.on_clicked(lambda _: redraw())
    noise_button.on_clicked(new_noise)
    trials_button.on_clicked(trials)
    fig.canvas.mpl_connect("button_press_event", on_click)

    redraw()
    # the widgets must stay referenced, or matplotlib stops responding to them
    widgets = {"heading": heading, "guess_dx": guess_dx, "guess_dy": guess_dy,
               "guess_dth": guess_dth, "range_noise": range_noise,
               "bearing_noise": bearing_noise, "sensors": sensor_boxes,
               "noise": noise_button, "trials": trials_button}
    return fig, widgets


if __name__ == "__main__":
    fig, widgets = build_gui()
    plt.show()
