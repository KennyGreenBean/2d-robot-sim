"""
test.py = checks that the frames, Jacobian and EKF correct step behave sensibly.

Run with:  python3 test.py
Each test sets up a case where we know the right answer and asserts it.
"""

import numpy as np

from frames import make_transform, transform_point, inverse, wrap_angle
from sensors import SENSORS
from simulate import LANDMARKS, TRUE_POSE, expected_reading, take_readings
from ekf import measurement_jacobian, correct

# same starting setup as main.py
GUESS_ERROR = np.array([0.5, -0.3, np.radians(10)])
START_UNCERTAINTY = np.diag([0.5**2, 0.5**2, np.radians(10)**2])


def error(pose):
    # (position error in m, heading error in deg) compared to the truth
    position_error = np.hypot(pose[0] - TRUE_POSE[0], pose[1] - TRUE_POSE[1])
    heading_error = np.degrees(abs(wrap_angle(pose[2] - TRUE_POSE[2])))
    return position_error, heading_error


def test_frames_round_trip():
    # converting a point there and back gives the original point
    T_a_b = make_transform(2.0, -1.0, 0.7)
    p = np.array([0.3, -1.7])
    assert np.allclose(transform_point(inverse(T_a_b), transform_point(T_a_b, p)), p)


def test_jacobian_matches_nudging():
    # hand-derived slopes = slopes measured by nudging x, y, theta a tiny amount
    eps = 1e-6
    for sensor in SENSORS:
        for landmark in LANDMARKS:
            pose = np.array([1.0, -2.0, 0.7])
            measured = np.zeros((2, 3))
            for k in range(3):
                step = np.zeros(3)
                step[k] = eps
                d_plus, a_plus = expected_reading(pose + step, landmark, sensor)
                d_minus, a_minus = expected_reading(pose - step, landmark, sensor)
                measured[0, k] = (d_plus - d_minus) / (2 * eps)
                measured[1, k] = wrap_angle(a_plus - a_minus) / (2 * eps)
            assert np.allclose(measurement_jacobian(pose, landmark, sensor), measured, atol=1e-5)


def test_perfect_guess_does_not_move():
    # guess = truth and no noise: nothing to fix
    readings = take_readings(TRUE_POSE, noisy=False)
    estimate, _ = correct(TRUE_POSE.copy(), START_UNCERTAINTY, readings)
    position_error, heading_error = error(estimate)
    assert position_error < 1e-9 and heading_error < 1e-7


def test_no_readings_changes_nothing():
    # no information: guess and uncertainty stay the same
    guess = TRUE_POSE + GUESS_ERROR
    estimate, uncertainty = correct(guess, START_UNCERTAINTY, [])
    assert np.allclose(estimate, guess)
    assert np.allclose(uncertainty, START_UNCERTAINTY)


def test_noise_free_readings_fix_the_guess():
    # perfect readings pull a wrong guess close to the truth
    # (not exactly: the Jacobian is a straight-line approximation at the guess)
    readings = take_readings(TRUE_POSE, noisy=False)
    estimate, _ = correct(TRUE_POSE + GUESS_ERROR, START_UNCERTAINTY, readings)
    position_error, heading_error = error(estimate)
    assert position_error < 0.1 and heading_error < 1.0


def test_confident_guess_barely_moves():
    # if we fully trust the guess, the sensors should hardly change it
    np.random.seed(0)
    guess = TRUE_POSE + GUESS_ERROR
    estimate, _ = correct(guess, START_UNCERTAINTY * 1e-6, take_readings(TRUE_POSE))
    assert np.hypot(*(estimate[:2] - guess[:2])) < 0.001


def test_uncertainty_shrinks():
    # readings add information, so every std dev must get smaller
    np.random.seed(0)
    _, uncertainty = correct(TRUE_POSE + GUESS_ERROR, START_UNCERTAINTY, take_readings(TRUE_POSE))
    assert np.all(np.diag(uncertainty) < np.diag(START_UNCERTAINTY))


def test_uncertainty_is_honest():
    # over many runs the truth should land inside the 95% region about 95% of the time
    # (5.991 = the 95% cut-off for a 2D normal distribution, squared std devs)
    runs, inside = 1000, 0
    for seed in range(runs):
        np.random.seed(seed)
        guess = TRUE_POSE + np.random.multivariate_normal(np.zeros(3), START_UNCERTAINTY)
        estimate, uncertainty = correct(guess, START_UNCERTAINTY, take_readings(TRUE_POSE))
        e = estimate[:2] - TRUE_POSE[:2]
        if e @ np.linalg.inv(uncertainty[:2, :2]) @ e <= 5.991:
            inside += 1
    percent = 100 * inside / runs
    print(f"    truth inside 95% region in {percent:.1f}% of runs")
    assert 90 <= percent <= 99


if __name__ == "__main__":
    tests = [
        test_frames_round_trip,
        test_jacobian_matches_nudging,
        test_perfect_guess_does_not_move,
        test_no_readings_changes_nothing,
        test_noise_free_readings_fix_the_guess,
        test_confident_guess_barely_moves,
        test_uncertainty_shrinks,
        test_uncertainty_is_honest,
    ]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError:
            print(f"FAIL  {test.__name__}")
            failed += 1
    print(f"\n{len(tests) - failed} of {len(tests)} tests passed")
