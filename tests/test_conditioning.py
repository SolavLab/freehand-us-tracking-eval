"""Calibration well-posedness, and the OpenCV convention it depends on.

These were standalone demonstration scripts in the manuscript repository, each
checking one property of the Shah margin against a synthetic motion with a known
answer. They are tests here: the properties are load-bearing -- the paper uses
the margin to argue every recording was well determined -- and a property that
is only checked when someone remembers to run a script is not checked.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from vipose.conditioning import shah_margin, shah_singular_values  # noqa: E402
from vipose.motion import angular_extent_deg  # noqa: E402

RNG = np.random.default_rng(1)


def _about_axis(axis, angles):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    return Rotation.from_rotvec(np.outer(angles, axis))


def test_single_axis_motion_is_degenerate():
    """Rotation about one axis gives margin 0: the calibration is not identifiable.

    The algebraic counterpart of the familiar requirement that the motion rotate
    about more than one axis. If every rotation shares an axis n, then n n^T is
    invariant under conjugation by all of them and supplies a second direction on
    which every term of K agrees.
    """
    r = _about_axis([0, 0, 1], RNG.uniform(-np.pi, np.pi, 400))
    assert shah_margin(r.as_matrix()) == pytest.approx(0.0, abs=1e-9)


def test_two_axis_and_isotropic_margins():
    """The reference points quoted in the paper's appendix: 0.0234 and 0.3270.

    Constructed exactly as the original validation did -- 400 samples sweeping
    0 to 120 degrees, half about x and then half about y -- because the margin
    is a property of the particular motion, not of "two axes" in the abstract.
    """
    n = 400
    theta = np.linspace(0, np.radians(120), n)

    two = np.zeros((n, 3))
    two[: n // 2, 0] = theta[: n // 2]
    two[n // 2:, 1] = theta[: n // 2]
    assert shah_margin(Rotation.from_rotvec(two).as_matrix()) == pytest.approx(0.0234, abs=1e-3)

    rng = np.random.default_rng(1)
    axes = rng.normal(size=(n, 3))
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    isotropic = Rotation.from_rotvec(axes * rng.uniform(0, 2, n)[:, None])
    assert shah_margin(isotropic.as_matrix()) == pytest.approx(0.327, abs=3e-2)


def test_sigma_1_is_exactly_one():
    """A theorem about K, not an approximation.

    Each term is a Kronecker product of two rotations and so is orthogonal; an
    average of length-preserving maps attains gain 1 exactly along the direction
    where every term agrees, which is the calibration itself.
    """
    for r in (Rotation.random(300, random_state=7), _about_axis([1, 1, 0], RNG.normal(size=300))):
        assert shah_singular_values(r.as_matrix())[0] == pytest.approx(1.0, abs=1e-9)


def test_margin_is_blind_to_the_slam_side():
    """The margin depends only on the reference trajectory.

    Substituting the noiseless relation leaves K = (I x R_Z) G (I x R_X^T) with
    G built from the reference alone; the outer factors are orthogonal and do not
    change singular values. This is why one margin is quoted per recording rather
    than one per pipeline, and it is worth testing because the claim is what
    licenses that.
    """
    reference = Rotation.random(500, random_state=11)
    margin = shah_margin(reference.as_matrix())
    for seed in (1, 2, 3):
        offset = Rotation.random(random_state=seed)
        assert shah_margin((offset * reference).as_matrix()) == pytest.approx(margin, abs=1e-9)


def test_non_rotations_are_rejected():
    """sigma_1 != 1 means the input is not a rotation stack, and must not pass."""
    # Uniformly scaled rotations: still orthogonal in direction, but the
    # scaling carries straight through to sigma_1, so the theorem fails.
    bad = 0.9 * Rotation.random(50, random_state=13).as_matrix()
    with pytest.raises(ValueError, match="sigma_1"):
        shah_margin(bad)


def test_angular_extent_is_the_diameter_not_the_deviation_from_the_first_pose():
    """Two clusters 90 deg apart, arrived at from a starting pose between them.

    Measured as a diameter the extent is 90 deg. Measured against the first pose
    -- which is what the withdrawn "angular span" did -- it reads about half
    that, and the answer moves with where the window happens to start.
    """
    a = _about_axis([0, 0, 1], np.full(50, np.radians(-45)))
    b = _about_axis([0, 0, 1], np.full(50, np.radians(+45)))
    start = _about_axis([0, 0, 1], [0.0])
    combined = Rotation.concatenate([start, a, b])

    assert angular_extent_deg(combined) == pytest.approx(90.0, abs=1e-6)

    first = combined[0]
    from_first = np.degrees((first.inv() * combined).magnitude()).max()
    assert from_first == pytest.approx(45.0, abs=1e-6)


def test_opencv_hand_eye_convention():
    """calibrateRobotWorldHandEye satisfies A Z = X B and returns (Z, X).

    Pinned empirically against a synthetic problem with a known answer, because
    assuming the other common convention (A X = Z B) yields a calibration about
    106 degrees wrong -- a failure that still produces plausible-looking output.
    """
    import cv2

    from vipose.calibration import solve_hand_eye
    from vipose.geometry.transforms import to_matrices

    rng = np.random.default_rng(0)
    n = 200
    t_marker_to_camera = np.eye(4)
    t_marker_to_camera[:3, :3] = Rotation.random(random_state=5).as_matrix()
    t_marker_to_camera[:3, 3] = [48.8, 32.9, -89.6]
    t_slamworld_to_vicon = np.eye(4)
    t_slamworld_to_vicon[:3, :3] = Rotation.random(random_state=6).as_matrix()
    t_slamworld_to_vicon[:3, 3] = [-925.8, 715.6, 764.3]

    marker_in_vicon = to_matrices(rng.uniform(-500, 500, (n, 3)),
                                  Rotation.random(n, random_state=9).as_rotvec())
    # Construct the SLAM side so the calibration is exactly known.
    camera_in_world = np.linalg.inv(t_slamworld_to_vicon) @ marker_in_vicon @ t_marker_to_camera

    got = solve_hand_eye(
        camera_in_world, marker_in_vicon, method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH
    )
    np.testing.assert_allclose(got.T_camera_to_marker, t_marker_to_camera, atol=1e-6)
    np.testing.assert_allclose(got.T_slamworld_to_vicon, t_slamworld_to_vicon, atol=1e-6)
