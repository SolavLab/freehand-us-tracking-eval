"""Angular kinematics of a pose trajectory.

The synchronization stages operate on the *geodesic* angular speed: the rotation
angle between successive orientations, divided by the interval. That quantity is
invariant to the coordinate frames the two trajectories are expressed in, which
is essential because those frames are unrelated until the spatial calibration
has been estimated.

It is not the same as the derivative of the rotation-vector magnitude. That
measures how fast the orientation moves away from the arbitrary reference the
rotation vector is anchored to, not how fast the body turns.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = ["rotational_increments", "angular_speed"]


def rotational_increments(rotvec: np.ndarray) -> np.ndarray:
    """Relative rotation between successive samples, as ``(n-1, 3)`` rotvecs."""
    rotvec = np.asarray(rotvec, dtype=float)
    if rotvec.ndim != 2 or rotvec.shape[1] != 3:
        raise ValueError(f"expected (n, 3) rotation vectors, got {rotvec.shape}")
    absolute = Rotation.from_rotvec(rotvec)
    return (absolute[1:] * absolute[:-1].inv()).as_rotvec()


def angular_speed(time_ms: np.ndarray, rotvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Geodesic angular speed, assigned to sample midpoints.

    Returns ``(t_mid_ms, omega)`` of length ``n-1``.

    **Units are radians per millisecond**, because ``time_ms`` is in
    milliseconds. The original implementation's comment claimed radians per
    second; it did not divide by 1000. Nothing downstream was wrong -- the
    objective only ever compares two series computed the same way -- but the
    figure is a thousand times smaller than "rad/s" suggests, which has misled
    readers of that code before.

    Samples with a NaN orientation are dropped before differencing, so a gap in
    the reference does not poison the neighbouring increments.
    """
    time_ms = np.asarray(time_ms, dtype=float)
    rotvec = np.asarray(rotvec, dtype=float)
    if len(time_ms) != len(rotvec):
        raise ValueError(f"{len(time_ms)} timestamps but {len(rotvec)} orientations")

    keep = ~np.isnan(rotvec).any(axis=1)
    t, rv = time_ms[keep], rotvec[keep]
    if len(t) < 2:
        raise ValueError("need at least two valid orientations to differentiate")

    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("timestamps are not strictly increasing")

    angle = np.linalg.norm(rotational_increments(rv), axis=1)
    return 0.5 * (t[1:] + t[:-1]), angle / dt
