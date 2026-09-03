"""Characterising the probe motion, independently of any tracking pipeline.

Everything here is computed from the Vicon marker-cluster trajectory inside the
shared evaluation window, so none of it depends on which pipeline is being
evaluated. These are the quantities in the manuscript's motion-characteristics
table.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from .conditioning import shah_margin, shah_singular_values

__all__ = ["MotionCharacteristics", "characterise", "angular_extent_deg"]

#: Savitzky-Golay differentiation: third order over a 0.10 s window.
#: The linear speeds are stable to under 1 % for windows from 0.05 to 0.40 s.
#: Accelerations shift 20-30 % over the same range, with the ordering between
#: recordings preserved, which is why the paper quotes them for comparison
#: rather than as absolute values. Jerk swings by an order of magnitude and is
#: not reported at all.
SAVGOL_WINDOW_S = 0.10
SAVGOL_ORDER = 3


@dataclass(frozen=True)
class MotionCharacteristics:
    """One recording's motion, as the paper reports it."""

    recording: str
    n_samples: int
    duration_s: float
    angular_extent_deg: float
    omega_median_deg_s: float
    omega_p95_deg_s: float
    speed_median_mm_s: float
    speed_p95_mm_s: float
    accel_median_mm_s2: float
    accel_p95_mm_s2: float
    path_length_m: float
    characteristic_frequency_hz: float
    shah_margin: float
    singular_values: tuple[float, float, float]
    savgol_window_samples: int

    def as_dict(self) -> dict:
        return asdict(self)


def angular_extent_deg(rotation: Rotation, *, block: int = 1500) -> float:
    """The largest geodesic angle between **any two** orientations.

    The diameter of the orientation set, computed from quaternion inner
    products: ``|q_i . q_j| = cos(theta/2)``. Blocked so the pairwise matrix is
    never materialised in full.

    Deliberately not the deviation from the first pose. That measure understates
    the extent and depends on where the window happens to start; it supported a
    claim about comparable angular ranges that was withdrawn.
    """
    q = rotation.as_quat()
    worst = 0.0
    for i in range(0, len(q), block):
        dot = np.abs(q[i:i + block] @ q.T)
        worst = max(worst, float(np.degrees(2 * np.arccos(np.clip(dot, 0.0, 1.0))).max()))
    return worst


def characterise(
    recording: str,
    time_ms: np.ndarray,
    translation: np.ndarray,
    rotvec: np.ndarray,
    *,
    sample_rate_hz: float,
) -> MotionCharacteristics:
    """Characterise one recording's motion within an evaluation window."""
    t = np.asarray(time_ms, dtype=float) / 1000.0
    p = np.asarray(translation, dtype=float)
    rotation = Rotation.from_rotvec(np.asarray(rotvec, dtype=float))
    if not (len(t) == len(p) == len(rotation)):
        raise ValueError(f"length mismatch: {len(t)}, {len(p)}, {len(rotation)}")

    # Geodesic angular speed between successive samples.
    dphi = np.degrees((rotation[:-1].inv() * rotation[1:]).magnitude())
    dt = np.diff(t)
    omega = dphi[dt > 0] / dt[dt > 0]

    window = int(round(SAVGOL_WINDOW_S * sample_rate_hz)) | 1   # must be odd
    speed = np.linalg.norm(
        savgol_filter(p, window, SAVGOL_ORDER, deriv=1, delta=1 / sample_rate_hz, axis=0), axis=1
    )
    accel = np.linalg.norm(
        savgol_filter(p, window, SAVGOL_ORDER, deriv=2, delta=1 / sample_rate_hz, axis=0), axis=1
    )

    matrices = rotation.as_matrix()
    return MotionCharacteristics(
        recording=recording,
        n_samples=len(t),
        duration_s=float(t[-1] - t[0]),
        angular_extent_deg=angular_extent_deg(rotation),
        omega_median_deg_s=float(np.median(omega)),
        omega_p95_deg_s=float(np.percentile(omega, 95)),
        speed_median_mm_s=float(np.median(speed)),
        speed_p95_mm_s=float(np.percentile(speed, 95)),
        accel_median_mm_s2=float(np.median(accel)),
        accel_p95_mm_s2=float(np.percentile(accel, 95)),
        path_length_m=float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)) / 1000.0),
        # Ratio of median acceleration to median speed: a characteristic rate of
        # the motion, separating the constrained pivot from the free conditions.
        characteristic_frequency_hz=float(np.median(accel) / np.median(speed)),
        shah_margin=shah_margin(matrices),
        singular_values=tuple(float(x) for x in shah_singular_values(matrices)[:3]),
        # Named for what it is. The original recorded this as "window_ms", but
        # the value is the Savitzky-Golay window length in SAMPLES (25 at
        # 240 Hz, i.e. 0.104 s), not milliseconds.
        savgol_window_samples=window,
    )
