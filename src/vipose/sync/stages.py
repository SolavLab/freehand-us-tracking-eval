"""Stages 1 and 2 of the temporal alignment.

Both operate on the geodesic angular-speed magnitude. The sign convention is
that a **positive offset means the Vicon clock runs ahead of the camera clock**,
so the reference is evaluated at ``t + offset`` when sampled onto camera times.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import fmin, fminbound
from scipy.signal import correlate

from ..kinematics import angular_speed

__all__ = [
    "crosscorrelation_offset",
    "omega_mismatch_rms",
    "refine_offset_omega",
    "STAGE2_MAXITER",
    "STAGE2_XTOL_MS",
]

# Nelder-Mead settings for stage 2, matching the published configuration.
#
# The original tree contained two different settings for this: the live unified
# path used these, and a second, unreachable copy inside the calibration
# function used maxiter=50, xtol=1e-6. Only these ran. Having one definition is
# the point -- the stage-2 objective has several near-equal minima about 12 ms
# apart, so two tolerance settings in one codebase can move a published median.
STAGE2_MAXITER = 20
STAGE2_XTOL_MS = 1e-5


def crosscorrelation_offset(
    track_time_ms: np.ndarray,
    track_rotvec: np.ndarray,
    reference_time_ms: np.ndarray,
    reference_rotvec: np.ndarray,
) -> float:
    """Stage 1: the offset in milliseconds, from the cross-correlation peak.

    Both angular-speed series are reduced to zero mean and unit variance, their
    cross-correlation is evaluated at every integer sample lag, and the peak is
    refined to sub-sample precision by cubic interpolation and bounded
    minimisation over one sample either side.

    The reference series is interpolated **linearly** onto the camera sample
    times here, which is what the published implementation does at this stage.
    """
    t_track, omega_track = angular_speed(track_time_ms, track_rotvec)
    t_ref, omega_ref = angular_speed(reference_time_ms, reference_rotvec)
    omega_ref_on_track = np.interp(t_track, t_ref, omega_ref)

    a = _standardise(omega_track)
    b = _standardise(omega_ref_on_track)

    corr = correlate(a, b, mode="full") / len(a)
    lags = np.arange(-len(a) + 1, len(a))
    peak = lags[int(np.argmax(corr))]

    spline = interp1d(lags, corr, kind="cubic", fill_value="extrapolate")
    lag_samples = float(fminbound(lambda x: -spline(x), peak - 1.0, peak + 1.0, xtol=1e-8))

    return _lag_samples_to_ms(np.asarray(track_time_ms, dtype=float), lag_samples)


def omega_mismatch_rms(
    track_time_ms: np.ndarray,
    track_rotvec: np.ndarray,
    reference_time_ms: np.ndarray,
    reference_rotvec: np.ndarray,
    offset_ms: float,
) -> float:
    """The stage-2 objective: RMS disagreement between the two speed series.

    The reference speed is resampled onto the camera midpoints by **cubic
    spline**, and the sum is restricted to camera samples lying inside the
    interval the reference spans, so no extrapolated value enters.
    """
    t_track, omega_track = angular_speed(track_time_ms, track_rotvec)
    t_ref, omega_ref = angular_speed(
        np.asarray(reference_time_ms, dtype=float) + float(offset_ms), reference_rotvec
    )
    resampled = CubicSpline(t_ref, omega_ref)(t_track)
    inside = (t_track >= t_ref.min()) & (t_track <= t_ref.max())
    if not inside.any():
        raise ValueError(f"offset {offset_ms} ms leaves no overlap between the series")
    return float(np.sqrt(np.mean((omega_track - resampled)[inside] ** 2)))


def refine_offset_omega(
    track_time_ms: np.ndarray,
    track_rotvec: np.ndarray,
    reference_time_ms: np.ndarray,
    reference_rotvec: np.ndarray,
    initial_offset_ms: float,
) -> float:
    """Stage 2: minimise :func:`omega_mismatch_rms` by Nelder-Mead."""

    def objective(x):
        return omega_mismatch_rms(
            track_time_ms, track_rotvec, reference_time_ms, reference_rotvec, float(x[0])
        )

    result = fmin(
        objective,
        np.asarray([float(initial_offset_ms)]),
        maxiter=STAGE2_MAXITER,
        xtol=STAGE2_XTOL_MS,
        disp=False,
    )
    return float(np.ravel(result)[0])


def _standardise(signal: np.ndarray) -> np.ndarray:
    signal = signal - np.mean(signal)
    sd = np.std(signal)
    return signal if sd < 1e-8 else signal / sd


def _lag_samples_to_ms(track_time_ms: np.ndarray, lag_samples: float) -> float:
    """Convert a lag in samples to milliseconds, via the camera's own clock.

    The camera sampling interval is not exactly uniform, so the conversion
    indexes the actual timestamps rather than multiplying by a nominal period,
    interpolating linearly between the two bracketing samples.
    """
    lower = int(np.floor(lag_samples))
    upper = int(np.ceil(lag_samples))
    frac = lag_samples - lower
    t0 = track_time_ms[0]
    last = len(track_time_ms) - 1

    def at(index: int) -> float:
        if index >= 0:
            return float(track_time_ms[min(index, last)] - t0)
        return -float(track_time_ms[min(-index, last)] - t0)

    t_lower, t_upper = at(lower), at(upper)
    return t_lower + frac * (t_upper - t_lower)
