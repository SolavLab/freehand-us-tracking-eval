"""Summary statistics over per-frame residual series.

Pure functions: no file I/O, no plotting, no global state. That separation is
the point of this module.

In the code this was ported from, the residual values were returned *by the
plotting functions* -- ``plot_df_to_df_distance`` returned ``(plt, distances)``
and ``plot_df_to_df_angles`` returned ``angles``. Every number in the paper's
results table was therefore a by-product of drawing a figure, which is why a
matplotlib error could destroy a twelve-cell run, and why running with plots
disabled still drew them. Nothing here imports pyplot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

__all__ = [
    "Summary",
    "summarize",
    "iqr",
    "percentile",
    "translational_residual",
    "rotational_residual",
    "correlation_time",
    "effective_sample_size",
    "median_standard_error",
]


@dataclass(frozen=True)
class Summary:
    """The statistics the paper reports for one residual series."""

    n: int
    median: float
    iqr: float
    p95: float
    mean: float
    sd: float
    maximum: float

    def as_dict(self) -> dict:
        return asdict(self)


def percentile(values, q: float) -> float:
    """Linear-interpolation percentile, matching numpy's default.

    Named explicitly because the choice of interpolation method changes the
    reported 95th percentile in the third decimal place, and the published
    numbers were produced with this one.
    """
    return float(np.percentile(np.asarray(values, dtype=float), q))


def iqr(values) -> float:
    """Interquartile range, 75th minus 25th percentile."""
    v = np.asarray(values, dtype=float)
    return percentile(v, 75) - percentile(v, 25)


def summarize(values) -> Summary:
    """Summarize one residual series.

    Empty input is an error rather than a silent NaN: an empty series means the
    evaluation window was computed wrongly, which is a bug worth surfacing, not
    a cell to report as blank.
    """
    v = np.asarray(values, dtype=float)
    if v.ndim != 1:
        raise ValueError(f"expected a 1-D series, got shape {v.shape}")
    if v.size == 0:
        raise ValueError("empty residual series")
    if not np.all(np.isfinite(v)):
        raise ValueError(
            f"residual series contains {int((~np.isfinite(v)).sum())} non-finite values"
        )
    return Summary(
        n=int(v.size),
        median=float(np.median(v)),
        iqr=iqr(v),
        p95=percentile(v, 95),
        mean=float(np.mean(v)),
        sd=float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
        maximum=float(np.max(v)),
    )


def _normalized_autocorrelation(values) -> np.ndarray:
    """Autocorrelation of a mean-subtracted series, scaled so that lag 0 is 1.

    Computed by FFT with zero padding to twice the length, which makes it the
    linear (not circular) autocorrelation. The unnormalized estimator divides
    every lag by ``n`` rather than by the number of overlapping pairs, so the
    tail is damped rather than noisy -- the usual bias-for-variance trade, and
    the one the truncation rule below assumes.
    """
    v = np.asarray(values, dtype=float)
    if v.ndim != 1:
        raise ValueError(f"expected a 1-D series, got shape {v.shape}")
    if v.size < 2:
        raise ValueError("need at least two samples")
    if not np.all(np.isfinite(v)):
        raise ValueError("series contains non-finite values")
    v = v - v.mean()
    if not np.any(v):
        raise ValueError("series is constant; its autocorrelation is undefined")
    n = v.size
    f = np.fft.rfft(v, 2 * n)
    a = np.fft.irfft(f * np.conj(f))[:n].real
    return a / a[0]


def correlation_time(values, dt_s: float) -> float:
    """Integrated autocorrelation time of a residual series, in seconds.

    ``dt_s`` is the sampling interval. The estimate is

        tau = dt * (1 + 2 * sum_k rho_k)

    with the sum running over the initial lags at which ``rho_k`` is still
    positive and stopping at the first that is not.

    The truncation is not a refinement, it is what makes the quantity exist.
    For this estimator the sum of ``rho_k`` over *all* lags is identically
    -1/2, so an untruncated sum returns tau = 0 for every input. Beyond the
    first sign change the lags carry mostly estimation noise about zero, so
    keeping the initial positive run discards the tail without a window
    parameter chosen by hand.

    Geyer's initial positive sequence truncates on sums of adjacent lag pairs
    instead, which is provably positive-decreasing for a reversible Markov
    chain. A tracking-residual series is not one, and on this dataset the two
    rules agree to two decimals in every cell, so the simpler form is used.
    """
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError(f"dt_s must be positive and finite, got {dt_s}")
    rho = _normalized_autocorrelation(values)
    nonpositive = np.flatnonzero(rho[1:] <= 0)
    stop = int(nonpositive[0]) + 1 if nonpositive.size else rho.size
    return float(dt_s * (1.0 + 2.0 * rho[1:stop].sum()))


def effective_sample_size(values, times_s) -> float:
    """Number of independent samples a residual series is worth.

    The recording duration divided by the correlation time. Successive frames
    sample almost the same tracking state, so this is far below the frame
    count -- 16 to 92 against 1400 to 3600 frames for the rotational residuals
    of this dataset -- and it, not the frame count, sets how precisely any
    statistic of the series is determined.

    ``times_s`` are the sample times in seconds; the sampling interval is
    taken as their median difference, so an occasional dropped frame does not
    move it.
    """
    t = np.asarray(times_s, dtype=float)
    v = np.asarray(values, dtype=float)
    if t.shape != v.shape:
        raise ValueError(f"shape mismatch: {v.shape} values, {t.shape} times")
    if t.size < 2:
        raise ValueError("need at least two samples")
    duration = float(t[-1] - t[0])
    if duration <= 0:
        raise ValueError("times are not increasing")
    return duration / correlation_time(v, float(np.median(np.diff(t))))


def median_standard_error(values, times_s) -> float:
    """Standard error of the median of an autocorrelated residual series.

    The asymptotic ``sqrt(pi / 2) * sigma / sqrt(n)`` for a median, with the
    effective sample size in place of ``n`` and a robust scale in place of the
    standard deviation. The scale is the interquartile range over 1.349, which
    is the Gaussian-consistent estimator: residual series of this kind have a
    long upper tail, and the sample standard deviation follows it, overstating
    the uncertainty of a statistic that does not.

    This is what fixes the resolution of a comparison between two recordings.
    """
    v = np.asarray(values, dtype=float)
    scale = iqr(v) / 1.349
    return float(np.sqrt(np.pi / 2.0) * scale / np.sqrt(effective_sample_size(v, times_s)))


def translational_residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance between two stacks of poses, in millimetres.

    ``a`` and ``b`` are ``(n, 4, 4)``. This is the paper's :math:`\\Delta d(t)`.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return np.linalg.norm(a[:, :3, 3] - b[:, :3, 3], axis=1)


def rotational_residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle of the relative rotation between two stacks of poses, in degrees.

    The paper's :math:`\\Delta\\varphi(t)`.

    Computed as the magnitude of ``Rotation.from_matrix(Aᵀ B)``, which scipy
    derives via the quaternion. That is deliberately *not* the
    ``arccos((tr R - 1) / 2)`` form the paper writes: the two agree
    analytically, but arccos of the trace loses precision near zero rotation --
    its derivative diverges there -- and these residuals are a tenth of a degree.
    The quaternion route is well conditioned across the whole range, and it is
    what produced the published numbers.
    """
    from scipy.spatial.transform import Rotation

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

    out = np.full(len(a), np.nan)
    ok = ~(np.isnan(a).any(axis=(1, 2)) | np.isnan(b).any(axis=(1, 2)))
    if ok.any():
        relative = np.einsum("nji,njk->nik", a[ok, :3, :3], b[ok, :3, :3])
        out[ok] = np.degrees(Rotation.from_matrix(relative).magnitude())
    return out
