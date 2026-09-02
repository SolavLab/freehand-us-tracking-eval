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

__all__ = ["Summary", "summarize", "iqr", "percentile"]


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
