"""Unit tests for the residual-metric helpers.

``test_manuscript.py`` pins these functions to the values the paper prints.
These tests pin them to cases whose answer is known independently of the
dataset, so a change that happens to preserve the published ranges still has
to be right.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import lfilter

from vipose.metrics import (
    bootstrap_standard_error,
    correlation_time,
    effective_sample_size,
    median_standard_error,
)

DT = 1 / 60.0


def ar1(phi: float, n: int = 100_000, seed: int = 0) -> np.ndarray:
    """A first-order autoregressive series, whose correlation time is known."""
    rng = np.random.default_rng(seed)
    return lfilter([1.0], [1.0, -phi], rng.normal(size=n))


@pytest.mark.parametrize("phi", [0.9, 0.95])
def test_recovers_the_correlation_time_of_an_ar1_series(phi):
    """For AR(1), tau = dt (1 + phi) / (1 - phi) exactly."""
    expected = DT * (1 + phi) / (1 - phi)
    got = correlation_time(ar1(phi), DT)
    assert got == pytest.approx(expected, rel=0.2), (
        f"phi={phi}: estimated {got:.4f} s against {expected:.4f} s"
    )


def test_white_noise_is_worth_its_frame_count():
    """With no correlation the estimate collapses to one sample per frame."""
    rng = np.random.default_rng(0)
    assert correlation_time(rng.normal(size=50_000), DT) == pytest.approx(DT, rel=0.05)


def test_untruncated_sum_would_be_degenerate():
    """Why the sum is truncated at all.

    Over every lag the normalized autocorrelation of this estimator sums to
    -1/2 identically, so an untruncated ``1 + 2 * sum`` returns zero for any
    input -- including one with real correlation, as here.
    """
    from vipose.metrics import _normalized_autocorrelation

    rho = _normalized_autocorrelation(ar1(0.9))
    assert rho[1:].sum() == pytest.approx(-0.5, abs=1e-9)
    assert 1 + 2 * rho[1:].sum() == pytest.approx(0.0, abs=1e-9)
    assert correlation_time(ar1(0.9), DT) > 10 * DT


def test_effective_sample_size_is_the_duration_over_the_correlation_time():
    x = ar1(0.9)
    t = np.arange(x.size) * DT
    assert effective_sample_size(x, t) == pytest.approx(
        (t[-1] - t[0]) / correlation_time(x, DT), rel=1e-9
    )


def test_correlated_series_is_worth_far_less_than_its_frame_count():
    x = ar1(0.95)
    t = np.arange(x.size) * DT
    assert effective_sample_size(x, t) < x.size / 30


def test_median_standard_error_is_the_robust_asymptotic_formula():
    """sqrt(pi/2) * (IQR / 1.349) / sqrt(N_eff), stated as an identity."""
    from vipose.metrics import iqr

    x = ar1(0.9)
    t = np.arange(x.size) * DT
    expected = np.sqrt(np.pi / 2) * (iqr(x) / 1.349) / np.sqrt(effective_sample_size(x, t))
    assert median_standard_error(x, t) == pytest.approx(expected, rel=1e-12)


def test_more_correlated_series_has_the_larger_standard_error():
    """At equal spread, correlation is what costs precision.

    Both series are scaled to the same interquartile range first, because an
    AR(1) process's variance also grows with phi and would otherwise account
    for part of the difference.
    """
    from vipose.metrics import iqr

    slow, fast = ar1(0.95), ar1(0.8)
    slow, fast = slow / iqr(slow), fast / iqr(fast)
    t_slow = np.arange(slow.size) * DT
    t_fast = np.arange(fast.size) * DT
    assert median_standard_error(slow, t_slow) > median_standard_error(fast, t_fast)


def test_the_scale_it_uses_ignores_an_extreme_outlier():
    """One wild frame moves the sample SD and not the interquartile range.

    The standard error itself still moves, because a spike is uncorrelated
    noise and shortens the correlation time. Robustness here is a property of
    the scale, which is the part that would otherwise track the long upper
    tail these residual distributions have.
    """
    from vipose.metrics import iqr

    x = ar1(0.9)
    spiked = x.copy()
    spiked[x.size // 2] += 500.0
    assert iqr(spiked) == pytest.approx(iqr(x), rel=1e-4)
    assert np.std(spiked, ddof=1) > 1.15 * np.std(x, ddof=1)


@pytest.mark.parametrize(
    "values, kwargs, message",
    [
        (np.ones(10), {}, "constant"),
        (np.array([1.0]), {}, "at least two"),
        (np.array([1.0, np.nan, 2.0]), {}, "non-finite"),
        (np.ones((3, 3)), {}, "1-D"),
    ],
)
def test_rejects_series_it_cannot_describe(values, kwargs, message):
    with pytest.raises(ValueError, match=message):
        correlation_time(values, DT, **kwargs)


@pytest.mark.parametrize("dt", [0.0, -1.0, np.nan])
def test_rejects_a_nonsensical_sampling_interval(dt):
    with pytest.raises(ValueError, match="dt_s must be positive"):
        correlation_time(ar1(0.9, n=1000), dt)


def test_effective_sample_size_rejects_mismatched_times():
    with pytest.raises(ValueError, match="shape mismatch"):
        effective_sample_size(np.zeros(10), np.zeros(9))


# --------------------------------------------------------------------------
# Moving-block bootstrap
# --------------------------------------------------------------------------


def times_for(x):
    return np.arange(x.size) * DT


def test_bootstrap_is_reproducible_from_its_seed():
    x = ar1(0.9, n=20_000)
    t = times_for(x)
    a = bootstrap_standard_error(x, t, replicates=500, seed=7)
    b = bootstrap_standard_error(x, t, replicates=500, seed=7)
    c = bootstrap_standard_error(x, t, replicates=500, seed=8)
    assert a == b
    assert a != c


def test_bootstrap_does_not_depend_on_chunk_size():
    """Chunking bounds memory; it must not touch the answer."""
    x = ar1(0.9, n=20_000)
    t = times_for(x)
    assert bootstrap_standard_error(x, t, replicates=400, seed=1, chunk=400) == (
        bootstrap_standard_error(x, t, replicates=400, seed=1, chunk=50)
    )


def test_bootstrap_matches_the_textbook_result_for_white_noise():
    """With no correlation, the median's standard error is the classical one."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=20_000)
    expected = np.sqrt(np.pi / 2) * x.std(ddof=1) / np.sqrt(x.size)
    got = bootstrap_standard_error(x, times_for(x), replicates=2000, seed=0)
    assert got == pytest.approx(expected, rel=0.15)


def test_bootstrap_exceeds_the_white_noise_result_when_correlated():
    """Correlation is what the block structure is there to carry."""
    x = ar1(0.95, n=20_000)
    naive = np.sqrt(np.pi / 2) * x.std(ddof=1) / np.sqrt(x.size)
    assert bootstrap_standard_error(x, times_for(x), replicates=2000, seed=0) > 3 * naive


def test_bootstrap_accepts_any_statistic():
    """A tail is estimated less precisely than a centre."""
    x = ar1(0.9, n=20_000)
    t = times_for(x)
    median = bootstrap_standard_error(x, t, replicates=2000, seed=0)
    p95 = bootstrap_standard_error(
        x, t, lambda a, axis: np.percentile(a, 95, axis=axis), replicates=2000, seed=0
    )
    assert p95 > median


def test_bootstrap_rejects_a_series_shorter_than_its_block():
    """A block that does not fit is a refusal, not a silently truncated block."""
    x = ar1(0.9, n=500)
    with pytest.raises(ValueError, match="too short for its correlation time"):
        bootstrap_standard_error(x, times_for(x), replicates=100, seed=0, block_multiple=500)


def test_bootstrap_rejects_mismatched_times_and_too_few_replicates():
    x = ar1(0.9, n=5000)
    with pytest.raises(ValueError, match="shape mismatch"):
        bootstrap_standard_error(x, times_for(x)[:-1], replicates=100)
    with pytest.raises(ValueError, match="at least two replicates"):
        bootstrap_standard_error(x, times_for(x), replicates=1)
