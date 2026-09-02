"""Resampling the reference onto camera timestamps, and the shared window."""

from __future__ import annotations

import numpy as np

from ..io.mocap import Marker, MocapData

__all__ = ["resample_reference", "shared_window"]


def resample_reference(
    reference: MocapData, target_time_ms: np.ndarray, offset_ms: float
) -> MocapData:
    """Sample the marker trajectories at ``target_time_ms``.

    The reference clock is shifted by ``offset_ms`` first, so a positive offset
    means the Vicon clock runs ahead of the camera clock.

    Interpolation is **linear**, per marker coordinate. This is what the
    published implementation does, and the port is faithful to it.

    .. warning::

       Appendix A of the paper states that the marker trajectories were
       resampled by cubic spline, and gives a reason (linear interpolation of a
       240 Hz trajectory introduces a ripple at the Vicon sample period). That
       is not what the code does. There is exactly one cubic interpolation in
       the original implementation, and it is in the stage-2 angular-speed
       objective -- see :func:`vipose.sync.stages.omega_mismatch_rms`, which is
       cubic. The marker resampling that feeds the residuals is linear, both in
       the original and here.

       Changing it would alter every published number, so it is left as it is
       and the discrepancy is recorded rather than silently resolved either way.

    A rigid body is fitted to the resampled markers afterwards -- markers are
    interpolated first, then fitted, not the reverse. That ordering matters: the
    rigid-body fit is nonlinear, so interpolating fitted poses would not give
    the same answer.
    """
    target = np.asarray(target_time_ms, dtype=float)
    source = np.asarray(reference.time_ms, dtype=float) + float(offset_ms)

    if target.min() < source.min() or target.max() > source.max():
        raise ValueError(
            f"target times [{target.min():.1f}, {target.max():.1f}] ms fall outside the "
            f"reference span [{source.min():.1f}, {source.max():.1f}] ms at offset "
            f"{offset_ms:.3f} ms; np.interp would silently clamp"
        )

    markers = tuple(
        Marker(
            m.name,
            np.column_stack(
                [np.interp(target, source, m.xyz[:, i]) for i in range(3)]
            ),
        )
        for m in reference.markers
    )
    return MocapData(
        frame=np.interp(target, source, reference.frame),
        time_ms=target,
        markers=markers,
        rate_hz=reference.rate_hz,
    )


def shared_window(
    reference_spans_ms: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    """Intersect per-pipeline spans, expressed in the reference clock.

    All three pipelines must be evaluated over the same stretch of motion, or
    the comparison would be between different segments of the recording. The
    window is the intersection of the intervals each pipeline can cover once its
    own temporal offset is applied -- and it is defined in the **reference**
    time base, because the three camera clocks differ from it by different
    amounts and so their raw timestamps neither agree nor should.
    """
    if not reference_spans_ms:
        raise ValueError("no spans given")
    start = max(s for s, _ in reference_spans_ms.values())
    end = min(e for _, e in reference_spans_ms.values())
    if not end > start:
        raise ValueError(
            f"pipelines share no common window: spans {reference_spans_ms} "
            f"intersect to [{start}, {end}]"
        )
    return float(start), float(end)
