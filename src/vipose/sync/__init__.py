"""Temporal alignment of a tracking trajectory to the reference.

The cameras and the Vicon system run on independent clocks, so the offset
between them is estimated per pipeline and per recording, in three stages:

1. :func:`~vipose.sync.stages.crosscorrelation_offset` -- cross-correlate the
   angular-speed magnitudes and refine the peak to sub-sample precision.
2. :func:`~vipose.sync.stages.refine_offset_omega` -- minimise the RMS
   disagreement between the two angular-speed series.
3. :func:`~vipose.sync.stages.refine_offset_residual` -- once a spatial
   calibration exists, minimise the median rotational residual.

Stages 1 and 2 use the angular-speed magnitude because it is the only quantity
comparable between the two systems before the spatial calibration exists: it is
invariant to the coordinate frames the trajectories are expressed in. Being a
scalar, it constrains the offset loosely -- which is what stage 3 exists to fix.
"""

from .resample import resample_reference, shared_window
from .stages import crosscorrelation_offset, omega_mismatch_rms, refine_offset_omega

__all__ = [
    "crosscorrelation_offset",
    "omega_mismatch_rms",
    "refine_offset_omega",
    "resample_reference",
    "shared_window",
]
