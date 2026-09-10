"""Visual-inertial pose tracking: temporal synchronization, hand-eye
calibration and residual metrics for comparing a camera trajectory against a
motion-capture reference.

The reusable algorithms are exported directly from this package -- see
``docs/algorithms.md`` for how they fit together and a worked example that
does not require ``Dataset`` or a dataset manifest. ``Cell``, ``Dataset`` and
``DatasetError`` are the manifest-driven layer used by the ``vipose`` CLI and
by this repository's own shipped example.

Nothing in this package imports matplotlib.pyplot. Figures live in
``vipose.report.figures`` and are drawn only from the command line, after
results have been written to disk.
"""

__version__ = "1.0.0"

from .calibration import HandEyeResult, MarkerFrame, marker_frame, solve_hand_eye
from .conditioning import shah_margin, shah_singular_values
from .geometry.rigid_body import RigidBodyTrajectory, fit_rigid_body
from .geometry.transforms import (
    from_matrices,
    relative_rotation,
    rotation_angle_deg,
    split_rt,
    to_matrices,
)
from .kinematics import angular_speed, rotational_increments
from .metrics import Summary, percentile, rotational_residual, summarize, translational_residual
from .pipeline import calibrate_pipeline
from .recordings import Cell, Dataset, DatasetError
from .sync.resample import resample_reference
from .sync.stages import crosscorrelation_offset, refine_offset_omega

__all__ = [
    "__version__",
    # dataset manifest layer
    "Cell",
    "Dataset",
    "DatasetError",
    # temporal synchronization
    "crosscorrelation_offset",
    "refine_offset_omega",
    "resample_reference",
    # rigid-body fitting
    "RigidBodyTrajectory",
    "fit_rigid_body",
    # hand-eye calibration
    "HandEyeResult",
    "MarkerFrame",
    "marker_frame",
    "solve_hand_eye",
    "shah_margin",
    "shah_singular_values",
    # the per-recording driver
    "calibrate_pipeline",
    # SE(3) and kinematics helpers
    "to_matrices",
    "from_matrices",
    "split_rt",
    "relative_rotation",
    "rotation_angle_deg",
    "rotational_increments",
    "angular_speed",
    # residual metrics
    "Summary",
    "summarize",
    "percentile",
    "translational_residual",
    "rotational_residual",
]
