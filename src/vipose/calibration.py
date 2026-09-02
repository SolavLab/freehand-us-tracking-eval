"""Robot-world/hand-eye calibration, and the marker-intrinsic frame.

The two trajectories cannot be compared directly: the pipeline reports the
camera pose in its own SLAM world frame ``W``, and the motion-capture system
reports the marker cluster in the laboratory frame ``V``. Bringing both into
``V`` needs two further transforms, ``T_V_W`` and ``T_M_C``. Neither is measured,
but both are constant within a recording -- ``W`` is fixed once tracking begins,
and camera and markers were rigidly attached -- so at every frame

    T_V_W . T_W_C(t) = T_V_M(t) . T_M_C

and a recording of many frames determines the twelve unknowns. That is the
robot-world/hand-eye problem.

.. important::

   OpenCV's ``calibrateRobotWorldHandEye`` satisfies ``A = X B Z⁻¹``, i.e.
   ``A Z = X B``, and returns the pair ``(Z, X)``. In this project's notation
   that is ``T_C_W(t) . T_W_V = T_C_M . T_M_V(t)``, so the arguments are the
   *inverses* of the trajectories above. Assuming ``A X = Z B`` instead yields a
   calibration roughly 106 degrees wrong. ``tests/test_opencv_convention.py``
   pins this empirically against six candidate laws.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

# OpenCV pulls in a Qt plugin that fails on a headless host unless this is set
# before the import. Confined to this module, which is the only one needing cv2.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import cv2  # noqa: E402

from .geometry.transforms import split_rt  # noqa: E402

__all__ = ["HandEyeResult", "MarkerFrame", "solve_hand_eye", "marker_frame", "SHAH"]

log = logging.getLogger(__name__)

#: Shah's method. Chosen because it is the one the published results used, and
#: because its well-posedness is measurable on the matrix it factorises -- see
#: ``vipose.conditioning``. Selecting a method by the residual it produces would
#: be fitting to the reported metric; ``docs/method-selection.md`` records the
#: comparison that was made.
SHAH = cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH


@dataclass(frozen=True)
class HandEyeResult:
    """The two constant transforms relating the reference to the SLAM frame."""

    T_camera_to_marker: np.ndarray   # (4, 4)  ``T_M_C``
    T_slamworld_to_vicon: np.ndarray  # (4, 4)  ``T_V_W``
    n_frames_used: int

    @property
    def T_vicon_to_slamworld(self) -> np.ndarray:
        return np.linalg.inv(self.T_slamworld_to_vicon)

    @property
    def camera_to_marker_distance_mm(self) -> float:
        """``|t_M_C|``, invariant to the choice of coordinate frame.

        The paper uses the spread of this quantity across recordings as the
        floor below which differences are not interpretable.
        """
        return float(np.linalg.norm(self.T_camera_to_marker[:3, 3]))


@dataclass(frozen=True)
class MarkerFrame:
    """An orthonormal frame defined by the marker constellation itself.

    Independent of the Kabsch fit's arbitrary reference frame, so the
    camera-to-marker transform expressed in it is comparable across recordings.
    """

    centroid: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray

    def rotation(self) -> np.ndarray:
        """``(3, 3)`` whose columns are the axes, mapping this frame to Vicon."""
        return np.column_stack([self.x_axis, self.y_axis, self.z_axis])

    def from_vicon(self) -> np.ndarray:
        """``(4, 4)`` rotating Vicon-frame quantities into this frame.

        Rotation only, no translation: the marker-intrinsic frame is used to
        express *orientations* comparably across recordings, and giving it the
        cluster centroid as an origin would make the result depend on where the
        cluster happened to be in the room.
        """
        out = np.eye(4)
        out[:3, :3] = self.rotation().T
        return out

    def express(self, transform: np.ndarray) -> np.ndarray:
        """Re-express a Vicon-frame transform in the marker-intrinsic frame.

        Applied to the camera-to-marker transform this gives the quantity that
        should be constant across recordings, since it no longer depends on the
        arbitrary reference frame the rigid-body fit is anchored to.
        """
        return self.from_vicon() @ np.asarray(transform, dtype=float)


def solve_hand_eye(
    camera_in_world: np.ndarray,
    marker_in_vicon: np.ndarray,
    *,
    method: int = SHAH,
) -> HandEyeResult:
    """Solve for ``T_M_C`` and ``T_V_W``.

    ``camera_in_world`` is ``T_W_C(t)`` and ``marker_in_vicon`` is ``T_V_M(t)``,
    both ``(n, 4, 4)``. Frames whose rotation is NaN or singular are dropped;
    an occluded reference frame therefore removes a sample rather than
    poisoning the fit.
    """
    camera_in_world = np.asarray(camera_in_world, dtype=float)
    marker_in_vicon = np.asarray(marker_in_vicon, dtype=float)
    if camera_in_world.shape != marker_in_vicon.shape:
        raise ValueError(
            f"shape mismatch: {camera_in_world.shape} vs {marker_in_vicon.shape}"
        )

    # OpenCV's convention -- see the module docstring.
    a = np.linalg.inv(camera_in_world)   # T_C_W(t)
    b = np.linalg.inv(marker_in_vicon)   # T_M_V(t)
    r_a, t_a = split_rt(a)
    r_b, t_b = split_rt(b)

    det_a = np.linalg.det(r_a)
    det_b = np.linalg.det(r_b)
    valid = ~(
        np.isnan(det_a) | np.isnan(det_b) | (np.abs(det_a) < 1e-10) | (np.abs(det_b) < 1e-10)
    )
    if valid.sum() < 3:
        raise ValueError(f"only {valid.sum()} usable frames; hand-eye needs at least 3")
    if not valid.all():
        log.info("hand-eye calibration using %d of %d frames", valid.sum(), len(valid))

    r_world_to_slam, t_world_to_slam, r_marker_to_camera, t_marker_to_camera = (
        cv2.calibrateRobotWorldHandEye(
            list(r_a[valid]), list(t_a[valid]),
            list(r_b[valid]), list(t_b[valid]),
            method=method,
        )
    )

    t_vicon_to_slamworld = _homogeneous(r_world_to_slam, t_world_to_slam)
    t_marker_to_cam = _homogeneous(r_marker_to_camera, t_marker_to_camera)
    return HandEyeResult(
        T_camera_to_marker=np.linalg.inv(t_marker_to_cam),
        T_slamworld_to_vicon=np.linalg.inv(t_vicon_to_slamworld),
        n_frames_used=int(valid.sum()),
    )


def marker_frame(positions: np.ndarray, *, order: tuple[str, ...] | None = None) -> MarkerFrame:
    """Build the marker-intrinsic frame from one frame's marker positions.

    ``positions`` is ``(5, 3)`` in the order ``top, back_left, back_right,
    front_right, front_left``.

    The Y axis runs from the midpoint of the back pair to the midpoint of the
    front pair. The X axis is **constructed orthogonal to it**, as
    ``Y x (top - back_midpoint)``, rather than taken as the back-left to
    back-right direction: the latter is orthonormal only if the marker layout
    is exactly rectangular, which it is not. Z completes the right-handed set.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.shape != (5, 3):
        raise ValueError(
            f"expected 5 markers x 3 coordinates, got {positions.shape}"
            + (f" for order {order}" if order else "")
        )
    if np.isnan(positions).any():
        raise ValueError("marker frame cannot be built from a frame with occluded markers")

    top, back_left, back_right, front_right, front_left = positions
    back_mid = (back_left + back_right) / 2.0
    front_mid = (front_left + front_right) / 2.0

    y_axis = _unit(front_mid - back_mid)
    x_axis = _unit(np.cross(y_axis, top - back_mid))
    z_axis = _unit(np.cross(x_axis, y_axis))
    return MarkerFrame(positions.mean(axis=0), x_axis, y_axis, z_axis)


def _homogeneous(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return out


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("degenerate marker geometry: axis has zero length")
    return v / n
