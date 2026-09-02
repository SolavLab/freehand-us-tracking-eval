"""Fitting a rigid body to the marker cluster.

The Vicon system reports individual marker positions; the reference trajectory
the comparison needs is the pose of the cluster as a rigid body. That is
recovered per frame by the Kabsch algorithm: the least-squares rotation aligning
the marker constellation in a reference frame to its position in the current
frame.

Occlusion is handled by fitting only the markers visible in a given frame, with
the translation corrected back to the *full* constellation's centroid so the
reported position does not jump when a marker drops out. The per-frame RMS
residual of the fit is the quantity the paper reports as the reference's own
uncertainty (below 0.28 mm mean, below 0.74 mm maximum across the four
recordings).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from ..io.mocap import MocapData

__all__ = ["RigidBodyTrajectory", "fit_rigid_body"]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RigidBodyTrajectory:
    """Pose of the marker cluster over time, in the Vicon frame."""

    frame: np.ndarray        # (n,)
    time_ms: np.ndarray      # (n,)
    translation: np.ndarray  # (n, 3) millimetres
    rotvec: np.ndarray       # (n, 3) rotation vectors, radians
    residual_mm: np.ndarray  # (n,) RMS of the fit

    def __len__(self) -> int:
        return int(self.frame.size)

    @property
    def rotation(self) -> Rotation:
        return Rotation.from_rotvec(self.rotvec)

    def matrices(self) -> np.ndarray:
        """``(n, 4, 4)`` homogeneous transforms, marker cluster in Vicon."""
        out = np.zeros((len(self), 4, 4))
        out[:, :3, :3] = self.rotation.as_matrix()
        out[:, :3, 3] = self.translation
        out[:, 3, 3] = 1.0
        return out

    @property
    def valid(self) -> np.ndarray:
        """Frames where the fit succeeded."""
        return ~np.isnan(self.residual_mm)


def fit_rigid_body(
    mocap: MocapData, *, min_markers: int = 4, reference_frame: int = 0
) -> RigidBodyTrajectory:
    """Fit the marker cluster frame by frame.

    ``reference_frame`` fixes the constellation the rotation is measured
    against. The resulting orientation is therefore relative to the cluster's
    pose in that frame, which is an arbitrary but constant choice -- it cancels
    in the hand-eye calibration, which absorbs any constant rotation.

    Frames with fewer than ``min_markers`` visible markers yield NaN rather than
    a fit from too little data.
    """
    positions = mocap.positions                       # (n, m, 3)
    n_frames, n_markers = positions.shape[:2]
    if n_markers < min_markers:
        raise ValueError(
            f"{n_markers} markers in the recording, at least {min_markers} required"
        )

    reference = positions[reference_frame]
    if np.isnan(reference).any():
        raise ValueError(
            f"reference frame {reference_frame} has occluded markers; "
            "the constellation must be complete in the frame the fit is anchored to"
        )
    reference_centroid = reference.mean(axis=0)

    translation = np.full((n_frames, 3), np.nan)
    rotvec = np.full((n_frames, 3), np.nan)
    residual = np.full(n_frames, np.nan)

    dropped = 0
    for i in range(n_frames):
        points = positions[i]
        visible = ~np.isnan(points).any(axis=1)
        if visible.sum() < min_markers:
            dropped += 1
            continue

        p = points[visible]
        q = reference[visible]
        p_centroid = p.mean(axis=0)
        q_centroid = q.mean(axis=0)
        p_centred = p - p_centroid
        q_centred = q - q_centroid

        # Kabsch: R maps the reference constellation onto the current one.
        u, _, vt = np.linalg.svd(p_centred.T @ q_centred)
        if np.linalg.det(u @ vt) < 0:
            vt = vt.copy()
            vt[-1, :] *= -1
        r = u @ vt

        rotvec[i] = Rotation.from_matrix(r).as_rotvec()
        # Correct the centroid back to the full constellation, so a dropped
        # marker does not displace the reported position of the body.
        translation[i] = p_centroid + r @ (reference_centroid - q_centroid)
        aligned = (r @ q_centred.T).T + p_centroid
        residual[i] = np.sqrt(np.mean(np.sum((p - aligned) ** 2, axis=1)))

    if dropped:
        log.warning(
            "rigid-body fit skipped %d of %d frames with fewer than %d visible markers",
            dropped, n_frames, min_markers,
        )
    return RigidBodyTrajectory(
        frame=mocap.frame, time_ms=mocap.time_ms,
        translation=translation, rotvec=rotvec, residual_mm=residual,
    )
