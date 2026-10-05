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

__all__ = [
    "RigidBodyTrajectory",
    "fit_rigid_body",
    "ReferenceUncertainty",
    "cluster_rms_radius",
    "reference_uncertainty",
]

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


@dataclass(frozen=True)
class ReferenceUncertainty:
    """What a fit residual implies about the reference pose it produces."""

    n_markers: int
    rms_radius_mm: float
    coordinate_noise_mm: float   # per coordinate, one axis of one marker
    marker_noise_mm: float       # per marker, magnitude of the 3-D displacement
    axis_deg: tuple[float, float, float]  # about each principal axis, ascending
    orientation_deg: float       # magnitude over the three axes


def cluster_rms_radius(positions) -> float:
    """RMS distance of the markers from their centroid, in millimetres.

    The lever arm that converts marker noise into orientation uncertainty. For
    this dataset's five-marker constellation it is 166 mm.
    """
    p = np.asarray(positions, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"positions must be (n, 3), got {p.shape}")
    return float(np.sqrt((((p - p.mean(axis=0)) ** 2).sum(axis=1)).mean()))


def reference_uncertainty(residual_mm: float, positions) -> ReferenceUncertainty:
    """Propagate a fit residual to the orientation uncertainty of the reference.

    The translational uncertainty of the reference is measured directly, as the
    residual of the fit. The orientation uncertainty is not, and has to be
    inferred from that residual and the geometry of the cluster. This is the
    derivation behind the 0.07 deg the manuscript quotes, which was carried by
    hand for several drafts and reconstructed wrongly twice.

    Two conversions, and both are easy to get wrong by a factor of sqrt(3):

    ``residual_mm`` is what :func:`fit_rigid_body` reports, the root mean over
    *markers* of the squared point-to-point error. It is therefore a
    displacement magnitude, not a per-coordinate scatter, and it understates
    the noise it is fitted to because the fit absorbs six degrees of freedom:
    with ``n`` markers only ``3n - 6`` of the ``3n`` coordinate errors survive
    into it. Undoing that gives the per-coordinate noise

        sigma = residual / sqrt((3n - 6) / n)

    and ``sqrt(3) * sigma``, equivalently ``residual * sqrt(n / (n - 2))``, is
    the per-marker displacement magnitude -- 0.36 mm here. Reading that 0.36 mm
    back as a per-coordinate scatter is the trap: it would imply a 0.48 mm
    residual and 0.12 deg, neither of which the data show.

    Orientation then follows from the Cramer-Rao bound for a rotation about an
    axis, which the Kabsch fit attains. A marker at ``r`` moves perpendicular
    to the axis by its perpendicular distance ``d`` times the angle, so each
    marker contributes ``d^2 / sigma^2`` of information and the per-axis
    standard error is ``sigma / sqrt(sum d^2)``. Summing those in quadrature
    over the three principal axes gives the magnitude of the orientation error.
    ``axis_deg`` reports the three separately: for a cluster close to
    spherically symmetric they agree, which is what "essentially isotropic"
    means in the manuscript.

    Rotation and translation are estimated jointly, but about a centred
    constellation they are uncorrelated, so the translation costs the rotation
    no precision and does not enter.
    """
    p = np.asarray(positions, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"positions must be (n, 3), got {p.shape}")
    n = p.shape[0]
    if n < 3:
        raise ValueError(f"need at least three markers to constrain a rotation, got {n}")
    if not np.isfinite(residual_mm) or residual_mm <= 0:
        raise ValueError(f"residual_mm must be positive and finite, got {residual_mm}")

    centred = p - p.mean(axis=0)
    sigma = residual_mm / np.sqrt((3 * n - 6) / n)
    scatter = centred.T @ centred
    lever = np.trace(scatter) - np.linalg.eigvalsh(scatter)
    if lever.min() <= 0 or not np.isfinite(lever).all():
        raise ValueError("markers are collinear; rotation about their axis is unconstrained")
    axis = np.degrees(sigma / np.sqrt(lever))
    return ReferenceUncertainty(
        n_markers=n,
        rms_radius_mm=cluster_rms_radius(p),
        coordinate_noise_mm=float(sigma),
        marker_noise_mm=float(np.sqrt(3) * sigma),
        axis_deg=tuple(float(a) for a in np.sort(axis)),
        orientation_deg=float(np.sqrt((axis**2).sum())),
    )
