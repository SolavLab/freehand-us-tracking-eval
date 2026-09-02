"""SE(3) helpers.

Rigid-body transformations are represented as ``(n, 4, 4)`` arrays throughout.
Orientations on disk are **rotation vectors** (axis-angle), never Euler angles:
reading those columns as Euler angles introduces an orientation-dependent gain
error of up to 20 % on every rotational metric, and because the three pipelines
use different camera-frame conventions the error is pipeline-dependent, which is
the worst possible shape for a comparison. This project has actually suffered
that defect; ``from_euler`` is forbidden anywhere in ``src/`` by a lint rule.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = [
    "to_matrices",
    "from_matrices",
    "split_rt",
    "rotation_angle_deg",
    "relative_rotation",
]


def to_matrices(translation: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    """``(n, 3)`` translations and rotation vectors -> ``(n, 4, 4)`` transforms.

    Rows containing NaN produce all-NaN transforms rather than raising, so an
    occluded reference frame propagates as missing data instead of a failure.
    """
    translation = np.atleast_2d(np.asarray(translation, dtype=float))
    rotvec = np.atleast_2d(np.asarray(rotvec, dtype=float))
    if translation.shape != rotvec.shape:
        raise ValueError(f"shape mismatch: {translation.shape} vs {rotvec.shape}")

    out = np.full((len(translation), 4, 4), np.nan)
    ok = ~(np.isnan(translation).any(axis=1) | np.isnan(rotvec).any(axis=1))
    out[ok] = np.eye(4)
    if ok.any():
        out[ok, :3, :3] = Rotation.from_rotvec(rotvec[ok]).as_matrix()
        out[ok, :3, 3] = translation[ok]
    return out


def from_matrices(matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(n, 4, 4)`` -> ``(translation, rotvec)``, NaN-preserving."""
    matrices = np.asarray(matrices, dtype=float)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
        raise ValueError(f"expected (n, 4, 4), got {matrices.shape}")

    translation = np.full((len(matrices), 3), np.nan)
    rotvec = np.full((len(matrices), 3), np.nan)
    ok = ~np.isnan(matrices).any(axis=(1, 2))
    if ok.any():
        translation[ok] = matrices[ok, :3, 3]
        rotvec[ok] = Rotation.from_matrix(matrices[ok, :3, :3]).as_rotvec()
    return translation, rotvec


def split_rt(matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(n, 4, 4)`` -> ``(rotations (n, 3, 3), translations (n, 3))``."""
    matrices = np.asarray(matrices, dtype=float)
    return matrices[:, :3, :3], matrices[:, :3, 3]


def relative_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``aᵀ b`` for stacks of rotation matrices."""
    return np.einsum("nji,njk->nik", np.asarray(a, float), np.asarray(b, float))


def rotation_angle_deg(rotations: np.ndarray) -> np.ndarray:
    """Rotation angle of each matrix in a stack, in degrees.

    ``arccos((tr R - 1) / 2)``, clipped so floating-point error just outside
    [-1, 1] does not produce NaN.
    """
    rotations = np.asarray(rotations, dtype=float)
    trace = np.trace(rotations, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))
