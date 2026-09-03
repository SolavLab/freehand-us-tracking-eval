"""Well-posedness of the hand-eye calibration.

A hand-eye calibration is determined only if the motion rotates about more than
one axis. Because the calibration here is solved by Shah's method, that
condition can be measured directly on the matrix Shah factorizes, with no
arbitrary reference pose and no lag to choose:

.. math::  K = \\frac{1}{N} \\sum_i R_{B_i} \\otimes R_{A_i}

Each term is a Kronecker product of two rotations and is therefore orthogonal,
so every singular value is at most 1. An average of length-preserving maps
contracts except along directions on which every term agrees -- and
:math:`\\operatorname{vec}(R_Z)` is such a direction, since the calibration is
constant. So :math:`\\sigma_1 = 1`, attained *at the calibration itself*.

A second unit singular value would mean a second consistent solution, i.e. the
calibration is not identifiable. The margin :math:`1 - \\sigma_2` therefore
measures how sharply the true solution is separated from the nearest competitor,
with zero marking a degenerate motion.

Substituting the noiseless relation shows the SLAM side drops out entirely, so
the margin depends only on the reference trajectory and is identical for all
three pipelines within a recording. That is why it is computed here from the
Vicon rotations alone.

This replaced an earlier measure, ``motion_axis_excitation``, which took every
pose relative to the *first* pose of the window. That anchor is arbitrary and
rotations do not commute, so its graded values moved a great deal when the
anchor changed -- on a synthetic motion that is exactly half about x and half
about y, it reads 0.218 anchored at the first frame and 1.000 anchored at the
middle. On the real recordings it reversed the Pivot/Freehand ordering. It
remains valid as a binary degeneracy test and nothing more.
"""

from __future__ import annotations

import numpy as np

__all__ = ["shah_margin", "shah_singular_values"]


def shah_singular_values(rotations: np.ndarray) -> np.ndarray:
    """Singular values of ``K``, computed from the reference rotations alone.

    ``rotations`` is ``(n, 3, 3)``. The ``1/N`` factor is retained -- reference
    implementations drop it, which rescales the singular values without
    affecting the vectors the calibration is recovered from, but here the
    singular values *are* the quantity of interest.
    """
    r = np.asarray(rotations, dtype=float)
    if r.ndim != 3 or r.shape[1:] != (3, 3):
        raise ValueError(f"expected (n, 3, 3) rotation matrices, got {r.shape}")
    if len(r) == 0:
        raise ValueError("no rotations given")
    k = np.einsum("nij,nkl->ikjl", r, r).reshape(9, 9) / len(r)
    return np.linalg.svd(k, compute_uv=False)


def shah_margin(rotations: np.ndarray, *, tol: float = 1e-9) -> float:
    """``1 - sigma_2``. Zero means a degenerate motion.

    Asserts ``sigma_1 == 1`` to within ``tol``: that identity is a theorem
    about this matrix, so a violation means the input is not a stack of proper
    rotations and the margin below it would be meaningless.

    Reference points, from synthetic motions: single-axis 0.000, two-axis
    0.0234, isotropic 0.3270.
    """
    s = shah_singular_values(rotations)
    if abs(s[0] - 1.0) > tol:
        raise ValueError(
            f"sigma_1 is {s[0]!r}, not 1. K averages orthogonal matrices, so this "
            "is a theorem, not an approximation -- the inputs are probably not "
            "proper rotation matrices."
        )
    return float(1.0 - s[1])
