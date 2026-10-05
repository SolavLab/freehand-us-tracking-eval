"""Calibration and residuals, against the legacy code and the manuscript.

The hand-eye solve is compared against the legacy argument construction
computed in this process, at full double precision. It is deliberately *not*
compared against ``tests/golden/*/calibration.json``: those transforms were
recovered from the legacy text dump, which printed roughly nine significant
digits, so the stored reference is itself truncated at about 1e-7 relative. The
per-frame residuals in ``residuals.csv`` are full precision and are compared
exactly.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402

from vipose import Dataset  # noqa: E402
from vipose.calibration import SHAH, marker_frame, solve_hand_eye  # noqa: E402
from vipose.geometry.rigid_body import (  # noqa: E402
    cluster_rms_radius,
    fit_rigid_body,
    reference_uncertainty,
)
from vipose.geometry.transforms import to_matrices  # noqa: E402
from vipose.io.mocap import load_mocap  # noqa: E402
from vipose.io.tracks import load_track  # noqa: E402
from vipose.metrics import rotational_residual, translational_residual  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden"
DATASET = Dataset.load(ROOT / "datasets" / "probe-tracking-2025-10-23", verify_hashes=False)
CELLS = DATASET.cells()


@functools.cache
def _prepared(cell):
    """The native chain up to the calibration inputs, at the published offset."""
    golden = json.loads((GOLDEN / cell.pipeline / cell.recording / "calibration.json").read_text())
    lo, hi = golden["local_window_ms"]
    track = load_track(DATASET.tracking_csv(cell)).cropped(lo, hi)
    mocap = load_mocap(
        DATASET.reference_csv(cell.recording),
        marker_prefix=DATASET.marker_prefix,
        expected_rate_hz=DATASET.vicon_rate_hz,
    )
    from vipose.sync.resample import resample_reference

    resampled = resample_reference(mocap, track.time_ms, golden["temporal_offset_ms"])
    rb = fit_rigid_body(resampled, min_markers=4)
    return (
        to_matrices(track.translation, track.rotvec),   # T_W_C(t)
        to_matrices(rb.translation, rb.rotvec),         # T_V_M(t)
        resampled,
        golden,
    )


def _legacy_hand_eye(camera_in_world, marker_in_vicon):
    """The original argument construction, verbatim, in double precision."""
    a = [np.linalg.inv(t) for t in camera_in_world]      # T_C_W
    b = [np.linalg.inv(t) for t in marker_in_vicon]      # T_M_V
    t_a, r_a = [t[:3, 3] for t in a], [t[:3, :3] for t in a]
    t_b, r_b = [t[:3, 3] for t in b], [t[:3, :3] for t in b]

    keep = [
        i
        for i, (r1, r2) in enumerate(zip(r_a, r_b, strict=True))
        if not (
            np.isnan(np.linalg.det(r1))
            or np.isnan(np.linalg.det(r2))
            or abs(np.linalg.det(r1)) < 1e-10
            or abs(np.linalg.det(r2)) < 1e-10
        )
    ]
    r_v2z, t_v2z, r_rb2z, t_rb2z = cv2.calibrateRobotWorldHandEye(
        [r_a[i] for i in keep], [t_a[i] for i in keep],
        [r_b[i] for i in keep], [t_b[i] for i in keep],
        method=SHAH,
    )

    def h(r, t):
        m = np.eye(4)
        m[:3, :3] = r
        m[:3, 3] = np.asarray(t).reshape(3)
        return m

    return np.linalg.inv(h(r_rb2z, t_rb2z)), np.linalg.inv(h(r_v2z, t_v2z)), len(keep)


@pytest.mark.parametrize("cell", CELLS, ids=str)
def test_hand_eye_matches_legacy_construction(cell):
    camera_in_world, marker_in_vicon, _, _ = _prepared(cell)
    native = solve_hand_eye(camera_in_world, marker_in_vicon)
    t_mc, t_vw, n = _legacy_hand_eye(camera_in_world, marker_in_vicon)

    assert native.n_frames_used == n
    np.testing.assert_allclose(native.T_camera_to_marker, t_mc, rtol=0, atol=0)
    np.testing.assert_allclose(native.T_slamworld_to_vicon, t_vw, rtol=0, atol=0)


@pytest.mark.parametrize("cell", CELLS, ids=str)
def test_residuals_match_the_published_series(cell):
    """The end-to-end native chain reproduces the published per-frame errors."""
    camera_in_world, marker_in_vicon, _, _ = _prepared(cell)
    calib = solve_hand_eye(camera_in_world, marker_in_vicon)

    slam = calib.T_slamworld_to_vicon @ camera_in_world
    marker = marker_in_vicon @ calib.T_camera_to_marker
    d = translational_residual(slam, marker)
    phi = rotational_residual(slam, marker)

    reference = np.loadtxt(
        GOLDEN / cell.pipeline / cell.recording / "residuals.csv", delimiter=",", skiprows=1
    )
    assert len(d) == len(reference)
    # Both series are recomputed from the same inputs through the same
    # operations, so the only difference is floating-point association order.
    np.testing.assert_allclose(d, reference[:, 3], rtol=0, atol=1e-12)
    np.testing.assert_allclose(phi, reference[:, 4], rtol=0, atol=1e-11)


@pytest.mark.parametrize("cell", CELLS, ids=str)
def test_camera_to_marker_matches_the_stored_transform(cell):
    """Agrees with the golden transform to the precision it was stored at.

    The legacy dump printed about nine significant digits, which is why this
    tolerance is relative 1e-6 rather than the 1e-9 used elsewhere. The limit is
    in the reference, not the computation.
    """
    camera_in_world, marker_in_vicon, _, golden = _prepared(cell)
    calib = solve_hand_eye(camera_in_world, marker_in_vicon)
    np.testing.assert_allclose(
        calib.T_camera_to_marker, np.asarray(golden["T_camera_to_marker"]), rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("cell", CELLS, ids=str)
def test_marker_frame_matches_golden(cell):
    _, _, resampled, golden = _prepared(cell)
    frame = marker_frame(resampled.positions[0], order=DATASET.marker_order)

    np.testing.assert_allclose(frame.centroid, golden["marker_centroid"], rtol=1e-6, atol=1e-6)
    for axis, key in (
        (frame.x_axis, "marker_x_axis"),
        (frame.y_axis, "marker_y_axis"),
        (frame.z_axis, "marker_z_axis"),
    ):
        np.testing.assert_allclose(axis, golden[key], rtol=1e-6, atol=1e-8)


def test_marker_frame_axes_are_orthonormal():
    """The X axis is constructed orthogonal, not taken between two markers.

    Taking back-left to back-right is orthonormal only for an exactly
    rectangular layout, which this cluster is not.
    """
    _, _, resampled, _ = _prepared(CELLS[0])
    frame = marker_frame(resampled.positions[0])
    r = frame.rotation()
    np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-12)


def test_camera_to_marker_distance_reproduces_the_calibration_floor():
    """Cross-condition spread of |t_M_C|, which the paper quotes as a floor.

    Reported there as 6.32 mm for ZED-SDK, 4.28 for ZED-cuVSLAM and 2.77 for
    RS-cuVSLAM across the three motion conditions.
    """
    expected = {"zed-sdk": 6.32, "zed-cuvslam": 4.28, "rs-cuvslam": 2.77}
    conditions = ["pivot", "mixed", "freehand"]
    from vipose.recordings import Cell

    for pipeline, want in expected.items():
        norms = []
        for recording in conditions:
            camera_in_world, marker_in_vicon, _, _ = _prepared(Cell(pipeline, recording))
            calib = solve_hand_eye(camera_in_world, marker_in_vicon)
            norms.append(calib.camera_to_marker_distance_mm)
        spread = max(norms) - min(norms)
        assert spread == pytest.approx(want, abs=0.02), (
            f"{pipeline}: |t_M_C| spread {spread:.3f} mm, paper reports {want} mm"
        )


# ---------------------------------------------------------------------------
# The reference's own orientation uncertainty.
#
# The manuscript's Discussion quotes 0.36 mm of per-marker noise and 0.07 deg
# of orientation uncertainty for a 166 mm cluster. That chain was carried by
# hand for several drafts and reconstructed wrongly twice, both times by
# reading the 0.36 mm as a per-coordinate scatter rather than a displacement
# magnitude -- an error of sqrt(3), which happens to leave the quoted 0.07 deg
# looking right if it is then fed into a per-axis formula. These tests pin
# every link so the next reconstruction is a test run rather than an argument.
# ---------------------------------------------------------------------------

# Table 3, mean per-frame RMS residual within the shared evaluation window.
TABLE3_RESIDUAL_MM = {
    "pivot": 0.279,
    "freehand": 0.231,
    "mixed": 0.209,
    "pivot-repeat": 0.265,
}


def _constellation(recording: str) -> np.ndarray:
    mocap = load_mocap(
        DATASET.reference_csv(recording),
        marker_prefix=DATASET.marker_prefix,
        expected_rate_hz=DATASET.vicon_rate_hz,
    )
    positions = mocap.positions
    return positions[np.isfinite(positions).all(axis=(1, 2))][0]


@pytest.mark.parametrize("recording", sorted(TABLE3_RESIDUAL_MM))
def test_marker_cluster_radius_is_the_166_mm_the_manuscript_quotes(recording):
    """One rigid frame, so every recording must give the same lever arm."""
    assert cluster_rms_radius(_constellation(recording)) == pytest.approx(166.0, abs=0.1)


def test_reference_orientation_uncertainty_matches_the_manuscript():
    """0.279 mm of residual over a 166 mm cluster gives 0.36 mm and 0.07 deg.

    The Discussion builds the floor on the worst of the four recordings, which
    is Pivot. That is the conservative choice for a bound.
    """
    u = reference_uncertainty(TABLE3_RESIDUAL_MM["pivot"], _constellation("pivot"))
    assert u.n_markers == 5
    assert round(u.marker_noise_mm, 2) == 0.36
    assert round(u.orientation_deg, 2) == 0.07
    # the per-coordinate scatter is smaller by sqrt(3); conflating the two is
    # the mistake this test exists to catch
    assert u.marker_noise_mm == pytest.approx(np.sqrt(3) * u.coordinate_noise_mm)
    assert round(u.coordinate_noise_mm, 2) == 0.21


def test_the_cluster_is_essentially_isotropic():
    """The manuscript's word for a 3% spread between the principal axes."""
    axis = np.array(reference_uncertainty(0.279, _constellation("pivot")).axis_deg)
    assert axis.max() / axis.min() < 1.05
    assert np.sqrt((axis**2).sum()) == pytest.approx(
        reference_uncertainty(0.279, _constellation("pivot")).orientation_deg
    )


def test_propagation_agrees_with_a_monte_carlo_of_the_fit():
    """The analytic bound against the fit it is meant to describe.

    Perturb the real constellation by the per-coordinate noise the derivation
    infers, refit, and check that both the residual it was derived from and
    the orientation error it predicts come back.
    """
    from scipy.spatial.transform import Rotation

    nominal = _constellation("pivot")
    nominal = nominal - nominal.mean(axis=0)
    u = reference_uncertainty(0.279, nominal)

    rng = np.random.default_rng(0)
    residuals, angles = [], []
    for _ in range(4000):
        noisy = nominal + rng.normal(scale=u.coordinate_noise_mm, size=nominal.shape)
        noisy -= noisy.mean(axis=0)
        left, _, right = np.linalg.svd(nominal.T @ noisy)
        flip = np.diag([1.0, 1.0, np.sign(np.linalg.det(left @ right))])
        rotation = left @ flip @ right
        residual = noisy - nominal @ rotation
        residuals.append((residual**2).sum(axis=1).mean())
        angles.append(Rotation.from_matrix(rotation).magnitude())

    assert np.sqrt(np.mean(residuals)) == pytest.approx(0.279, rel=0.02)
    simulated = np.degrees(np.sqrt(np.mean(np.square(angles))))
    assert simulated == pytest.approx(u.orientation_deg, rel=0.02)


def test_reference_uncertainty_rejects_geometry_it_cannot_use():
    collinear = np.stack([np.arange(5.0), np.zeros(5), np.zeros(5)], axis=1)
    with pytest.raises(ValueError, match="collinear"):
        reference_uncertainty(0.279, collinear)
    with pytest.raises(ValueError, match="at least three markers"):
        reference_uncertainty(0.279, np.zeros((2, 3)))
    with pytest.raises(ValueError, match="must be positive"):
        reference_uncertainty(0.0, _constellation("pivot"))
    with pytest.raises(ValueError, match=r"must be \(n, 3\)"):
        cluster_rms_radius(np.zeros((5, 2)))
