"""Each ported module must agree with the code it replaces, elementwise.

The golden-master test covers the pipeline end to end, but it needs a full
evaluation run (~30 minutes). These tests compare a replacement module directly
against its legacy counterpart on the shipped data, which takes seconds -- so a
port can be checked as it is written rather than only at the end.

They import ``vipose._legacy`` and will be deleted along with it.
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

from vipose import Dataset  # noqa: E402
from vipose.geometry.rigid_body import fit_rigid_body  # noqa: E402
from vipose.io.mocap import load_mocap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATASET = Dataset.load(ROOT / "datasets" / "probe-tracking-2025-10-23", verify_hashes=False)
RECORDINGS = DATASET.recordings()


@pytest.fixture(scope="module")
def legacy():
    """The original loader and fit, imported lazily.

    Importing this package selects a matplotlib backend and sets Qt environment
    variables as a side effect, which is one of the reasons it is being
    replaced.
    """
    from vipose._legacy import SynchronizationNew as sn

    return sn


@functools.cache
def _load_new(recording):
    return load_mocap(
        DATASET.reference_csv(recording),
        marker_prefix=DATASET.marker_prefix,
        expected_rate_hz=DATASET.vicon_rate_hz,
        expected_markers=len(DATASET.marker_order),
    )


@pytest.mark.parametrize("recording", RECORDINGS)
def test_load_mocap_matches_legacy(recording, legacy):
    """Marker positions, frame numbers and timestamps are identical."""
    new = _load_new(recording)
    old = legacy.load_mocap(str(DATASET.reference_csv(recording)))

    assert len(new) == len(old.Frame)
    assert new.names == tuple(m.name for m in old.Marker)
    np.testing.assert_array_equal(new.frame, old.Frame)
    np.testing.assert_allclose(new.time_ms, old.Time_ms, rtol=0, atol=0)
    for i, marker in enumerate(new.markers):
        np.testing.assert_array_equal(marker.X, old.Marker[i].X, err_msg=f"{marker.name} X")
        np.testing.assert_array_equal(marker.Y, old.Marker[i].Y, err_msg=f"{marker.name} Y")
        np.testing.assert_array_equal(marker.Z, old.Marker[i].Z, err_msg=f"{marker.name} Z")


@pytest.mark.parametrize("recording", RECORDINGS)
def test_rigid_body_fit_matches_legacy(recording, legacy):
    """Pose and residual agree to floating-point equality.

    Both implementations run the same per-frame SVD in the same order, so the
    tolerance is exact rather than merely tight.
    """
    new = fit_rigid_body(_load_new(recording), min_markers=4)
    old_mocap = legacy.load_mocap(str(DATASET.reference_csv(recording)))
    df, _ = legacy.calculate_vicon_rigid_body(old_mocap, min_markers=4, print_stats=False)

    np.testing.assert_array_equal(new.translation[:, 0], df["Translation_X"].to_numpy())
    np.testing.assert_array_equal(new.translation[:, 1], df["Translation_Y"].to_numpy())
    np.testing.assert_array_equal(new.translation[:, 2], df["Translation_Z"].to_numpy())
    np.testing.assert_array_equal(new.rotvec[:, 0], df["Rotation_X"].to_numpy())
    np.testing.assert_array_equal(new.rotvec[:, 1], df["Rotation_Y"].to_numpy())
    np.testing.assert_array_equal(new.rotvec[:, 2], df["Rotation_Z"].to_numpy())
    np.testing.assert_array_equal(new.residual_mm, df["Residuals"].to_numpy())


@pytest.mark.parametrize("recording", RECORDINGS)
def test_mocap_is_immutable(recording):
    """Shifting or cropping the reference must not disturb the original.

    The legacy resampler mutated the caller's ``Time_ms`` in place. That was
    safe only because the CSV was re-read for every pipeline; sharing one load
    across the three pipelines of a recording -- the obvious optimisation --
    would have shifted the second pipeline's clock by the first's lag and
    changed every published number without raising anything.
    """
    m = _load_new(recording)
    before = m.time_ms.copy()

    shifted = m.shifted(-1234.5)
    np.testing.assert_array_equal(m.time_ms, before)
    np.testing.assert_allclose(shifted.time_ms, before - 1234.5)

    lo, hi = before[len(before) // 4], before[len(before) // 2]
    cropped = m.cropped(lo, hi)
    np.testing.assert_array_equal(m.time_ms, before)
    assert len(cropped) < len(m)
    assert cropped.time_ms[0] >= lo and cropped.time_ms[-1] <= hi
    assert all(len(mk.xyz) == len(cropped) for mk in cropped.markers)


def test_sample_rate_disagreement_is_an_error():
    """The manifest rate is authoritative, and a mismatch is loud."""
    with pytest.raises(ValueError, match="file declares"):
        load_mocap(
            DATASET.reference_csv(RECORDINGS[0]),
            marker_prefix=DATASET.marker_prefix,
            expected_rate_hz=120.0,
        )


def test_wrong_marker_count_is_an_error():
    with pytest.raises(ValueError, match="markers matching"):
        load_mocap(
            DATASET.reference_csv(RECORDINGS[0]),
            marker_prefix=DATASET.marker_prefix,
            expected_rate_hz=DATASET.vicon_rate_hz,
            expected_markers=99,
        )


def test_unknown_marker_prefix_is_an_error():
    """The original would silently find nothing and fail later."""
    with pytest.raises(ValueError, match="no marker names"):
        load_mocap(
            DATASET.reference_csv(RECORDINGS[0]),
            marker_prefix="NOT_A_SUBJECT:",
            expected_rate_hz=DATASET.vicon_rate_hz,
        )


# --------------------------------------------------------------------------
# vipose.io.tracks, vipose.geometry.transforms, vipose.kinematics
# --------------------------------------------------------------------------

from vipose.geometry import transforms  # noqa: E402
from vipose.io.tracks import load_track  # noqa: E402
from vipose.kinematics import angular_speed, rotational_increments  # noqa: E402
from vipose.recordings import Cell  # noqa: E402

CELLS = DATASET.cells()


@pytest.mark.parametrize("cell", CELLS, ids=str)
def test_load_track_matches_legacy(cell, legacy):
    """Pose columns and the derived relative clock are identical."""
    path = DATASET.tracking_csv(cell)
    new = load_track(path)
    old = legacy.get_df_ZED(str(path))

    assert len(new) == len(old)
    np.testing.assert_array_equal(new.frame, old["Frame"].to_numpy())
    np.testing.assert_array_equal(new.time_ms, old["Time_ms"].to_numpy())
    for i, axis in enumerate("XYZ"):
        np.testing.assert_array_equal(
            new.translation[:, i], old[f"Translation_{axis}"].to_numpy()
        )
        np.testing.assert_array_equal(new.rotvec[:, i], old[f"Rotation_{axis}"].to_numpy())


@pytest.mark.parametrize("cell", CELLS, ids=str)
def test_transform_roundtrip_matches_legacy(cell, legacy):
    """to_matrices/from_matrices agree with df_to_T/T_to_df."""
    path = DATASET.tracking_csv(cell)
    new = load_track(path)
    old = legacy.get_df_ZED(str(path))

    mats_new = transforms.to_matrices(new.translation, new.rotvec)
    mats_old = np.stack(legacy.df_to_T(old))
    np.testing.assert_allclose(mats_new, mats_old, rtol=0, atol=0)

    # and back again
    t, rv = transforms.from_matrices(mats_new)
    df_old = legacy.T_to_df(list(mats_old))
    np.testing.assert_allclose(t[:, 0], df_old["Translation_X"].to_numpy(), rtol=0, atol=0)
    np.testing.assert_allclose(rv[:, 2], df_old["Rotation_Z"].to_numpy(), rtol=0, atol=1e-12)

    rot, trans = transforms.split_rt(mats_new)
    t_old, r_old = legacy.split_rt(list(mats_old))
    np.testing.assert_array_equal(trans, np.stack(t_old))
    np.testing.assert_array_equal(rot, np.stack(r_old))


@pytest.mark.parametrize("cell", CELLS, ids=str)
def test_angular_speed_matches_legacy(cell, legacy):
    """The synchronization objective's input series is identical.

    This one matters most of the three: stages 1 and 2 align on exactly this
    signal, and its minimum is shallow -- several near-equal troughs about 12 ms
    apart -- so any difference here could move a published temporal offset.
    """
    path = DATASET.tracking_csv(cell)
    new = load_track(path)
    old = legacy.get_df_ZED(str(path))

    t_new, w_new = angular_speed(new.time_ms, new.rotvec)
    t_old, w_old = legacy.angular_speed_magnitude_from_rotvec(old)

    np.testing.assert_array_equal(t_new, t_old)
    np.testing.assert_allclose(w_new, w_old, rtol=0, atol=0)

    inc_new = rotational_increments(new.rotvec)
    inc_old = legacy.rotational_increments_from_rotvec(old)
    np.testing.assert_allclose(inc_new, inc_old, rtol=0, atol=0)


@pytest.mark.parametrize("recording", RECORDINGS)
def test_rigid_body_angular_speed_matches_legacy(recording, legacy):
    """Same, on the reference side, where the fit feeds the objective."""
    new_rb = fit_rigid_body(_load_new(recording), min_markers=4)
    old_mocap = legacy.load_mocap(str(DATASET.reference_csv(recording)))
    df, _ = legacy.calculate_vicon_rigid_body(old_mocap, min_markers=4, print_stats=False)

    t_new, w_new = angular_speed(new_rb.time_ms, new_rb.rotvec)
    t_old, w_old = legacy.angular_speed_magnitude_from_rotvec(df)
    np.testing.assert_array_equal(t_new, t_old)
    np.testing.assert_allclose(w_new, w_old, rtol=0, atol=0)


def test_angular_speed_units_are_rad_per_millisecond():
    """A constant 1 rad/s about one axis, sampled in milliseconds.

    Pins the unit, because the original's comment said radians per second while
    the arithmetic divided by an interval in milliseconds.
    """
    t = np.arange(0.0, 1000.0, 10.0)                    # ms
    rotvec = np.zeros((len(t), 3))
    rotvec[:, 2] = t / 1000.0                            # 1 rad after 1000 ms
    _, omega = angular_speed(t, rotvec)
    np.testing.assert_allclose(omega, 1e-3, rtol=1e-9)


def test_pose_continuity_reproduces_the_papers_method():
    """Window-independent continuity counts match the manuscript.

    The paper reports, within the evaluation windows, zero repeated poses for
    ZED-cuVSLAM and 40 for RS-cuVSLAM. Both are window-independent here -- every
    RS repetition falls inside its window -- so they must reproduce exactly on
    the full tracks.
    """
    counts = {}
    for pipeline in DATASET.pipelines:
        repeated = omitted = 0
        for recording in RECORDINGS:
            track = load_track(DATASET.tracking_csv(Cell(pipeline, recording)))
            repeated += len(track.repeated_poses())
            omitted += track.omitted_frames()
        counts[pipeline] = (repeated, omitted)

    assert counts["zed-cuvslam"][0] == 0, "the paper reports no repeated poses for ZED-cuVSLAM"
    assert counts["rs-cuvslam"][0] == 40, "the paper reports 40 repeated poses for RS-cuVSLAM"
    assert counts["rs-cuvslam"][1] == 0, "the paper reports no omissions for RS-cuVSLAM"
    # Counts over the full track can only exceed the in-window counts.
    assert counts["zed-sdk"][0] >= 54
    assert counts["zed-cuvslam"][1] >= 192


# --------------------------------------------------------------------------
# vipose.sync
# --------------------------------------------------------------------------

from vipose.sync.resample import resample_reference, shared_window  # noqa: E402
from vipose.sync.stages import (  # noqa: E402
    crosscorrelation_offset,
    omega_mismatch_rms,
    refine_offset_omega,
)

# Stage 1 and 2 offsets recorded in the published metadata_unified.json files,
# taken from the paper's run rather than from this code.
PUBLISHED_STAGES_MS = {
    ("zed-sdk", "pivot"): (-1216.9607695300306, -1209.5298968807706),
    ("zed-cuvslam", "pivot"): (-2289.9798086565406, -2271.948481395),
    ("rs-cuvslam", "pivot"): (-2287.2256130736, -2290.1720695),
}


@functools.cache
def _sync_inputs(cell):
    """Cached: the rigid-body fit is ~14k per-frame SVDs per recording."""
    track = load_track(DATASET.tracking_csv(cell))
    mocap = _load_new(cell.recording)
    rb = fit_rigid_body(mocap, min_markers=4)
    return track, mocap, rb


@functools.cache
def _legacy_sync_inputs_cached(cell):
    import contextlib
    import io

    from vipose._legacy import SynchronizationNew as sn

    old_track = sn.get_df_ZED(str(DATASET.tracking_csv(cell)))
    old_mocap = sn.load_mocap(str(DATASET.reference_csv(cell.recording)))
    with contextlib.redirect_stdout(io.StringIO()):
        old_rb, _ = sn.calculate_vicon_rigid_body(old_mocap, min_markers=4, print_stats=False)
    return old_track, old_mocap, old_rb


def _legacy_sync_inputs(cell, legacy):
    # The legacy resampler mutates the mocap object it is given, so hand out a
    # fresh copy of that one while still caching the expensive fit.
    old_track, old_mocap, old_rb = _legacy_sync_inputs_cached(cell)
    fresh_mocap = legacy.load_mocap(str(DATASET.reference_csv(cell.recording)))
    return old_track, fresh_mocap, old_rb


@pytest.mark.parametrize("cell", CELLS, ids=str)
def test_stage1_offset_matches_legacy(cell, legacy):
    import contextlib
    import io

    track, _, rb = _sync_inputs(cell)
    new = crosscorrelation_offset(track.time_ms, track.rotvec, rb.time_ms, rb.rotvec)

    old_track, _, old_rb = _legacy_sync_inputs(cell, legacy)
    with contextlib.redirect_stdout(io.StringIO()):
        old = legacy.estimate_temporal_offset_zed_vicon(old_track, old_rb, 0.0)
    assert new == old, f"{cell}: stage 1 {new} vs legacy {old}"


@pytest.mark.parametrize("cell", CELLS, ids=str)
def test_stage2_objective_matches_legacy(cell, legacy):
    """The objective itself, on a grid, not just its argmin.

    Comparing minima would hide a difference in objective shape, and the shape
    is what matters: the minimum is shallow, with several near-equal troughs
    spanning about 12 ms, so an optimiser can move between them under a
    perturbation far smaller than the difference in reported offset.
    """
    track, _, rb = _sync_inputs(cell)
    old_track, _, old_rb = _legacy_sync_inputs(cell, legacy)

    centre = crosscorrelation_offset(track.time_ms, track.rotvec, rb.time_ms, rb.rotvec)
    for delta in (-20.0, -8.0, -2.0, 0.0, 2.0, 8.0, 20.0):
        offset = centre + delta
        new = omega_mismatch_rms(track.time_ms, track.rotvec, rb.time_ms, rb.rotvec, offset)
        old = legacy.synchronize_zed_vicon_trial(old_track, old_rb, offset, discard_fraction=0.0)
        assert new == old, f"{cell}: objective at {offset:+.1f} ms is {new} vs legacy {old}"


@pytest.mark.parametrize("cell", CELLS, ids=str)
def test_stage2_refined_offset_matches_legacy(cell, legacy):
    from scipy.optimize import fmin

    track, _, rb = _sync_inputs(cell)
    start = crosscorrelation_offset(track.time_ms, track.rotvec, rb.time_ms, rb.rotvec)
    new = refine_offset_omega(track.time_ms, track.rotvec, rb.time_ms, rb.rotvec, start)

    old_track, _, old_rb = _legacy_sync_inputs(cell, legacy)
    old = float(
        fmin(
            lambda x: legacy.synchronize_zed_vicon_trial(
                old_track, old_rb, x, discard_fraction=0.0
            ),
            start,
            maxiter=20,
            xtol=1e-5,
            disp=False,
        )[0]
    )
    assert new == old, f"{cell}: stage 2 {new} vs legacy {old}"


@pytest.mark.parametrize("key", sorted(PUBLISHED_STAGES_MS), ids=lambda k: f"{k[0]}-{k[1]}")
def test_stages_reproduce_the_published_offsets(key):
    """Stages 1 and 2 land on the values recorded in the paper's own run."""
    cell = Cell(*key)
    expected_stage1, expected_stage2 = PUBLISHED_STAGES_MS[key]
    track, _, rb = _sync_inputs(cell)

    stage1 = crosscorrelation_offset(track.time_ms, track.rotvec, rb.time_ms, rb.rotvec)
    assert stage1 == pytest.approx(expected_stage1, abs=1e-6)
    stage2 = refine_offset_omega(track.time_ms, track.rotvec, rb.time_ms, rb.rotvec, stage1)
    assert stage2 == pytest.approx(expected_stage2, abs=1e-6)


@pytest.mark.parametrize("cell", CELLS, ids=str)
def test_resample_reference_matches_legacy(cell, legacy):
    """Marker resampling is linear, and identical to the original.

    Appendix A states the marker trajectories were resampled by cubic spline.
    They were not: the original uses np.interp here, and the only cubic
    interpolation in that implementation is in the stage-2 angular-speed
    objective. This test pins the behaviour that produced the published numbers.
    """
    # Use this cell's published offset and evaluation window -- the real
    # configuration. An arbitrary offset can push the camera timestamps outside
    # the reference span, where the original silently clamped via np.interp and
    # this port raises instead; comparing there would compare a bug to its fix.
    golden = json.loads(
        (ROOT / "tests" / "golden" / cell.pipeline / cell.recording
         / "calibration.json").read_text()
    )
    offset = golden["temporal_offset_ms"]
    lo, hi = golden["local_window_ms"]

    track, mocap, rb = _sync_inputs(cell)
    cropped = track.cropped(lo, hi)
    new = resample_reference(mocap, cropped.time_ms, offset)

    old_track, old_mocap, old_rb = _legacy_sync_inputs(cell, legacy)
    old_cropped = old_track[
        (old_track["Time_ms"] >= lo) & (old_track["Time_ms"] <= hi)
    ].reset_index(drop=True)
    old = legacy.synchronize_zed_vicon(old_cropped, old_rb, old_mocap, offset)

    assert len(new) == len(old.Frame) == len(cropped)
    np.testing.assert_allclose(new.time_ms, old.Time_ms, rtol=0, atol=0)
    np.testing.assert_allclose(new.frame, old.Frame, rtol=0, atol=0)
    for i, marker in enumerate(new.markers):
        np.testing.assert_allclose(marker.X, old.Marker[i].X, rtol=0, atol=0)
        np.testing.assert_allclose(marker.Y, old.Marker[i].Y, rtol=0, atol=0)
        np.testing.assert_allclose(marker.Z, old.Marker[i].Z, rtol=0, atol=0)


def test_resample_refuses_to_extrapolate():
    """np.interp clamps silently; this must not."""
    mocap = _load_new(RECORDINGS[0])
    with pytest.raises(ValueError, match="outside the reference span"):
        resample_reference(mocap, np.array([mocap.time_ms[-1] + 5000.0]), 0.0)


def test_shared_window_intersects():
    assert shared_window({"a": (0.0, 10.0), "b": (2.0, 8.0), "c": (1.0, 9.0)}) == (2.0, 8.0)
    with pytest.raises(ValueError, match="no common window"):
        shared_window({"a": (0.0, 1.0), "b": (5.0, 6.0)})


@pytest.mark.parametrize("cell", CELLS, ids=str)
def test_cropping_a_track_preserves_its_clock_origin(cell):
    """The relative clock must stay anchored to the uncropped first sample.

    Found while porting: with the origin recomputed from the current first
    sample, cropping re-zeroed the clock and shifted every timestamp earlier --
    by 67 ms for zed-cuvslam/pivot, four frames. Since the evaluation window is
    expressed in this clock, that would have silently mis-placed the window and
    resampled the reference at the wrong instants.
    """
    track = load_track(DATASET.tracking_csv(cell))
    lo = float(track.time_ms[len(track) // 4])
    hi = float(track.time_ms[len(track) // 2])

    cropped = track.cropped(lo, hi)
    assert cropped.origin_ms == track.origin_ms
    assert cropped.time_ms[0] >= lo
    assert cropped.time_ms[-1] <= hi
    # the surviving samples keep the times they had before cropping
    keep = (track.time_ms >= lo) & (track.time_ms <= hi)
    np.testing.assert_array_equal(cropped.time_ms, track.time_ms[keep])
    np.testing.assert_array_equal(cropped.timestamp_ms, track.timestamp_ms[keep])
    # and cropping does not disturb the original
    assert len(track) > len(cropped)
