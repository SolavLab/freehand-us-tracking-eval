"""Each ported module must agree with the code it replaces, elementwise.

The golden-master test covers the pipeline end to end, but it needs a full
evaluation run (~30 minutes). These tests compare a replacement module directly
against its legacy counterpart on the shipped data, which takes seconds -- so a
port can be checked as it is written rather than only at the end.

They import ``vipose._legacy`` and will be deleted along with it.
"""

from __future__ import annotations

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
