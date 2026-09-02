"""A fresh evaluation must reproduce the golden master elementwise.

This is the regression gate for the port. ``tests/golden`` was produced in
Phase 0 by the original code and verified against the 2026-08-18 published
artifacts with a maximum per-frame difference of exactly zero, and against
Table 3 of the manuscript. So matching it is equivalent to matching the paper.

Comparison is elementwise on the per-frame residual series, which is far
stronger than comparing summary statistics: a median can survive substantial
rearrangement of the underlying series.

The tests need a result store. Point at one with

    VIPOSE_RESULT_STORE=results/<run-id> pytest tests/test_golden_master.py

or leave it unset and the most recent directory under ``results/`` is used. With
no store at all the tests skip rather than fail, because a fresh clone has not
run anything yet.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from vipose import Dataset
from vipose.results import read_calibration, read_residuals

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden"
DATASET = ROOT / "datasets" / "probe-tracking-2025-10-23"

# Tolerances, as set out in the port plan.
#
# Tier 1, the per-frame residuals.
#
# Not exactly zero, for one specific and understood reason: stage 3 re-derives
# each temporal offset by bounded Brent minimisation with a 0.25 ms tolerance,
# so a run can settle on a very slightly different offset and hence a very
# slightly different calibration. Measured across all twelve cells the offsets
# agree to 3.3e-08 ms and the residuals to 1.4e-08 mm and 2.5e-09 degrees --
# fourteen picometres, against residuals of a few millimetres.
#
# The bound is set four orders of magnitude above that and still eleven orders
# below anything the paper reports, so it detects a real change while tolerating
# the optimiser's convergence path. With the offset pinned instead, the chain is
# exact to 1e-12; tests/test_calibration.py checks that.
RESIDUAL_ATOL_MM = 1e-4
RESIDUAL_ATOL_DEG = 1e-5
RESIDUAL_ATOL = RESIDUAL_ATOL_MM
# Tier 2, the fitted transforms.
# The stored transforms came from the legacy text dump, which printed about nine
# significant digits, so the reference is itself truncated near 1e-7 relative.
TRANSFORM_ATOL = 1e-4
# Tier 3, the temporal offsets and evaluation windows. Treated as failure rather
# than rounding: the stage-2 objective has several near-equal minima about 12 ms
# apart, so a small shift can indicate the optimiser landed in a different one.
OFFSET_ATOL_MS = 1e-4

TRANSFORM_KEYS = (
    "T_camera_to_marker",
    "T_slamworld_to_vicon",
    "T_vicon_to_slamworld",
    "T_camera_to_marker_intrinsic",
)


def _find_store() -> Path | None:
    explicit = os.environ.get("VIPOSE_RESULT_STORE")
    if explicit:
        return Path(explicit)
    results = ROOT / "results"
    if not results.is_dir():
        return None
    candidates = [d for d in results.iterdir() if (d / "cells").is_dir()]
    return max(candidates, key=lambda d: d.stat().st_mtime) if candidates else None


STORE = _find_store()
pytestmark = pytest.mark.skipif(
    STORE is None,
    reason="no result store; run `vipose evaluate` or set VIPOSE_RESULT_STORE",
)

DATASET_OBJ = Dataset.load(DATASET, verify_hashes=False)
CELLS = DATASET_OBJ.cells()


@pytest.mark.parametrize("cell", CELLS, ids=str)
def test_residuals_match_golden_elementwise(cell):
    fresh = read_residuals(STORE / "cells" / cell.pipeline / cell.recording / "residuals.csv")
    ref = read_residuals(GOLDEN / cell.pipeline / cell.recording / "residuals.csv")

    assert len(fresh) == len(ref), f"{cell}: {len(fresh)} frames, golden has {len(ref)}"
    np.testing.assert_array_equal(
        fresh.source_frame, ref.source_frame, err_msg=f"{cell}: source_frame differs"
    )
    np.testing.assert_array_equal(
        fresh.time_ms, ref.time_ms, err_msg=f"{cell}: time_ms differs"
    )
    for name, bound in (("d_trans_mm", RESIDUAL_ATOL_MM), ("d_rot_deg", RESIDUAL_ATOL_DEG)):
        got, want = getattr(fresh, name), getattr(ref, name)
        delta = float(np.max(np.abs(got - want)))
        assert delta <= bound, (
            f"{cell}: {name} differs from the golden master by up to {delta:.3e} "
            f"(worst at frame {int(np.argmax(np.abs(got - want)))})"
        )


@pytest.mark.parametrize("cell", CELLS, ids=str)
def test_calibration_matches_golden(cell):
    fresh = read_calibration(STORE / "cells" / cell.pipeline / cell.recording / "calibration.json")
    ref = read_calibration(GOLDEN / cell.pipeline / cell.recording / "calibration.json")

    for key in TRANSFORM_KEYS:
        delta = float(np.max(np.abs(fresh[key] - ref[key])))
        assert delta <= TRANSFORM_ATOL, f"{cell}: {key} differs by up to {delta:.3e}"

    # The camera-to-marker distance is invariant to the choice of coordinate
    # frame, and the paper quotes its cross-condition spread as the floor below
    # which differences are not interpretable. Worth checking on its own.
    norm = lambda c: float(np.linalg.norm(np.asarray(c["T_camera_to_marker"])[:3, 3]))  # noqa: E731
    assert abs(norm(fresh) - norm(ref)) < 1e-6, f"{cell}: |t_camera_to_marker| moved"

    assert abs(fresh["temporal_offset_ms"] - ref["temporal_offset_ms"]) <= OFFSET_ATOL_MS, (
        f"{cell}: temporal offset {fresh['temporal_offset_ms']} vs golden "
        f"{ref['temporal_offset_ms']}"
    )
    for field in ("vicon_window_ms", "local_window_ms"):
        np.testing.assert_allclose(
            fresh[field], ref[field], atol=OFFSET_ATOL_MS, err_msg=f"{cell}: {field} differs"
        )


@pytest.mark.parametrize("cell", CELLS, ids=str)
def test_offset_refinement_stages_match_golden(cell):
    fresh = read_calibration(STORE / "cells" / cell.pipeline / cell.recording / "calibration.json")
    ref = read_calibration(GOLDEN / cell.pipeline / cell.recording / "calibration.json")
    a, b = fresh["offset_refinement_stages"], ref["offset_refinement_stages"]

    assert set(a) == set(b), f"{cell}: stage keys differ: {set(a) ^ set(b)}"
    for key, want in b.items():
        got = a[key]
        if isinstance(want, bool) or not isinstance(want, (int, float)):
            assert got == want, f"{cell}: stage {key} is {got!r}, golden has {want!r}"
        else:
            assert abs(got - want) <= OFFSET_ATOL_MS, (
                f"{cell}: stage {key} is {got}, golden has {want}"
            )


def test_offsets_file_matches_golden():
    """The pinned offsets stay pinned."""
    pinned = json.loads((GOLDEN / "offsets.json").read_text())
    for cell in CELLS:
        fresh = read_calibration(
            STORE / "cells" / cell.pipeline / cell.recording / "calibration.json"
        )
        assert abs(fresh["temporal_offset_ms"] - pinned[cell.pipeline][cell.recording]) <= (
            OFFSET_ATOL_MS
        ), f"{cell}: offset drifted from tests/golden/offsets.json"


def test_shared_window_is_identical_across_pipelines():
    """Re-checked on the fresh run, not only on the frozen copy."""
    for recording in DATASET_OBJ.recordings():
        windows = [
            read_calibration(STORE / "cells" / p / recording / "calibration.json")[
                "vicon_window_ms"
            ]
            for p in DATASET_OBJ.pipelines
        ]
        assert all(w == windows[0] for w in windows), (
            f"{recording}: pipelines disagree on the shared window: {windows}"
        )
