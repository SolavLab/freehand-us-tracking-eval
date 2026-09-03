"""The golden master must reproduce the manuscript.

These tests hard-code the published values. That is deliberate: the point is to
detect any drift between the code and the paper, so the expected values must
come from the paper rather than from the code that is being tested.

Table 3 of the manuscript, "Hand-eye calibration residuals for each pipeline and
recording, within the shared evaluation window", median (IQR) and 95th
percentile.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vipose import Dataset
from vipose.metrics import summarize
from vipose.report.tables import residuals_table, to_latex
from vipose.results import read_calibration, read_residuals

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden"
DATASET = ROOT / "datasets" / "probe-tracking-2025-10-23"

# (pipeline, recording) -> frames, d_median, d_iqr, d_p95, phi_median, phi_iqr, phi_p95
TABLE3 = {
    ("zed-cuvslam", "pivot"):        (3253, 2.63, 1.76, 5.13, 0.100, 0.076, 0.213),
    ("zed-cuvslam", "mixed"):        (3520, 2.51, 1.95, 5.74, 0.115, 0.089, 0.237),
    ("zed-cuvslam", "freehand"):     (2757, 4.09, 2.85, 7.38, 0.117, 0.078, 0.223),
    ("zed-cuvslam", "pivot-repeat"): (2217, 3.11, 2.08, 5.95, 0.121, 0.074, 0.224),
    ("rs-cuvslam", "pivot"):         (1652, 3.08, 2.44, 7.78, 0.149, 0.104, 0.307),
    ("rs-cuvslam", "mixed"):         (1792, 4.10, 2.97, 8.44, 0.156, 0.131, 0.309),
    ("rs-cuvslam", "freehand"):      (1406, 3.81, 2.86, 7.66, 0.190, 0.111, 0.402),
    ("rs-cuvslam", "pivot-repeat"):  (1122, 3.03, 3.11, 9.39, 0.130, 0.115, 0.412),
    ("zed-sdk", "pivot"):            (3304, 5.26, 3.40, 9.13, 0.193, 0.142, 0.390),
    ("zed-sdk", "mixed"):            (3583, 4.08, 3.14, 9.77, 0.144, 0.109, 0.318),
    ("zed-sdk", "freehand"):         (2812, 4.77, 3.88, 9.55, 0.148, 0.108, 0.304),
    ("zed-sdk", "pivot-repeat"):     (2243, 4.32, 2.29, 9.01, 0.156, 0.106, 0.326),
}

# Table 2, "Characteristics of the probe motion in each recording": duration of
# the shared evaluation window, in seconds.
TABLE2_DURATION_S = {"pivot": 55.1, "mixed": 59.7, "freehand": 46.9, "pivot-repeat": 37.4}


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return Dataset.load(DATASET)


@pytest.mark.parametrize("key", sorted(TABLE3), ids=lambda k: f"{k[0]}-{k[1]}")
def test_cell_matches_table3(key):
    """Each cell reproduces its row of Table 3 at the precision the paper prints."""
    pipeline, recording = key
    n, dm, di, dp, am, ai, ap = TABLE3[key]
    r = read_residuals(GOLDEN / pipeline / recording / "residuals.csv")
    trans, rot = summarize(r.d_trans_mm), summarize(r.d_rot_deg)

    assert trans.n == n, f"frame count {trans.n}, Table 3 says {n}"
    for label, got, expected, dp_ in (
        ("translational median", trans.median, dm, 2),
        ("translational IQR", trans.iqr, di, 2),
        ("translational 95th", trans.p95, dp, 2),
        ("rotational median", rot.median, am, 3),
        ("rotational IQR", rot.iqr, ai, 3),
        ("rotational 95th", rot.p95, ap, 3),
    ):
        assert round(got, dp_) == expected, (
            f"{label}: {got:.6f} rounds to {round(got, dp_)}, Table 3 says {expected}"
        )


@pytest.mark.parametrize("recording", sorted(TABLE2_DURATION_S))
def test_evaluation_window_matches_table2(dataset, recording):
    """The window spans the duration Table 2 reports.

    An independent check: Table 2's durations are computed from the Vicon
    trajectory, whereas these spans come from the camera timestamps.
    """
    for pipeline in dataset.pipelines:
        r = read_residuals(GOLDEN / pipeline / recording / "residuals.csv")
        expected = TABLE2_DURATION_S[recording]
        assert abs(r.duration_s - expected) < 0.6, (
            f"{pipeline}/{recording}: window spans {r.duration_s:.2f} s, Table 2 says {expected} s"
        )


def test_all_pipelines_share_an_evaluation_window(dataset):
    """Within a recording, all three pipelines are evaluated over one window.

    This is the fairness property the unified mode exists to provide: without
    it the pipelines would be compared over different segments of the motion.

    The window is shared in the *Vicon* time base, not in absolute time. Each
    pipeline's timestamps are on its own acquisition-platform clock, and the
    three clocks are offset from the Vicon clock by different amounts (for the
    pivot recording, by -1221, -2272 and -2287 ms). So the check is on
    ``vicon_window_ms``, which must be identical, and not on the raw
    timestamps, which must not be.
    """
    for recording in dataset.recordings():
        windows = {
            p: read_calibration(GOLDEN / p / recording / "calibration.json")["vicon_window_ms"]
            for p in dataset.pipelines
        }
        first = next(iter(windows.values()))
        for pipeline, window in windows.items():
            assert window == pytest.approx(first, abs=1e-9), (
                f"{recording}: {pipeline} has Vicon window {window}, expected {first}"
            )


@pytest.mark.parametrize("recording", sorted(TABLE2_DURATION_S))
def test_local_window_is_the_vicon_window_shifted_by_the_offset(dataset, recording):
    """``local_window == vicon_window + temporal_offset``, exactly.

    Pins the sign convention of the temporal offset, which is otherwise easy to
    invert: a positive offset means the Vicon clock runs ahead of the camera
    clock. Getting it backwards would shift every reference sample by twice the
    lag while leaving the pipeline apparently working.
    """
    for pipeline in dataset.pipelines:
        c = read_calibration(GOLDEN / pipeline / recording / "calibration.json")
        offset = c["temporal_offset_ms"]
        expected = [c["vicon_window_ms"][0] + offset, c["vicon_window_ms"][1] + offset]
        assert c["local_window_ms"] == pytest.approx(expected, abs=1e-6), (
            f"{pipeline}/{recording}: local {c['local_window_ms']} != vicon + offset {expected}"
        )


def test_latex_table_is_well_formed(dataset, tmp_path):
    """The emitted LaTeX has one row per cell and the manuscript's row order."""
    table = residuals_table(dataset, GOLDEN)
    assert len(table["rows"]) == 12
    assert [r["pipeline"] for r in table["rows"][:4]] == ["zed-cuvslam"] * 4

    tex = to_latex(table)
    data_rows = [ln for ln in tex.splitlines() if ln.rstrip().endswith(r"\\")]
    assert len(data_rows) == 12, f"expected 12 data rows, got {len(data_rows)}"
    for line in data_rows:
        assert line.count("&") == 6, f"expected 7 columns: {line}"
    assert tex.count(r"\midrule") == 2
    assert tex.count(r"\addlinespace") == 3


def test_excluded_recording_is_documented(dataset):
    """The recording dropped from the paper is recorded, with its reason.

    Neither the exclusion nor the fact that one condition was acquired twice is
    currently stated in the manuscript. Keeping both in the manifest means the
    omission is at least visible to anyone reading the code.
    """
    excluded = dataset.excluded()
    assert excluded, "no excluded recordings recorded"
    for rid, meta in excluded.items():
        assert meta.get("exclusion"), f"{rid} has no stated reason"
        assert meta.get("published") is False


def test_repeat_recording_is_linked(dataset):
    repeats = {
        r: m for r, m in dataset.manifest["recordings"].items() if "repeat_of" in m
    }
    assert repeats, "the repeated acquisition is not marked as such"
    for rid, meta in repeats.items():
        assert meta["repeat_of"] in dataset.manifest["recordings"]
        assert (
            dataset.manifest["recordings"][meta["repeat_of"]]["condition"] == meta["condition"]
        ), f"{rid} claims to repeat a different condition"


# --------------------------------------------------------------------------
# Table 2 -- motion characteristics
# --------------------------------------------------------------------------

# Table 2 of the manuscript. Every quantity is computed from the Vicon
# marker-cluster trajectory inside the shared evaluation window, so it is
# independent of which pipeline is evaluated.
# recording -> duration_s, angular_extent, omega_med, omega_p95,
#              speed_med, speed_p95, path_m, shah_margin
TABLE2 = {
    "pivot":        (55.1, 108.3, 25.0, 58.6, 168, 428,  9.74, 0.031),
    "freehand":     (46.9,  77.3, 21.0, 56.8, 269, 538, 13.39, 0.029),
    "mixed":        (59.7,  98.1, 21.7, 62.1, 209, 525, 12.88, 0.036),
    "pivot-repeat": (37.4, 116.1, 23.6, 51.7, 201, 386,  6.97, 0.044),
}


@pytest.fixture(scope="module")
def motion(dataset):
    from vipose.report.tables import motion_table

    return {row["recording"]: row for row in motion_table(dataset, GOLDEN)["rows"]}


@pytest.mark.parametrize("recording", sorted(TABLE2))
def test_motion_characteristics_match_table2(motion, recording):
    """Each recording reproduces its row of Table 2 at the printed precision."""
    dur, ext, wm, wp, vm, vp, path, margin = TABLE2[recording]
    row = motion[recording]
    for label, got, expected, dp in (
        ("duration_s", row["duration_s"], dur, 1),
        ("angular_extent_deg", row["angular_extent_deg"], ext, 1),
        ("omega_median", row["omega_median_deg_s"], wm, 1),
        ("omega_p95", row["omega_p95_deg_s"], wp, 1),
        ("speed_median", row["speed_median_mm_s"], vm, 0),
        ("speed_p95", row["speed_p95_mm_s"], vp, 0),
        ("path_length_m", row["path_length_m"], path, 2),
        ("shah_margin", row["shah_margin"], margin, 3),
    ):
        assert round(got, dp) == expected, (
            f"{recording} {label}: {got:.6f} rounds to {round(got, dp)}, "
            f"Table 2 says {expected}"
        )


def test_every_recording_is_well_conditioned(motion):
    """The paper's claim that no motion was degenerate.

    Margins span 0.029 to 0.044, and the variation *between conditions*
    (0.029-0.036) is smaller than that between the two repetitions of the pivot
    condition (0.031 and 0.044) -- which is what licenses the paper's statement
    that differences in conditioning do not account for the differences in
    tracking error.
    """
    margins = {r: row["shah_margin"] for r, row in motion.items()}
    assert min(margins.values()) > 0.02, f"a motion is close to degenerate: {margins}"

    conditions = [margins[r] for r in ("pivot", "mixed", "freehand")]
    between_conditions = max(conditions) - min(conditions)
    between_repeats = abs(margins["pivot"] - margins["pivot-repeat"])
    assert between_conditions < between_repeats, (
        f"spread between conditions ({between_conditions:.4f}) is no longer smaller "
        f"than between the pivot repetitions ({between_repeats:.4f}); the paper's "
        "argument about conditioning depends on this"
    )


def test_characteristic_frequency_separates_constrained_from_free(motion):
    """The paper's |a|/|v| grouping: the two pivots against mixed and freehand."""
    f = {r: row["characteristic_frequency_hz"] for r, row in motion.items()}
    pivots = (f["pivot"], f["pivot-repeat"])
    free = (f["mixed"], f["freehand"])
    assert max(pivots) < min(free), (
        f"pivot {pivots} no longer separates from mixed/freehand {free}"
    )


def test_motion_latex_matches_the_manuscript(dataset, tmp_path):
    """The emitted Table 2 body is one row per recording, in the paper's order."""
    from vipose.report.tables import motion_table, motion_to_latex

    table = motion_table(dataset, GOLDEN)
    assert [r["recording"] for r in table["rows"]] == [
        "pivot", "freehand", "mixed", "pivot-repeat"
    ]
    tex = motion_to_latex(table)
    rows = [ln for ln in tex.splitlines() if ln.rstrip().endswith(r"\\")]
    assert len(rows) == 4
    for line in rows:
        assert line.count("&") == 8, f"expected 9 columns: {line}"
    assert tex.count(r"\addlinespace") == 1
