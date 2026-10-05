"""Emit the manuscript's tables as LaTeX and JSON.

Tables are generated, never hand-typed. That is not a convenience: in this
project a median rotational residual in degrees was once copied into the
manuscript as a cross-correlation coefficient, and survived several drafts. A
number that is transcribed by hand is a number that can drift from the analysis
that produced it.

The LaTeX emitted here is the table *body* -- the rows between ``\\midrule`` and
``\\bottomrule`` -- so it can be diffed against the manuscript directly.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np

from ..metrics import (
    bootstrap_difference_standard_error,
    bootstrap_standard_error,
    summarize,
)
from ..recordings import Cell, Dataset
from ..results import read_residuals

__all__ = [
    "residuals_table",
    "write_residuals_table",
    "motion_table",
    "write_motion_table",
    "reference_fit_table",
    "write_reference_fit_table",
]

# Row order follows the manuscript, which groups by recording -- so the three
# pipelines of one condition sit together and can be read against each other --
# with the repeated pivot acquisition set apart from the three conditions. The
# pipeline order within a group is the numbering of the manuscript's Table 1.
PIPELINE_ORDER = ["zed-sdk", "zed-cuvslam", "rs-cuvslam"]
CONDITION_ORDER = ["pivot", "mixed", "freehand"]
REPEAT = "pivot-repeat"

# Precision the manuscript prints at. Translational residuals are quoted to
# 0.01 mm and rotational to 0.001 deg; emitting more would imply a resolution
# the measurement does not have.
MM_DP = 2
DEG_DP = 3

# The confidence intervals the manuscript prints. 1.96 standard deviations of
# the statistic over this many moving-block bootstrap replicates; the seed is
# derived per cell so that a value never depends on the order in which the
# cells happen to be computed.
REPLICATES = 20000
Z = 1.96


def _p95(a, axis):
    return np.percentile(a, 95, axis=axis)


def _half_width(values, times_s, statistic, key: str) -> float:
    """Half-width of the 95% interval for one statistic of one cell."""
    se = bootstrap_standard_error(
        values, times_s, statistic, replicates=REPLICATES, seed=zlib.crc32(key.encode())
    )
    return Z * se


def residuals_table(dataset: Dataset, store: str | Path) -> dict:
    """Compute the residuals table from a result store or golden master.

    ``store`` is a directory containing ``<pipeline>/<recording>/residuals.csv``
    (the golden-master layout) or ``cells/<pipeline>/<recording>/residuals.csv``
    (a run's layout); both are accepted.
    """
    store = Path(store)
    rows, cells = [], {}
    for recording in CONDITION_ORDER + [REPEAT]:
        for pipeline in PIPELINE_ORDER:
            cell = Cell(pipeline, recording)
            r = read_residuals(_residuals_path(store, cell))
            t = r.time_ms / 1000.0
            cells[cell] = {
                "translational_mm": (r.d_trans_mm, t),
                "rotational_deg": (r.d_rot_deg, t),
            }
            trans, rot = summarize(r.d_trans_mm), summarize(r.d_rot_deg)
            td, rd = trans.as_dict(), rot.as_dict()
            for d, series, unit in ((td, r.d_trans_mm, "trans"), (rd, r.d_rot_deg, "rot")):
                for name, fn in (("median", np.median), ("p95", _p95)):
                    d[f"{name}_half_width"] = _half_width(
                        series, t, fn, f"{pipeline}/{recording}/{unit}/{name}"
                    )
            rows.append(
                {
                    "pipeline": pipeline,
                    "pipeline_label": dataset.label(pipeline=pipeline),
                    "recording": recording,
                    "recording_label": dataset.label(recording=recording),
                    "is_repeat": recording == REPEAT,
                    "frames": trans.n,
                    "translational_mm": td,
                    "rotational_deg": rd,
                }
            )
    _mark_best_and_separated(rows, cells)
    return {"table": "residuals", "dataset": dataset.id, "rows": rows}


def _mark_best_and_separated(rows: list[dict], series: dict) -> None:
    """Flag the lowest value of each recording, and whether it stands apart.

    Two marks, because they answer different questions. The lowest value says
    which pipeline came out ahead in that cell; separation says whether the
    data support the claim. A value can be the best without the comparison
    resolving, which is the common case here, so marking only the best would
    overstate and marking only the separated ones would hide the ordering.

    Separation is judged on the difference itself: the two cells are
    resampled together and the best is held separated from another pipeline
    only where the 95% interval for their difference excludes zero. Asking
    instead whether the two printed intervals overlap would be a stricter
    test than the stated level -- standard errors combine in quadrature, not
    linearly -- and would reject differences the data do establish.

    Separation is against *both* other pipelines, a stricter test than
    winning one pairwise comparison.
    """
    for row in rows:
        row["best"] = {}
        row["separated"] = {}
    by_recording: dict[str, list[dict]] = {}
    for row in rows:
        by_recording.setdefault(row["recording"], []).append(row)
    for recording, group in by_recording.items():
        for component in ("translational_mm", "rotational_deg"):
            for name, fn in (("median", np.median), ("p95", _p95)):
                key = f"{component}:{name}"
                best = min(group, key=lambda row, _c=component, _n=name: row[_c][_n])
                best["best"][key] = True
                best["separated"][key] = all(
                    _separated(best, other, series, recording, component, name, fn)
                    for other in group
                    if other is not best
                )


def _separated(best: dict, other: dict, series: dict, recording, component, name, fn) -> bool:
    """Does the 95% interval for the difference between two cells exclude zero?"""
    a_values, a_times = series[Cell(best["pipeline"], recording)][component]
    b_values, b_times = series[Cell(other["pipeline"], recording)][component]
    pair = "/".join(sorted((best["pipeline"], other["pipeline"])))
    se = bootstrap_difference_standard_error(
        a_values,
        a_times,
        b_values,
        b_times,
        fn,
        replicates=REPLICATES,
        seed=zlib.crc32(f"{pair}/{recording}/{component}/{name}".encode()),
    )
    return abs(other[component][name] - best[component][name]) > Z * se


def to_latex(table: dict) -> str:
    """Render the table body as LaTeX rows."""
    # Pipeline macros as the manuscript defines them.
    macro = {
        "zed-cuvslam": r"\zedCuVSLAM{}",
        "rs-cuvslam": r"\rsCuVSLAM{}",
        "zed-sdk": r"\zedSDK{}",
    }

    def value(row, component, name, dp):
        d = row[component]
        key = f"{component}:{name}"
        body = f"{d[name]:.{dp}f}~$\\pm$~{d[f'{name}_half_width']:.{dp}f}"
        if row["best"].get(key):
            body = rf"\textbf{{{body}}}"
            if row["separated"].get(key):
                body = rf"\underline{{{body}}}"
        return body

    lines, previous = [], None
    for row in table["rows"]:
        if previous is not None and row["recording"] != previous:
            lines.append(r"\midrule" if row["is_repeat"] else r"\addlinespace")
        first = row["recording"] != previous
        lines.append(
            f"{row['recording_label'] if first else '':<16s} & "
            f"{macro.get(row['pipeline'], row['pipeline_label']):<14s} & {row['frames']:5d} & "
            f"{value(row, 'translational_mm', 'median', MM_DP):<34s} & "
            f"{value(row, 'translational_mm', 'p95', MM_DP):<34s} & "
            f"{value(row, 'rotational_deg', 'median', DEG_DP):<36s} & "
            f"{value(row, 'rotational_deg', 'p95', DEG_DP):<36s} \\\\"
        )
        previous = row["recording"]
    return "\n".join(lines) + "\n"


def write_residuals_table(dataset: Dataset, store: str | Path, out_dir: str | Path) -> dict:
    """Write ``residuals.tex`` and ``residuals.json`` into ``out_dir``."""
    import json

    table = residuals_table(dataset, store)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "residuals.tex").write_text(to_latex(table))
    with open(out / "residuals.json", "w") as f:
        json.dump(table, f, indent=2)
        f.write("\n")
    return table


def _residuals_path(store: Path, cell: Cell) -> Path:
    for candidate in (
        store / cell.pipeline / cell.recording / "residuals.csv",
        store / "cells" / cell.pipeline / cell.recording / "residuals.csv",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no residuals.csv for {cell} under {store}")


# --------------------------------------------------------------------------
# Motion characteristics (the manuscript's Table 2)
# --------------------------------------------------------------------------

MOTION_ORDER = ["pivot", "freehand", "mixed"]  # the manuscript's row order
MOTION_REPEAT = "pivot-repeat"


def motion_table(dataset: Dataset, store: str | Path) -> dict:
    """Compute the motion-characteristics table from a result store.

    Every quantity comes from the Vicon marker-cluster trajectory inside the
    shared evaluation window, so the table is independent of which pipeline is
    evaluated -- and the code asserts that, rather than assuming it.
    """
    import json

    from ..geometry.rigid_body import fit_rigid_body
    from ..io.mocap import load_mocap
    from ..motion import characterise

    store = Path(store)
    rows = []
    for recording in MOTION_ORDER + [MOTION_REPEAT]:
        windows = set()
        for pipeline in dataset.pipelines:
            calib = _calibration_path(store, pipeline, recording)
            windows.add(tuple(json.loads(calib.read_text())["vicon_window_ms"]))
        if len(windows) != 1:
            raise ValueError(
                f"{recording}: the evaluation window differs between pipelines "
                f"({windows}); the motion characterisation would not be "
                "pipeline-independent"
            )
        lo, hi = windows.pop()

        mocap = load_mocap(
            dataset.reference_csv(recording),
            marker_prefix=dataset.marker_prefix,
            expected_rate_hz=dataset.vicon_rate_hz,
        )
        rb = fit_rigid_body(mocap, min_markers=4)
        keep = (rb.time_ms >= lo) & (rb.time_ms <= hi)
        rows.append(
            characterise(
                recording,
                rb.time_ms[keep],
                rb.translation[keep],
                rb.rotvec[keep],
                sample_rate_hz=dataset.vicon_rate_hz,
            ).as_dict()
            | {"label": dataset.label(recording=recording), "is_repeat": recording == MOTION_REPEAT}
        )
    return {"table": "motion", "dataset": dataset.id, "rows": rows}


def motion_to_latex(table: dict) -> str:
    lines = []
    for row in table["rows"]:
        if row["is_repeat"]:
            lines.append(r"\addlinespace")
        lines.append(
            f"{row['label']:<16s} & {row['duration_s']:.1f} & "
            f"{row['angular_extent_deg']:.1f} & {row['omega_median_deg_s']:.1f} & "
            f"{row['omega_p95_deg_s']:.1f} & {row['speed_median_mm_s']:.0f} & "
            f"{row['speed_p95_mm_s']:.0f} & {row['path_length_m']:5.2f} & "
            f"{row['shah_margin']:.3f} \\\\"
        )
    return "\n".join(lines) + "\n"


def write_motion_table(dataset: Dataset, store: str | Path, out_dir: str | Path) -> dict:
    import json

    table = motion_table(dataset, store)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "motion.tex").write_text(motion_to_latex(table))
    with open(out / "motion.json", "w") as f:
        json.dump(table, f, indent=2)
        f.write("\n")
    return table


def _calibration_path(store: Path, pipeline: str, recording: str) -> Path:
    for candidate in (
        store / pipeline / recording / "calibration.json",
        store / "cells" / pipeline / recording / "calibration.json",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no calibration.json for {pipeline}/{recording} under {store}")


# --------------------------------------------------------------------------
# Reference quality (the manuscript's Vicon rigid-body residual table)
# --------------------------------------------------------------------------


def reference_fit_table(dataset: Dataset, store: str | Path) -> dict:
    """Per-frame RMS residual of the Kabsch fit, within the evaluation window.

    Computed on the motion-capture system's **native 240 Hz samples**, not on
    the trajectory resampled to camera timestamps, since the table characterises
    the motion-capture system rather than the evaluation grid.

    The window is applied **before** the fit, not after. That is the whole
    subtlety: the Kabsch fit measures each frame's marker constellation against
    a reference constellation taken from the first frame it is given, so
    cropping afterwards leaves the reference at the first frame of the *whole
    recording* and reports residuals against a constellation from outside the
    window. The difference is not small and does not have a consistent sign --
    0.244 against 0.279 mm for the pivot recording, 0.319 against 0.265 for its
    repeat.

    (`cell.json` reports a different quantity: the fit residual on the resampled
    trajectory, describing the samples that actually entered that cell's
    residuals.)
    """
    import json

    from ..geometry.rigid_body import fit_rigid_body
    from ..io.mocap import load_mocap
    from ..metrics import summarize

    store = Path(store)
    rows = []
    for recording in MOTION_ORDER + [MOTION_REPEAT]:
        windows = {
            tuple(json.loads(_calibration_path(store, p, recording).read_text())["vicon_window_ms"])
            for p in dataset.pipelines
        }
        if len(windows) != 1:
            raise ValueError(f"{recording}: evaluation window differs between pipelines")
        lo, hi = windows.pop()

        mocap = load_mocap(
            dataset.reference_csv(recording),
            marker_prefix=dataset.marker_prefix,
            expected_rate_hz=dataset.vicon_rate_hz,
        )
        rb = fit_rigid_body(mocap.cropped(lo, hi), min_markers=4)
        stats = summarize(rb.residual_mm[rb.valid])
        rows.append(
            {
                "recording": recording,
                "label": dataset.label(recording=recording),
                "is_repeat": recording == MOTION_REPEAT,
                "n_samples": int(rb.valid.sum()),
                "duration_s": float((rb.time_ms[-1] - rb.time_ms[0]) / 1000.0),
                "residual_mm": stats.as_dict(),
            }
        )
    return {"table": "reference_fit", "dataset": dataset.id, "rows": rows}


def reference_fit_to_latex(table: dict) -> str:
    lines = []
    for row in table["rows"]:
        if row["is_repeat"]:
            lines.append(r"\addlinespace")
        r = row["residual_mm"]
        lines.append(
            f"{row['label']:<16s} & {row['duration_s']:.1f} & "
            f"${r['mean']:.3f} \\pm {r['sd']:.3f}$ & {r['maximum']:.3f} \\\\"
        )
    return "\n".join(lines) + "\n"


def write_reference_fit_table(dataset: Dataset, store: str | Path, out_dir: str | Path) -> dict:
    import json

    table = reference_fit_table(dataset, store)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "reference_fit.tex").write_text(reference_fit_to_latex(table))
    with open(out / "reference_fit.json", "w") as f:
        json.dump(table, f, indent=2)
        f.write("\n")
    return table
