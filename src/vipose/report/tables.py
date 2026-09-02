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

from pathlib import Path

from ..metrics import summarize
from ..recordings import Cell, Dataset
from ..results import read_residuals

__all__ = ["residuals_table", "write_residuals_table"]

# Row order follows the manuscript, which groups by pipeline with the repeated
# pivot acquisition set apart from the three motion conditions.
PIPELINE_ORDER = ["zed-cuvslam", "rs-cuvslam", "zed-sdk"]
CONDITION_ORDER = ["pivot", "mixed", "freehand"]
REPEAT = "pivot-repeat"

# Precision the manuscript prints at. Translational residuals are quoted to
# 0.01 mm and rotational to 0.001 deg; emitting more would imply a resolution
# the measurement does not have.
MM_DP = 2
DEG_DP = 3


def residuals_table(dataset: Dataset, store: str | Path) -> dict:
    """Compute the residuals table from a result store or golden master.

    ``store`` is a directory containing ``<pipeline>/<recording>/residuals.csv``
    (the golden-master layout) or ``cells/<pipeline>/<recording>/residuals.csv``
    (a run's layout); both are accepted.
    """
    store = Path(store)
    rows = []
    for pipeline in PIPELINE_ORDER:
        for recording in CONDITION_ORDER + [REPEAT]:
            cell = Cell(pipeline, recording)
            path = _residuals_path(store, cell)
            r = read_residuals(path)
            trans, rot = summarize(r.d_trans_mm), summarize(r.d_rot_deg)
            rows.append(
                {
                    "pipeline": pipeline,
                    "pipeline_label": dataset.label(pipeline=pipeline),
                    "recording": recording,
                    "recording_label": dataset.label(recording=recording),
                    "is_repeat": recording == REPEAT,
                    "frames": trans.n,
                    "translational_mm": trans.as_dict(),
                    "rotational_deg": rot.as_dict(),
                }
            )
    return {"table": "residuals", "dataset": dataset.id, "rows": rows}


def to_latex(table: dict) -> str:
    """Render the table body as LaTeX rows."""
    # Pipeline macros as the manuscript defines them.
    macro = {"zed-cuvslam": r"\zedCuVSLAM{}", "rs-cuvslam": r"\rsCuVSLAM{}",
             "zed-sdk": r"\zedSDK{}"}
    lines, previous = [], None
    for row in table["rows"]:
        if previous is not None and row["pipeline"] != previous:
            lines.append(r"\midrule")
        elif row["is_repeat"]:
            lines.append(r"\addlinespace")
        first = row["pipeline"] != previous
        name = macro.get(row["pipeline"], row["pipeline_label"])
        t, r = row["translational_mm"], row["rotational_deg"]
        lines.append(
            f"{name if first else '':<16s} & {row['recording_label']:<14s} & "
            f"{row['frames']:5d} & "
            f"{t['median']:.{MM_DP}f} ({t['iqr']:.{MM_DP}f}) & {t['p95']:.{MM_DP}f} & "
            f"{r['median']:.{DEG_DP}f} ({r['iqr']:.{DEG_DP}f}) & {r['p95']:.{DEG_DP}f} \\\\"
        )
        previous = row["pipeline"]
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
