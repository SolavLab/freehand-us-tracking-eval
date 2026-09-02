"""Compatibility bridge between the dataset layout and the legacy evaluator.

The legacy code discovers its inputs by walking a directory tree and matching
filename patterns (``{test}_ZED_SDK_trajectory.csv`` and so on) keyed on
``test1``..``test5``, and writes its outputs into the same tree. The release
addresses recordings by condition and keeps inputs immutable.

This module reconciles the two by staging a tree of symlinks with the names the
legacy code expects, running it there, and translating its output back. It is
temporary: once the numerics are absorbed into ``vipose`` proper, both this file
and ``vipose._legacy`` are deleted.

The approach is not speculative -- it is exactly how the golden master was
produced in Phase 0, and that run reproduced the published artifacts with a
maximum per-frame difference of zero.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .recordings import Cell, Dataset
from .results import Residuals

__all__ = [
    "LEGACY_PIPELINE_DIR",
    "LEGACY_PIPELINE_NUM",
    "LEGACY_CSV_TEMPLATE",
    "legacy_test_name",
    "stage_inputs",
    "legacy_outputs",
    "convert_cell",
]

# The legacy directory name and trajectory-filename template for each pipeline.
# Both come from `dataset.yaml`'s `legacy_dir` where possible; the templates
# have no counterpart in the manifest because they exist only here.
LEGACY_PIPELINE_DIR = {
    "zed-sdk": "pipeline1_zed_sdk",
    "zed-cuvslam": "pipeline2_isaac_svo2",
    "rs-cuvslam": "pipeline3_isaac_realsense",
}
LEGACY_PIPELINE_NUM = {"zed-sdk": 1, "zed-cuvslam": 2, "rs-cuvslam": 3}
LEGACY_CSV_TEMPLATE = {
    "zed-sdk": "{test}_ZED_SDK_trajectory.csv",
    "zed-cuvslam": "{test}_cuVSLAM_results.csv",
    "rs-cuvslam": "{test}_realsense_cuVSLAM_results.csv",
}


def legacy_test_name(dataset: Dataset, recording: str) -> str:
    """``"test2"`` for ``"pivot"``, from the manifest's ``legacy_test``."""
    return f"test{dataset.manifest['recordings'][recording]['legacy_test']}"


def stage_inputs(dataset: Dataset, recording: str, workdir: str | Path) -> tuple[Path, Path]:
    """Build the legacy-shaped input tree for one recording.

    Returns ``(results_dir, data_dir)`` to pass to the legacy
    ``PipelineConfig``. Inputs are symlinked, so the dataset stays immutable and
    nothing is copied; outputs land under ``results_dir/comparison``.
    """
    workdir = Path(workdir)
    test = legacy_test_name(dataset, recording)
    results = workdir / "results"
    data = workdir / "data" / test
    data.mkdir(parents=True, exist_ok=True)

    _link(dataset.reference_csv(recording), data / f"{test}_mocap.csv")

    for pipeline in dataset.pipelines:
        d = results / LEGACY_PIPELINE_DIR[pipeline] / test
        d.mkdir(parents=True, exist_ok=True)
        name = LEGACY_CSV_TEMPLATE[pipeline].format(test=test)
        _link(dataset.tracking_csv(Cell(pipeline, recording)), d / name)

    (results / "comparison").mkdir(parents=True, exist_ok=True)
    return results, workdir / "data"


def legacy_outputs(dataset: Dataset, cell: Cell, results_dir: str | Path) -> dict[str, Path]:
    """Paths the legacy code writes for one cell, after a unified run."""
    test = legacy_test_name(dataset, cell.recording)
    num = LEGACY_PIPELINE_NUM[cell.pipeline]
    d = Path(results_dir) / "comparison" / LEGACY_PIPELINE_DIR[cell.pipeline] / test
    stem = f"pipeline{num}_{test}_calibration_results_unified"
    return {
        "distances": d / f"{stem}_distances.csv",
        "angles": d / f"{stem}_angles.csv",
        "calibration_txt": d / f"{stem}.txt",
        "metadata": d / f"pipeline{num}_{test}_metadata_unified.json",
        "cropped_trajectory": d
        / (LEGACY_CSV_TEMPLATE[cell.pipeline].format(test=test).replace(".csv", "_unified.csv")),
    }


def convert_cell(dataset: Dataset, cell: Cell, results_dir: str | Path) -> tuple[Residuals, dict]:
    """Translate one cell of legacy output into the release's types.

    Two things are recovered here that the legacy format loses:

    * **Time.** The legacy residual CSVs carry only a positional row counter,
      split across two files. Timestamps come from the cropped trajectory the
      legacy code writes alongside them, which aligns row-for-row. The
      alignment is asserted, not assumed.
    * **Machine-readable transforms.** The calibration was written as a text
      dump of numpy ``repr``, which nothing can read back -- which is why
      downstream consumers had been re-deriving quantities already computed.
    """
    paths = legacy_outputs(dataset, cell, results_dir)
    d = pd.read_csv(paths["distances"])
    a = pd.read_csv(paths["angles"])
    tr = pd.read_csv(paths["cropped_trajectory"])
    meta = json.loads(paths["metadata"].read_text())

    dcol = next(c for c in d.columns if c != "Frame")
    acol = next(c for c in a.columns if c != "Frame")
    if not d["Frame"].equals(a["Frame"]):
        raise ValueError(f"{cell}: distance and angle frame columns differ")
    if len(tr) != len(d):
        raise ValueError(f"{cell}: cropped trajectory has {len(tr)} rows, residuals {len(d)}")
    if list(d["Frame"]) != list(range(len(d))):
        raise ValueError(f"{cell}: legacy Frame column is not a positional counter")

    residuals = Residuals(
        frame=d["Frame"].to_numpy(dtype="int64"),
        source_frame=tr["Frame"].to_numpy(dtype="int64"),
        time_ms=tr["Timestamp"].to_numpy(dtype="int64"),
        d_trans_mm=d[dcol].to_numpy(dtype=float),
        d_rot_deg=a[acol].to_numpy(dtype=float),
    )

    calibration = _parse_calibration_txt(paths["calibration_txt"].read_text())
    calibration.update(
        temporal_offset_ms=meta["temporal_offset_ms"],
        offset_refinement_stages=meta["offset_refinement_stages"],
        vicon_window_ms=meta["vicon_window"],
        local_window_ms=meta["local_window"],
    )
    return residuals, calibration


_NUMBER = re.compile(r"-?\d+\.\d+e[+-]\d+|-?\d+\.\d+|-?\d+")

_MATRICES = (
    ("T_camera_to_marker", "T_zed2rb"),
    ("T_slamworld_to_vicon", "T_zed02vicon"),
    ("T_vicon_to_slamworld", "T_vicon2zed0"),
    ("T_camera_to_marker_intrinsic", "T_zed2marker_intrinsic"),
)
_VECTORS = (
    ("marker_centroid", "Marker centroid"),
    ("marker_x_axis", "Marker-intrinsic X-axis"),
    ("marker_y_axis", "Marker-intrinsic Y-axis"),
    ("marker_z_axis", "Marker-intrinsic Z-axis"),
)


def _parse_calibration_txt(text: str) -> dict:
    """Extract transforms from the legacy text dump. The last place it is read."""
    out: dict = {}
    for key, label in _MATRICES:
        m = re.search(re.escape(label) + r"[^\[]*(\[\[.*?\]\])", text, re.S)
        if not m:
            continue
        values = [float(x) for x in _NUMBER.findall(m.group(1))]
        if len(values) != 16:
            raise ValueError(f"{label}: expected 16 numbers, found {len(values)}")
        out[key] = np.asarray(values, dtype=float).reshape(4, 4).tolist()
    for key, label in _VECTORS:
        m = re.search(re.escape(label) + r"[^\[]*\[([^\]]*)\]", text, re.S)
        if m:
            out[key] = [float(x) for x in _NUMBER.findall(m.group(1))]
    missing = [k for k, _ in _MATRICES if k not in out]
    if missing:
        raise ValueError(f"calibration dump is missing {missing}")
    return out


def _link(target: Path, link: Path) -> None:
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(Path(target).resolve())
