"""Reading and writing the per-cell result store.

One directory per cell, two files with fixed names:

    <root>/cells/<pipeline>/<recording>/residuals.csv
    <root>/cells/<pipeline>/<recording>/calibration.json

The pipeline and recording live in the path, so they are not repeated in the
filenames. The legacy layout repeated both in every filename
(``pipeline1_test2_calibration_results_unified_angles.csv``) and split the two
residual series across two files, with no time column in either.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .recordings import Cell

__all__ = [
    "Residuals",
    "read_residuals",
    "write_residuals",
    "read_calibration",
    "write_calibration",
]

RESIDUAL_COLUMNS = ["frame", "source_frame", "time_ms", "d_trans_mm", "d_rot_deg"]


@dataclass(frozen=True)
class Residuals:
    """Per-frame calibration residuals for one cell."""

    frame: np.ndarray
    source_frame: np.ndarray
    time_ms: np.ndarray
    d_trans_mm: np.ndarray
    d_rot_deg: np.ndarray

    def __len__(self) -> int:
        return int(self.frame.size)

    @property
    def duration_s(self) -> float:
        return float(self.time_ms[-1] - self.time_ms[0]) / 1000.0


def cell_dir(root: str | Path, cell: Cell) -> Path:
    return Path(root) / "cells" / cell.pipeline / cell.recording


def read_residuals(path: str | Path) -> Residuals:
    """Read a residuals CSV, validating its schema.

    A missing or extra column is an error. The legacy format is deliberately
    not accepted: it lacked time entirely, so silently tolerating it would
    reintroduce series that cannot be placed on a clock.
    """
    df = pd.read_csv(path)
    missing = [c for c in RESIDUAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}; found {list(df.columns)}")
    if not df["time_ms"].is_monotonic_increasing:
        raise ValueError(f"{path}: time_ms is not monotonically increasing")
    return Residuals(
        frame=df["frame"].to_numpy(dtype=np.int64),
        source_frame=df["source_frame"].to_numpy(dtype=np.int64),
        time_ms=df["time_ms"].to_numpy(dtype=np.int64),
        d_trans_mm=df["d_trans_mm"].to_numpy(dtype=float),
        d_rot_deg=df["d_rot_deg"].to_numpy(dtype=float),
    )


def write_residuals(path: str | Path, r: Residuals) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "frame": r.frame,
            "source_frame": r.source_frame,
            "time_ms": r.time_ms,
            "d_trans_mm": r.d_trans_mm,
            "d_rot_deg": r.d_rot_deg,
        }
    ).to_csv(path, index=False, lineterminator="\n")


def read_calibration(path: str | Path) -> dict:
    """Read a calibration JSON, returning 4x4 transforms as arrays."""
    data = json.loads(Path(path).read_text())
    for key, value in list(data.items()):
        if key.startswith("T_") and isinstance(value, list):
            data[key] = np.asarray(value, dtype=float)
        elif key.startswith("marker_") and isinstance(value, list):
            data[key] = np.asarray(value, dtype=float)
    return data


def write_calibration(path: str | Path, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    serialisable = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in data.items()
    }
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2, sort_keys=True)
        f.write("\n")
