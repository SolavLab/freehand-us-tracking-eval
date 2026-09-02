"""Run the full evaluation and write a result store.

One process per recording, driven by :func:`evaluate_all`. The native engine
does not leak memory, so this is now only for isolation and parallelism rather
than a necessity -- but it also means one recording failing does not lose the
others, and each recording's full log is kept beside its results.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .io.mocap import load_mocap
from .io.tracks import load_track
from .metrics import summarize
from .pipeline import evaluate_recording
from .recordings import Cell, Dataset
from .results import cell_dir, write_calibration, write_residuals

__all__ = ["evaluate_recording_to_store", "evaluate_all", "run_id"]

log = logging.getLogger(__name__)


def run_id(dataset: Dataset) -> str:
    """A directory name identifying this run: UTC timestamp and dataset id."""
    return f"{datetime.now(timezone.utc):%Y-%m-%dT%H%MZ}-{dataset.id}"


def evaluate_recording_to_store(
    dataset: Dataset, recording: str, out_root: str | Path
) -> list[Cell]:
    """Evaluate one recording and write it into the store."""
    out_root = Path(out_root)
    tracks = {
        p: load_track(dataset.tracking_csv(Cell(p, recording))) for p in dataset.pipelines
    }
    reference = load_mocap(
        dataset.reference_csv(recording),
        marker_prefix=dataset.marker_prefix,
        expected_rate_hz=dataset.vicon_rate_hz,
        expected_markers=len(dataset.marker_order),
    )

    results = evaluate_recording(dataset, recording, tracks, reference)

    windows = {}
    for pipeline, result in results.items():
        d = cell_dir(out_root, result.cell)
        write_residuals(d / "residuals.csv", result.residuals)
        write_calibration(
            d / "calibration.json",
            {
                "T_camera_to_marker": result.calibration.T_camera_to_marker,
                "T_slamworld_to_vicon": result.calibration.T_slamworld_to_vicon,
                "T_vicon_to_slamworld": result.calibration.T_vicon_to_slamworld,
                "T_camera_to_marker_intrinsic": result.marker.express(
                    result.calibration.T_camera_to_marker
                ),
                "marker_centroid": result.marker.centroid,
                "marker_x_axis": result.marker.x_axis,
                "marker_y_axis": result.marker.y_axis,
                "marker_z_axis": result.marker.z_axis,
                "camera_to_marker_distance_mm": result.calibration.camera_to_marker_distance_mm,
                "temporal_offset_ms": result.temporal_offset_ms,
                "offset_refinement_stages": result.stages,
                "vicon_window_ms": list(result.vicon_window_ms),
                "local_window_ms": list(result.local_window_ms),
                "n_frames_used": result.calibration.n_frames_used,
                "provenance": {"vipose_version": __version__, "dataset": dataset.id},
            },
        )
        t = summarize(result.residuals.d_trans_mm)
        a = summarize(result.residuals.d_rot_deg)
        ref = result.reference_residual_mm
        (d / "cell.json").write_text(
            json.dumps(
                {
                    "pipeline": pipeline,
                    "recording": recording,
                    "frames": len(result.residuals),
                    "duration_s": result.residuals.duration_s,
                    "temporal_offset_ms": result.temporal_offset_ms,
                    "translational_mm": t.as_dict(),
                    "rotational_deg": a.as_dict(),
                    "reference_fit_mm": summarize(ref[~_isnan(ref)]).as_dict(),
                },
                indent=2, sort_keys=True,
            )
            + "\n"
        )
        windows[pipeline] = {
            "temporal_offset_ms": result.temporal_offset_ms,
            "local_window_ms": list(result.local_window_ms),
            "offset_refinement_stages": result.stages,
        }

    shared = {tuple(r.vicon_window_ms) for r in results.values()}
    if len(shared) != 1:
        raise RuntimeError(
            f"{recording}: pipelines disagree on the shared Vicon window: {shared}"
        )
    rec_dir = out_root / "recordings" / recording
    rec_dir.mkdir(parents=True, exist_ok=True)
    (rec_dir / "window.json").write_text(
        json.dumps(
            {
                "recording": recording,
                "vicon_window_ms": list(shared.pop()),
                "pipelines": windows,
            },
            indent=2, sort_keys=True,
        )
        + "\n"
    )
    return [r.cell for r in results.values()]


def evaluate_all(
    dataset: Dataset,
    out_root: str | Path,
    *,
    recordings: list[str] | None = None,
) -> Path:
    """Evaluate every recording, one subprocess each."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    todo = recordings or dataset.recordings()

    for i, recording in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {recording}", flush=True)
        logfile = out_root / f"{recording}.log"
        with open(logfile, "w") as fh:
            proc = subprocess.run(
                [
                    sys.executable, "-m", "vipose.evaluate",
                    "--dataset", str(dataset.root),
                    "--out", str(out_root),
                    "--recording", recording,
                ],
                stdout=fh, stderr=subprocess.STDOUT, env=_worker_env(),
            )
        if proc.returncode != 0:
            raise RuntimeError(f"{recording} failed with exit {proc.returncode}; see {logfile}")
        print(f"        done ({logfile.name})", flush=True)

    _write_run_metadata(dataset, out_root, todo)
    return out_root


def _worker_env() -> dict[str, str]:
    """Environment for a worker process.

    BLAS thread counts are pinned to one so results do not depend on how a
    library happens to decompose an SVD; both the hand-eye calibration and the
    rigid-body fit are SVD-based.
    """
    env = dict(os.environ)
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        MPLBACKEND="Agg",
        QT_QPA_PLATFORM="offscreen",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    return env


def _write_run_metadata(dataset: Dataset, out_root: Path, recordings: list[str]) -> None:
    import cv2
    import numpy
    import pandas
    import scipy

    (out_root / "run.yaml").write_text(
        "\n".join(
            [
                "# Provenance for this evaluation run.",
                f"vipose_version: {__version__}",
                f"dataset: {dataset.id}",
                f"recordings: [{', '.join(recordings)}]",
                f"pipelines: [{', '.join(dataset.pipelines)}]",
                "environment:",
                f"  python: {platform.python_version()}",
                f"  numpy: {numpy.__version__}",
                f"  scipy: {scipy.__version__}",
                f"  pandas: {pandas.__version__}",
                f"  opencv: {cv2.__version__}",
                "",
            ]
        )
    )


def _isnan(a):
    import numpy as np

    return np.isnan(a)


def _main(argv: list[str] | None = None) -> int:
    """Per-recording worker, re-entered as a subprocess by :func:`evaluate_all`."""
    import argparse

    p = argparse.ArgumentParser(prog="python -m vipose.evaluate")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--recording", required=True)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ds = Dataset.load(args.dataset, verify_hashes=False)
    cells = evaluate_recording_to_store(ds, args.recording, args.out)
    print(f"wrote {len(cells)} cells for {args.recording}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
