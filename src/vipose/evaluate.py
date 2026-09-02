"""Run the full evaluation: synchronization, hand-eye calibration, residuals.

One process per recording. That is not a stylistic choice: the legacy evaluator
creates a matplotlib figure on every calibration and never closes it, and stage
3 performs roughly a hundred calibrations per recording, so a single process
evaluating all four passes 2 GB within minutes and exhausts memory before
finishing. Per-recording processes cap the peak near 4 GB and reclaim it in
between.

Splitting this way does not change the results. Unification is within a
recording -- the shared evaluation window is computed across the three pipelines
of one recording, never across recordings -- and the published artifacts are
themselves timestamped in two groups, so the run that produced the paper's
numbers was already split by recording.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from ._bridge import convert_cell, stage_inputs
from .metrics import summarize
from .recordings import Cell, Dataset
from .results import cell_dir, write_calibration, write_residuals

__all__ = ["evaluate_recording", "evaluate_all", "run_id"]


def run_id(dataset: Dataset) -> str:
    """A directory name that identifies this run: UTC timestamp plus versions."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%MZ")
    return f"{stamp}-{dataset.id}"


def evaluate_recording(
    dataset: Dataset, recording: str, out_root: str | Path, *, verbose: bool = True
) -> list[Cell]:
    """Evaluate all pipelines of one recording, in this process.

    Writes ``<out_root>/cells/<pipeline>/<recording>/`` and
    ``<out_root>/recordings/<recording>/window.json``.
    """
    # Importing the legacy package mutates matplotlib and Qt global state, so it
    # is imported here rather than at module scope: a caller that only wants
    # `run_id` should not acquire a matplotlib backend as a side effect.
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from ._legacy import plot_path_unified as legacy  # noqa: F401  (side-effecting import)

    out_root = Path(out_root)
    with tempfile.TemporaryDirectory(prefix=f"vipose-{recording}-") as tmp:
        results_dir, data_dir = stage_inputs(dataset, recording, tmp)
        config = legacy.PipelineConfig(results_dir, data_dir)
        pipeline_nums = [_pipeline_num(dataset, p) for p in dataset.pipelines]

        legacy.process_test_unified_calibration(
            config,
            int(dataset.manifest["recordings"][recording]["legacy_test"]),
            pipelines=pipeline_nums,
            verbose=verbose,
        )

        written = []
        windows = {}
        for pipeline in dataset.pipelines:
            cell = Cell(pipeline, recording)
            residuals, calibration = convert_cell(dataset, cell, results_dir)
            calibration["provenance"] = {
                "vipose_version": __version__,
                "dataset": dataset.id,
                "engine": "vipose._legacy (verbatim original)",
            }
            d = cell_dir(out_root, cell)
            write_residuals(d / "residuals.csv", residuals)
            write_calibration(d / "calibration.json", calibration)

            t, a = summarize(residuals.d_trans_mm), summarize(residuals.d_rot_deg)
            (d / "cell.json").write_text(
                json.dumps(
                    {
                        "pipeline": pipeline,
                        "recording": recording,
                        "frames": len(residuals),
                        "duration_s": residuals.duration_s,
                        "temporal_offset_ms": calibration["temporal_offset_ms"],
                        "translational_mm": t.as_dict(),
                        "rotational_deg": a.as_dict(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            windows[pipeline] = {
                "vicon_window_ms": calibration["vicon_window_ms"],
                "local_window_ms": calibration["local_window_ms"],
                "temporal_offset_ms": calibration["temporal_offset_ms"],
                "offset_refinement_stages": calibration["offset_refinement_stages"],
            }
            written.append(cell)

        # The shared window must be identical across pipelines; that is the
        # fairness property unified mode provides, so it is checked, not assumed.
        shared = [w["vicon_window_ms"] for w in windows.values()]
        if any(w != shared[0] for w in shared):
            raise RuntimeError(
                f"{recording}: pipelines disagree on the shared Vicon window: {shared}"
            )
        rec_dir = out_root / "recordings" / recording
        rec_dir.mkdir(parents=True, exist_ok=True)
        (rec_dir / "window.json").write_text(
            json.dumps(
                {"recording": recording, "vicon_window_ms": shared[0], "pipelines": windows},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    return written


def evaluate_all(
    dataset: Dataset,
    out_root: str | Path,
    *,
    recordings: list[str] | None = None,
    verbose: bool = True,
) -> Path:
    """Evaluate every recording, one subprocess each. Returns ``out_root``."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    todo = recordings or dataset.recordings()

    for i, recording in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {recording}", flush=True)
        log = out_root / f"{recording}.log"
        with open(log, "w") as fh:
            proc = subprocess.run(
                [
                    sys.executable, "-m", "vipose.evaluate",
                    "--dataset", str(dataset.root),
                    "--out", str(out_root),
                    "--recording", recording,
                ],
                stdout=fh, stderr=subprocess.STDOUT,
                env=_worker_env(),
            )
        if proc.returncode != 0:
            raise RuntimeError(f"{recording} failed with exit {proc.returncode}; see {log}")
        print(f"        done -> {out_root / 'cells'} (log: {log.name})", flush=True)

    _write_run_metadata(dataset, out_root, todo)
    return out_root


def _worker_env() -> dict[str, str]:
    """Environment for a worker process.

    ``DISPLAY`` and ``WAYLAND_DISPLAY`` are removed, not just overridden.
    Importing the legacy calibration module calls ``matplotlib.use("TkAgg")``
    whenever a display is present, which silently defeats ``MPLBACKEND=Agg`` --
    verified: the backend really is TkAgg after that import on a desktop
    session. An interactive backend in a batch run is slower, can open windows,
    and fails outright in CI, so the display is hidden from the worker instead.

    Thread counts are pinned to one so results do not depend on how a BLAS
    library happens to decompose an SVD. The hand-eye calibration and the Kabsch
    fit are both SVD-based.
    """
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
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
                "engine: vipose._legacy (verbatim original code)",
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


def _pipeline_num(dataset: Dataset, pipeline: str) -> int:
    from ._bridge import LEGACY_PIPELINE_NUM

    return LEGACY_PIPELINE_NUM[pipeline]


def _main(argv: list[str] | None = None) -> int:
    """Per-recording worker, re-entered as a subprocess by :func:`evaluate_all`."""
    import argparse

    p = argparse.ArgumentParser(prog="python -m vipose.evaluate")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--recording", required=True)
    args = p.parse_args(argv)

    ds = Dataset.load(args.dataset, verify_hashes=False)
    cells = evaluate_recording(ds, args.recording, args.out)
    print(f"wrote {len(cells)} cells for {args.recording}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
