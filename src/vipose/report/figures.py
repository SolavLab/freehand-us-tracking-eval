"""The manuscript's figures.

This is the only module in the package that imports pyplot, and it is imported
only from the command line -- never by the evaluation. In the code this replaces,
the residual values were returned *by* the plotting functions, so results could
not be computed without drawing figures; that coupling is what made a matplotlib
error able to destroy a twelve-cell run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..geometry.rigid_body import fit_rigid_body
from ..io.mocap import load_mocap
from ..io.tracks import load_track
from ..kinematics import angular_speed
from ..recordings import Cell, Dataset
from ..results import read_calibration

__all__ = ["omega_overlay", "stage3_objective", "write_appendix_figures", "PIPELINE_COLOURS"]

#: One colour per pipeline, shared with the results figure so that a reader
#: carries a single colour mapping through the whole paper. These are the
#: matplotlib tab10 blue, green and orange used by the violin plots.
PIPELINE_COLOURS = {"zed-sdk": "#1f77b4", "zed-cuvslam": "#2ca02c", "rs-cuvslam": "#ff7f0e"}
PIPELINE_ROWS = ["zed-sdk", "zed-cuvslam", "rs-cuvslam"]
CONDITIONS = ["pivot", "mixed", "freehand"]

#: rad/ms -> deg/s. The angular speed is per millisecond because the timestamps
#: are; see vipose.kinematics.angular_speed.
RAD_PER_MS_TO_DEG_PER_S = float(np.degrees(1000.0))


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "legend.frameon": False, "axes.labelsize": 8, "axes.titlesize": 8,
    })
    return plt


def omega_overlay(
    dataset: Dataset, store: str | Path, out_dir: str | Path,
    *, recording: str = "pivot", seconds: float = 10.0,
) -> Path:
    """Angular speed of each pipeline against the reference, at its own offset.

    The reference is differentiated at its native 240 Hz and the resulting
    speed interpolated onto the camera timestamps -- the order the
    synchronization objective itself uses. Differentiating after resampling
    instead gives a visibly smoother trace, because the interval becomes 17 ms
    rather than 4.2 ms, so the order matters and is stated in the caption.
    """
    plt = _style()
    store, out_dir = Path(store), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mocap = load_mocap(
        dataset.reference_csv(recording), marker_prefix=dataset.marker_prefix,
        expected_rate_hz=dataset.vicon_rate_hz,
    )
    rb = fit_rigid_body(mocap, min_markers=4)
    t_ref, omega_ref = angular_speed(rb.time_ms, rb.rotvec)

    fig, axes = plt.subplots(3, 1, figsize=(6.6, 4.4), sharex=True, sharey=True)
    for ax, pipeline in zip(axes, PIPELINE_ROWS, strict=True):
        calib = read_calibration(_calib(store, pipeline, recording))
        offset, (lo, hi) = calib["temporal_offset_ms"], calib["local_window_ms"]

        track = load_track(dataset.tracking_csv(Cell(pipeline, recording)))
        t_cam, omega_cam = angular_speed(track.time_ms, track.rotvec)
        on_camera = np.interp(t_cam, t_ref + offset, omega_ref)

        inside = (t_cam >= lo) & (t_cam <= hi)
        r = float(np.corrcoef(omega_cam[inside], on_camera[inside])[0, 1])

        # A representative interior segment, not the start: the beginning of
        # every recording is the deliberate panning used to aid synchronization.
        t0 = lo + 0.15 * (hi - lo)
        show = (t_cam >= t0) & (t_cam <= t0 + seconds * 1000.0)

        ax.plot((t_cam[show] - t0) / 1000.0, on_camera[show] * RAD_PER_MS_TO_DEG_PER_S,
                color="0.35", lw=0.7, label="Vicon")
        ax.plot((t_cam[show] - t0) / 1000.0, omega_cam[show] * RAD_PER_MS_TO_DEG_PER_S,
                color=PIPELINE_COLOURS[pipeline], lw=0.8,
                label=dataset.label(pipeline=pipeline))
        ax.text(0.985, 0.9, f"$r = {r:.3f}$", transform=ax.transAxes, ha="right", va="top")
        ax.legend(loc="upper left", ncol=2, handlelength=1.4, columnspacing=1.0)
        ax.margins(x=0)

    axes[-1].set_xlabel("Time within evaluation window [s]")
    axes[1].set_ylabel(r"Angular speed $\omega$ [deg/s]")
    fig.tight_layout(pad=0.4)
    path = out_dir / "sync_omega_overlay.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def stage3_objective(dataset: Dataset, store: str | Path, out_dir: str | Path) -> Path:
    """Median rotational residual against temporal offset, per condition.

    The curve is computed here rather than read from a stored sweep, so it
    cannot fall out of step with the pipeline that produced the offsets. Each
    panel is plotted relative to that cell's stage-2 estimate, with a triangle
    at the grid minimum.
    """
    from ..pipeline import stage3_objective_curve

    plt = _style()
    store, out_dir = Path(store), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.3), sharey=True)
    for ax, recording in zip(axes, CONDITIONS, strict=True):
        mocap = load_mocap(
            dataset.reference_csv(recording), marker_prefix=dataset.marker_prefix,
            expected_rate_hz=dataset.vicon_rate_hz,
        )
        for pipeline in PIPELINE_ROWS:
            calib = read_calibration(_calib(store, pipeline, recording))
            stages = calib["offset_refinement_stages"]
            stage2 = stages["stage2_optimized_ms"]
            window = tuple(calib["local_window_ms"])
            track = load_track(dataset.tracking_csv(Cell(pipeline, recording)))

            # Sweep about the stage-2 estimate, which is what the panel's
            # x axis is relative to.
            offsets, values = stage3_objective_curve(track, mocap, stage2, window)
            ax.plot(offsets, values, "-o", color=PIPELINE_COLOURS[pipeline],
                    lw=0.9, ms=2.4, label=dataset.label(pipeline=pipeline))
            k = int(np.argmin(values))
            ax.plot(offsets[k], values[k], "v", color=PIPELINE_COLOURS[pipeline], ms=5, mew=0)

        ax.axvline(0.0, color="0.6", lw=0.6, ls=":")
        ax.set_title(dataset.label(recording=recording))
        ax.set_xlabel("Offset relative to stage 2 [ms]")
        ax.margins(x=0.04)

    axes[0].set_ylabel(r"median $\Delta\varphi$ [deg]")
    axes[0].legend(loc="upper left", handlelength=1.4)
    fig.tight_layout(pad=0.4)
    path = out_dir / "sync_stage3_objective.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def write_appendix_figures(dataset: Dataset, store: str | Path, out_dir: str | Path) -> list[Path]:
    return [
        omega_overlay(dataset, store, out_dir),
        stage3_objective(dataset, store, out_dir),
    ]


def _calib(store: Path, pipeline: str, recording: str) -> Path:
    for candidate in (
        store / pipeline / recording / "calibration.json",
        store / "cells" / pipeline / recording / "calibration.json",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no calibration.json for {pipeline}/{recording} under {store}")
