"""Orchestration: from inputs to per-frame residuals, for one recording.

All three pipelines of a recording are processed together, because the
evaluation window is shared between them. Sequence:

1. Stage 1 and stage 2 estimate each pipeline's temporal offset from the *full*
   trajectories.
2. Those offsets determine a shared window: the intersection, in the reference
   clock, of the intervals the three pipelines can cover.
3. Stage 3 refines each offset against the median rotational residual, which is
   only available once a spatial calibration exists. Since the window depends on
   the offsets and the offsets depend on the window, the two are iterated to a
   fixed point.
4. Each pipeline is calibrated at its refined offset over the shared window, and
   the residuals recorded.

No plotting anywhere. In the implementation this replaces, the residual values
were returned *by* the plotting functions, so results could not be computed
without drawing figures -- which is why a matplotlib error could destroy a
twelve-cell run and why the figures leaked until the process ran out of memory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from .calibration import HandEyeResult, marker_frame, solve_hand_eye
from .geometry.rigid_body import fit_rigid_body
from .geometry.transforms import to_matrices
from .io.mocap import MocapData
from .io.tracks import Track
from .metrics import rotational_residual, translational_residual
from .recordings import Cell, Dataset
from .results import Residuals
from .sync.resample import resample_reference
from .sync.stages import crosscorrelation_offset, refine_offset_omega

__all__ = [
    "CellResult",
    "evaluate_recording",
    "shared_window_from_offsets",
    "stage3_objective_curve",
]

log = logging.getLogger(__name__)

# Stage-3 search, matching the published configuration.
GRID_SPAN_MS = 16.0        # initial half-width, doubled if the minimum lands on an edge
GRID_STEP_MS = 4.0
MAX_SPAN_MS = 64.0
BRENT_XATOL_MS = 0.25
FIXED_POINT_TOL_MS = 0.5
MAX_FIXED_POINT_ITER = 4


@dataclass(frozen=True)
class CellResult:
    """Everything the evaluation produces for one pipeline x recording."""

    cell: Cell
    residuals: Residuals
    calibration: HandEyeResult
    marker: object
    temporal_offset_ms: float
    vicon_window_ms: tuple[float, float]
    local_window_ms: tuple[float, float]
    stages: dict
    reference_residual_mm: np.ndarray


def shared_window_from_offsets(
    offsets_ms: dict[str, float], spans_ms: dict[str, tuple[float, float]]
) -> tuple[tuple[float, float], dict[str, tuple[float, float]]]:
    """Intersect the pipelines' coverage in the reference clock.

    A pipeline sample at local time ``t`` corresponds to reference time
    ``t - offset``, so a positive offset means the reference clock runs ahead.
    Returns the shared window in reference time and its image in each
    pipeline's own clock.
    """
    in_reference = {
        p: (lo - offsets_ms[p], hi - offsets_ms[p]) for p, (lo, hi) in spans_ms.items()
    }
    start = max(lo for lo, _ in in_reference.values())
    end = min(hi for _, hi in in_reference.values())
    if start > end:
        raise ValueError(
            f"pipelines share no evaluation window: in reference time they cover "
            f"{in_reference}, intersecting to [{start:.2f}, {end:.2f}] ms"
        )
    local = {p: (start + offsets_ms[p], end + offsets_ms[p]) for p in offsets_ms}
    return (float(start), float(end)), local


def _calibrate(
    track: Track, reference: MocapData, offset_ms: float, window_ms: tuple[float, float]
):
    """Calibrate one pipeline at a given offset over a given window."""
    cropped = track.cropped(*window_ms)
    resampled = resample_reference(reference, cropped.time_ms, offset_ms)
    rb = fit_rigid_body(resampled, min_markers=4)

    camera_in_world = to_matrices(cropped.translation, cropped.rotvec)
    marker_in_vicon = to_matrices(rb.translation, rb.rotvec)
    calib = solve_hand_eye(camera_in_world, marker_in_vicon)

    slam = calib.T_slamworld_to_vicon @ camera_in_world
    marker_path = marker_in_vicon @ calib.T_camera_to_marker
    residuals = Residuals(
        frame=np.arange(len(cropped), dtype=np.int64),
        source_frame=cropped.frame,
        time_ms=cropped.timestamp_ms,
        d_trans_mm=translational_residual(slam, marker_path),
        d_rot_deg=rotational_residual(slam, marker_path),
    )
    return residuals, calib, resampled, rb


def _median_rotational_residual(track, reference, offset_ms, window_ms) -> float:
    residuals, *_ = _calibrate(track, reference, offset_ms, window_ms)
    return float(np.median(residuals.d_rot_deg))


def _refine_stage3(track, reference, offset_ms, window_ms, label=""):
    """Minimise the median rotational residual with respect to the offset.

    A coarse grid first, then bounded Brent inside the bracketing interval. A
    pure Brent search is not used because the objective has small local wiggles
    from frame resampling. A refined value is accepted only if it does not
    increase the objective.
    """
    cache: dict[float, float] = {}

    def objective(x: float) -> float:
        key = round(float(x), 9)
        if key not in cache:
            cache[key] = _median_rotational_residual(track, reference, key, window_ms)
        return cache[key]

    span = GRID_SPAN_MS
    while True:
        grid = offset_ms + np.arange(-span, span + GRID_STEP_MS / 2, GRID_STEP_MS)
        values = np.array([objective(g) for g in grid])
        k = int(np.argmin(values))
        if 0 < k < len(grid) - 1 or span >= MAX_SPAN_MS:
            break
        span *= 2
        log.info("%s stage-3 minimum at a grid edge, widening to +/-%.0f ms", label, span)

    best = float(grid[k])
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    brent = minimize_scalar(
        objective, bounds=(lo, hi), method="bounded", options={"xatol": BRENT_XATOL_MS}
    )
    if objective(float(brent.x)) <= values[k]:
        best = float(brent.x)

    return best, {
        "grid_span_ms": float(span),
        "grid_edge_hit": not (0 < k < len(grid) - 1),
        "n_evaluations": len(cache),
        "median_before": objective(offset_ms),
        "median_after": objective(best),
    }


def stage3_objective_curve(
    track: Track,
    reference: MocapData,
    offset_ms: float,
    window_ms: tuple[float, float],
    *,
    span_ms: float = GRID_SPAN_MS,
    step_ms: float = GRID_STEP_MS,
) -> tuple[np.ndarray, np.ndarray]:
    """Median rotational residual as a function of the temporal offset.

    Returns ``(offsets, median_phi)`` with offsets **relative to**
    ``offset_ms``. This is the stage-3 objective, exposed because its *shape*
    is the point: it is shallow, and how shallow decides how tightly the scalar
    angular-speed criterion of stages 1 and 2 can localise the offset at all.
    Appendix A plots it.
    """
    relative = np.arange(-span_ms, span_ms + step_ms / 2, step_ms)
    values = np.array([
        _median_rotational_residual(track, reference, offset_ms + d, window_ms)
        for d in relative
    ])
    return relative, values


def evaluate_recording(
    dataset: Dataset,
    recording: str,
    tracks: dict[str, Track],
    reference: MocapData,
    *,
    stage3: bool = True,
) -> dict[str, CellResult]:
    """Evaluate every pipeline of one recording."""
    reference_full = fit_rigid_body(reference, min_markers=4)

    # Stages 1 and 2, on the full trajectories.
    offsets: dict[str, float] = {}
    stages: dict[str, dict] = {}
    for pipeline, track in tracks.items():
        s1 = crosscorrelation_offset(
            track.time_ms, track.rotvec, reference_full.time_ms, reference_full.rotvec
        )
        s2 = refine_offset_omega(
            track.time_ms, track.rotvec, reference_full.time_ms, reference_full.rotvec, s1
        )
        offsets[pipeline] = s2
        stages[pipeline] = {
            "stage1_crosscorr_ms": float(s1),
            "stage2_optimized_ms": float(s2),
            "stage2_delta_ms": float(s2 - s1),
        }
        log.info("%s/%s stage 1 %.3f -> stage 2 %.3f ms", pipeline, recording, s1, s2)

    spans = {p: (float(t.time_ms[0]), float(t.time_ms[-1])) for p, t in tracks.items()}
    vicon_window, local_windows = shared_window_from_offsets(offsets, spans)
    initial_offsets = dict(offsets)

    if stage3:
        for iteration in range(MAX_FIXED_POINT_ITER):
            moved = 0.0
            refined = {}
            for pipeline, track in tracks.items():
                new, meta = _refine_stage3(
                    track, reference, offsets[pipeline], local_windows[pipeline],
                    label=f"{pipeline}/{recording}",
                )
                moved = max(moved, abs(new - offsets[pipeline]))
                refined[pipeline] = new
                stages[pipeline].update(
                    meta,
                    stage3_applied=True,
                    stage3_offset_ms=float(new),
                    stage3_delta_ms=float(new - initial_offsets[pipeline]),
                    fixed_point_iterations=iteration + 1,
                )
            offsets = refined
            vicon_window, local_windows = shared_window_from_offsets(offsets, spans)
            if moved < FIXED_POINT_TOL_MS:
                break
        else:
            log.warning(
                "%s: stage-3 fixed point did not settle within %d iterations",
                recording, MAX_FIXED_POINT_ITER,
            )
    else:
        for pipeline in stages:
            stages[pipeline]["stage3_applied"] = False

    results = {}
    for pipeline, track in tracks.items():
        residuals, calib, resampled, rb = _calibrate(
            track, reference, offsets[pipeline], local_windows[pipeline]
        )
        results[pipeline] = CellResult(
            cell=Cell(pipeline, recording),
            residuals=residuals,
            calibration=calib,
            marker=marker_frame(resampled.positions[0], order=tuple(dataset.marker_order)),
            temporal_offset_ms=float(offsets[pipeline]),
            vicon_window_ms=vicon_window,
            local_window_ms=local_windows[pipeline],
            stages=stages[pipeline],
            reference_residual_mm=rb.residual_mm,
        )
    return results
