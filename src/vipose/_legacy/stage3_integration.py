"""
Stage-3 temporal refinement, as a drop-in for the unified pipeline.

WHY
---
Stages 1-2 align the angular-SPEED MAGNITUDE profiles.  That is the only
criterion available before the spatial calibration exists, because it is
invariant to the unknown relative orientation of the two frames.  But it is a
scalar, and it constrains the offset only loosely: measured over nine
pipeline x condition cells it localises the offset to ~1-3 ms for the two
cuVSLAM pipelines but only to ~12 ms for the ZED SDK, whose angular-rate
signal is noisier (cross-correlation peak 0.74 against 0.88).

Once a hand-eye calibration exists that restriction is gone.  Stage 3 re-picks
the offset by minimising the median rotational residual and re-runs the
calibration at the refined value.

FIXED POINT
-----------
In unified mode the common evaluation window is derived from all pipelines'
offsets, so refining the offsets moves the window, which moves the residuals,
which can move the optimum again.  This module therefore iterates
offsets -> window -> offsets until the offsets stop moving (typically one or
two passes; the shifts are <= ~12 ms against ~50 s windows).

COST
----
Each offset evaluation is a full calibration (~5 s).  With the default
+/-16 ms grid at 4 ms plus a bounded refinement, expect ~15 evaluations per
pipeline per iteration, so roughly 4 minutes per pipeline per test.  Set
`enabled=False` to bypass entirely and reproduce the stage-2 behaviour.

HOW TO WIRE IT IN  (plot_path_unified.py)
-----------------------------------------
1. Import, near the other imports at the top:

       from stage3_integration import refine_offsets_stage3

2. Replace the body of STEP 3 (currently the `time_ranges_local = {...}` block
   through `window_result = compute_common_vicon_window(offsets_ms, ...)`)
   with the fixed-point call.  Concretely, after

       time_ranges_local = {
           pipeline_num: (df_ZED['Time_ms'].min(), df_ZED['Time_ms'].max())
           for pipeline_num, df_ZED in tracking_data.items()
       }

   insert:

       offsets_ms, window_result, stage3_info = refine_offsets_stage3(
           offsets_ms=offsets_ms,
           time_ranges_local=time_ranges_local,
           csv_paths={pn: str(pipeline_config.get_pipeline_csv(pn, test_num))
                      for pn in tracking_data},
           mocap_path=str(pipeline_config.get_mocap_csv(test_num)),
           compute_window=compute_common_vicon_window,
           config=CalibrationConfig(),
           verbose=verbose,
       )

   and delete the now-redundant

       window_result = compute_common_vicon_window(offsets_ms, time_ranges_local)

   Everything downstream (vicon_window, local_windows, Steps 4-5) is unchanged
   and picks up the refined offsets automatically.

3. Record the refinement in the metadata so it is traceable.  Where
   `offset_refinement_stages` is written, merge in `stage3_info[pipeline_num]`.

VERIFY
------
Compare against stage3_out/stage3.json, which was produced independently with
the window held fixed.  The refined offsets should agree to within ~1 ms; a
larger difference means the fixed-point iteration moved the window enough to
matter, which is worth knowing rather than hiding.
"""

import numpy as np

try:
    from scipy.optimize import minimize_scalar
except ImportError:                                   # pragma: no cover
    minimize_scalar = None

from plot_path_lib import process_zed_vicon_calibration

GRID_SPAN_MS = 16.0      # initial half-width; doubled if the min lands on an edge
GRID_STEP_MS = 4.0
MAX_SPAN_MS = 64.0
BRENT_XATOL_MS = 0.25
FIXED_POINT_TOL_MS = 0.5
MAX_FIXED_POINT_ITER = 4


def _median_residual(csv_path, mocap_path, config, lag_ms, window, cache):
    """Median rotational residual (deg) for one candidate offset."""
    key = round(float(lag_ms), 4)
    if key in cache:
        return cache[key]
    try:
        res = process_zed_vicon_calibration(
            csv_path, mocap_csv_path=mocap_path, config=config, verbose=False,
            pre_cropped_time_range=window, pre_computed_lag=float(lag_ms))
        angles = np.asarray(res.angles, dtype=float) if res.angles is not None else None
        val = float(np.nanmedian(angles)) if angles is not None and angles.size else np.inf
    except Exception:
        val = np.inf                                   # a failed fit is never the optimum
    cache[key] = val
    return val


def refine_one_offset(csv_path, mocap_path, config, initial_lag_ms, window,
                      verbose=False, label=""):
    """Coarse grid then bounded refinement of a single pipeline's offset."""
    cache = {}
    obj = lambda l: _median_residual(csv_path, mocap_path, config, l, window, cache)

    span = GRID_SPAN_MS
    while True:
        grid = initial_lag_ms + np.arange(-span, span + GRID_STEP_MS / 2, GRID_STEP_MS)
        vals = np.array([obj(l) for l in grid])
        k = int(np.argmin(vals))
        if 0 < k < len(grid) - 1 or span >= MAX_SPAN_MS:
            break
        span *= 2
        if verbose:
            print(f"      {label} minimum at grid edge, widening to +/-{span:.0f} ms")

    best = float(grid[k])
    if minimize_scalar is not None:
        lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
        r = minimize_scalar(obj, bounds=(lo, hi), method="bounded",
                            options={"xatol": BRENT_XATOL_MS})
        if obj(float(r.x)) <= vals[k]:                 # never accept a worse point
            best = float(r.x)

    edge_hit = not (0 < k < len(grid) - 1)
    return best, dict(grid_span_ms=float(span), grid_edge_hit=bool(edge_hit),
                      n_evaluations=len(cache),
                      median_before=obj(initial_lag_ms), median_after=obj(best))


def refine_offsets_stage3(offsets_ms, time_ranges_local, csv_paths, mocap_path,
                          compute_window, config, verbose=True, enabled=True):
    """Iterate offsets <-> common window until the offsets converge.

    Returns (refined_offsets_ms, window_result, stage3_info).
    With enabled=False, returns the inputs untouched so the caller is
    unchanged -- use this to reproduce stage-2 behaviour.
    """
    window_result = compute_window(offsets_ms, time_ranges_local)
    if not enabled:
        return offsets_ms, window_result, {pn: {"stage3_applied": False} for pn in offsets_ms}

    if verbose:
        print(f"\n{'='*70}\nStage 3: refining offsets against the rotational residual"
              f"\n{'='*70}")

    offsets = dict(offsets_ms)
    info = {pn: {} for pn in offsets}

    for it in range(MAX_FIXED_POINT_ITER):
        if not window_result.get('valid', True):
            if verbose:
                print("  common window invalid; keeping stage-2 offsets")
            break
        local_windows = window_result['local_windows']

        new_offsets, moved = {}, 0.0
        for pn, lag in offsets.items():
            if pn not in csv_paths or pn not in local_windows:
                new_offsets[pn] = lag
                continue
            t_min, t_max = local_windows[pn]
            refined, meta = refine_one_offset(
                csv_paths[pn], mocap_path, config, lag,
                {'t_min': t_min, 't_max': t_max},
                verbose=verbose, label=f"pipeline {pn}")
            new_offsets[pn] = refined
            moved = max(moved, abs(refined - lag))
            info[pn] = dict(meta, stage3_applied=True,
                            stage3_offset_ms=float(refined),
                            stage3_delta_ms=float(refined - offsets_ms[pn]),
                            fixed_point_iterations=it + 1)
            if verbose:
                print(f"  iter {it+1} pipeline {pn}: {lag:10.2f} -> {refined:10.2f} ms "
                      f"({refined - lag:+6.2f})  median dphi "
                      f"{meta['median_before']:.4f} -> {meta['median_after']:.4f} deg")

        offsets = new_offsets
        window_result = compute_window(offsets, time_ranges_local)
        if moved < FIXED_POINT_TOL_MS:
            if verbose:
                print(f"  converged after {it+1} iteration(s) "
                      f"(max move {moved:.2f} ms < {FIXED_POINT_TOL_MS} ms)")
            break
    else:
        if verbose:
            print(f"  WARNING: no convergence in {MAX_FIXED_POINT_ITER} iterations; "
                  f"using the last estimate")

    if verbose:
        for pn in sorted(offsets):
            d = offsets[pn] - offsets_ms[pn]
            print(f"  pipeline {pn}: stage2 {offsets_ms[pn]:10.2f} -> "
                  f"stage3 {offsets[pn]:10.2f} ms  ({d:+6.2f})")

    return offsets, window_result, info
