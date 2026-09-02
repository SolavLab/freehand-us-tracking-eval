# Usage

Read `setup.md` first and install the `analysis` environment. Everything on this
page except the last section needs only that.

## Verify what you have

```bash
vipose verify
```

Checks all sixteen input files against the sha256 hashes in `dataset.yaml`, then
summarises the golden master. It fails on the first discrepancy rather than
proceeding with a partial dataset — the original code returned `None` for a
missing input and continued, which silently dropped a cell from a comparison
reported as complete.

## R1 — regenerate the tables

```bash
vipose tables
```

Reads `tests/golden/` and writes `tables/residuals.tex` and `residuals.json`.
The `.tex` is the table body only, so it can be diffed against the manuscript:

```bash
diff <(sed -n '/\\label{tab:residuals}/,/\\bottomrule/p' path/to/results.tex) tables/residuals.tex
```

To render the tables from your own run instead of the reference, pass
`--store results/<run-id>`.

## R2 — run the full evaluation

```bash
vipose evaluate
```

Roughly thirty minutes. Writes `results/<run-id>/`, where the run id is a UTC
timestamp and the dataset id:

```
results/<run-id>/
  run.yaml                                   versions, dataset, arguments
  recordings/<recording>/window.json         shared window and per-pipeline offsets
  cells/<pipeline>/<recording>/residuals.csv per-frame errors
  cells/<pipeline>/<recording>/calibration.json
  cells/<pipeline>/<recording>/cell.json     summary statistics
  <recording>.log                            full output of that recording's worker
```

Then check it against the published values:

```bash
pytest tests/ -v
```

The residual comparison is elementwise and exact. There is no random number
generator anywhere in the pipeline, and the reference was reproduced
bit-identically across a two-week gap, so a non-zero difference is a real change
rather than floating-point noise.

Useful options:

```bash
vipose evaluate --recordings pivot            # one recording
vipose evaluate --recordings pivot,freehand   # a subset
vipose evaluate --out /tmp/scratch-run        # somewhere other than results/
VIPOSE_RESULT_STORE=results/<run-id> pytest tests/test_golden_master.py
```

### Why one process per recording

`vipose evaluate` spawns a separate process for each recording. The legacy
evaluator creates a matplotlib figure on every calibration and never closes it,
and stage 3 performs about a hundred calibrations per recording, so a single
process evaluating all four passes 2 GB within minutes and exhausts memory
before finishing. Per-recording processes cap the peak near 4 GB.

This does not affect results. The shared evaluation window is computed across
the three pipelines *of one recording* and never across recordings, and the
published artifacts are themselves timestamped in two groups — so the run that
produced the paper's numbers was already split this way.

Each worker also runs with `DISPLAY` removed and BLAS thread counts pinned to
one. Both matter: importing the legacy calibration module calls
`matplotlib.use("TkAgg")` whenever a display is present, which silently defeats
`MPLBACKEND=Agg`, and the calibration and rigid-body fit are both SVD-based, so
thread count could otherwise affect the last digits.

## What the evaluation does

For each recording, and then for each pipeline within it:

1. **Synchronize.** The cameras and the Vicon system run on independent clocks.
   Stage 1 cross-correlates the angular-speed magnitudes; stage 2 refines by
   Nelder–Mead on the same scalar objective; stage 3, once a spatial calibration
   exists, refines against the median rotational residual. Stage 3 and the
   shared window are iterated to a fixed point, which converges in two passes.
2. **Crop to a shared window.** All three pipelines are restricted to the span
   in which all of them have valid poses, so they are compared over the same
   motion. The window is shared in the *Vicon* time base — the three acquisition
   clocks differ from it by different amounts, so their raw timestamps do not
   and should not agree.
3. **Calibrate.** Robot-world/hand-eye calibration recovers the two constant
   transforms relating the SLAM world frame and the marker cluster to the camera,
   using OpenCV's `calibrateRobotWorldHandEye` with Shah's method.
4. **Measure.** The residual between the two resulting estimates of the camera
   pose gives the per-frame translational and rotational error.

`docs/data_formats.md` specifies every file involved, and the appendices of the
paper derive the synchronization scheme and the calibration well-posedness
condition.

## R3 — from the raw recordings

Needs the ~20 GiB archive catalogued in
`datasets/probe-tracking-2025-10-23/raw.yaml`, plus the vendor environments from
`setup.md`. Point `VIPOSE_RAW_ROOT` at the archive.

| Stage | Environment | Command |
|---|---|---|
| RealSense `.bag` → ROS 2 bag | `ros-convert` | `extract/convert/realsense_bag_to_rosbag2.py` |
| Pipeline I poses from SVO2 | `zed-sdk` | `extract/zed_sdk/zed_sdk_trajectory.py` |
| Pipelines II and III poses | `isaac-ros` container | `extract/cuvslam/run_*.sh` |

Each writes a pose CSV conforming to
`datasets/probe-tracking-2025-10-23/schema/poses.schema.json`, which the
evaluation validates on load. A change in an extractor therefore surfaces as a
load-time schema error rather than as a quietly different number.

These stages are **not** bit-reproducible — see the reproduction ladder in the
README.
