# Porting notes

The evaluation was reimplemented from four modules of roughly 4,700 lines into
about 2,200 lines across twenty. This records what changed, what was
deliberately left alone, and how each step was checked — so that a reader who
wants to know whether the published numbers survived the rewrite can see the
answer rather than take it on trust.

## How it was verified

The original code was preserved byte-identically while the replacement was
written, and every module was compared against its counterpart on the shipped
data before the original was removed. Those comparisons were exact —
`assert_array_equal`, no tolerance — for the reference loader, the rigid-body
fit, the pose loader, the SE(3) helpers, the angular speed, the stage-1 offset,
the stage-2 objective on a grid, the stage-2 refined offset, and the marker
resampling.

The end-to-end pipeline reproduces the published per-frame residuals to
**1.4e-08 mm** and **2.5e-09 degrees** across all twelve pipeline × recording
cells, with frame indices and timestamps identical. The residual is not exactly
zero for one understood reason: stage 3 re-derives each temporal offset by
bounded Brent minimisation with a 0.25 ms tolerance, so a run can settle a
fraction of a femtosecond away and give a marginally different calibration.
With the offset pinned instead, the chain is exact to 1e-12.

Two further checks are independent of the residual comparison. The
cross-condition spread of the camera-to-marker distance, which the paper quotes
as an interpretation floor, reproduces to within 0.02 mm (6.32 / 4.28 / 2.77 mm).
And the pose-continuity counts reproduce the manuscript exactly on the
window-independent values.

## Deliberately preserved

These would each change published numbers, so they were left as they are and
documented where they will be read.

**Marker resampling is linear.** Appendix A states the marker trajectories were
resampled by cubic spline, and gives a reason. That is not what the code does:
there was exactly one cubic interpolation in the original, in the stage-2
angular-speed objective, and the marker resampling that feeds the residuals used
`np.interp`. `vipose.sync.resample.resample_reference` carries a warning saying
so.

**The Vicon sample rate stays at 240 Hz from the manifest.** The original
hard-coded it and never read the value the file declares, behind a comment that
said 120 Hz. Since the rate enters every timestamp, interpolation and reported
lag, reading it from the file instead would be a re-analysis. The manifest value
is authoritative and the file is now read to *verify* it, with a mismatch
raising.

**`load_mocap` selects markers by the `ZED:` prefix.** A property of this Nexus
export, not a bug. Parameterised in the manifest.

**The rotational residual uses `Rotation.magnitude()`**, not the
`arccos((tr R − 1)/2)` form the paper writes. They agree analytically, but
arccos of the trace loses precision near zero rotation and these residuals are
around a tenth of a degree. The quaternion route is well conditioned and is what
produced the published numbers.

## Changed, with reasons

**Bugs fixed in code that never ran.** `marker_coordinate_option` had one live
branch and one commented-out `else`, so any other value raised
`UnboundLocalError`; the option is gone and one construction remains. The
`calibration_option == 1` path and its naive quaternion averaging — no sign
disambiguation, so `q` and `−q` would cancel — are gone with it. The rebase
machinery is gone: it is off in the settled configuration, and it was the
subject of a corrected multiplication-order defect.

**One definition of the stage-2 optimiser.** The original had two: the live path
used `maxiter=20, xtol=1e-5` and a second, unreachable copy used `50, 1e-6`.
Only the former ran. The objective has several near-equal minima about 12 ms
apart, so two tolerance settings in one codebase is a real hazard.

**A clock-origin bug, found while porting.** `Track.time_ms` was originally
computed as `timestamp − timestamp[0]`, so cropping a track re-zeroed its clock.
The evaluation window is expressed in that clock, so a cropped track would have
placed every sample earlier than it belongs — 67 ms for one cell, four frames —
and the reference would have been resampled at the wrong instants. The origin is
now stored at load and carried through cropping.

**Immutability.** `MocapData` and `Track` are frozen, with `shifted()` and
`cropped()` returning new objects. The original resampler mutated the caller's
`Time_ms` in place, which was safe *only* because the CSV was re-read for every
pipeline. Loading it once per recording and sharing it across the three
pipelines — the obvious optimisation, and one a later reader would certainly try
— would have shifted the second pipeline's clock by the first's lag and changed
every published number without raising anything.

**Silent failures made loud.** A missing input file now raises at startup
instead of returning `None` and being skipped, which could drop a cell from a
comparison reported as complete. Resampling outside the reference span raises
instead of letting `np.interp` clamp — that fired immediately on two cells and
is how the clock-origin bug surfaced. A schema violation raises instead of
calling `sys.exit(1)` from inside a library.

**Metrics separated from plotting.** Residuals are computed by pure functions
and written to disk before any figure is drawn.

**The star import is gone.** `from SynchronizationNew import *` re-exported 54
names into the calibration module, of which 22 were used — including `error`,
which was `os.error`, via `from os import error` on line 1. The real dependency
list was derived mechanically by walking the compiled code objects rather than
by reading. `ruff` now treats `F403`/`F405` as errors.

## Not ported

`preprocess_zed_data` is a no-op at the settled configuration — all four of its
flags are `False` — so there was nothing to port. The 18 unreachable functions
in the original module, `generate_area_file.py` (unreferenced, and pipeline I
never loads a prebuilt map), the parallel `postprocess_calibration.py` metrics
implementation, and the two violin scripts not used in the paper are all
excluded; `docs/` and the release commit history record why.
