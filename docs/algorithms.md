# Algorithms

`vipose` compares a camera trajectory against a motion-capture reference in
four steps: **synchronize** the two clocks, **crop** to the span they share,
**calibrate** the constant transform between the camera and the reference,
and **measure** the per-frame residual. Each step is a small set of plain-array
functions, importable directly from `vipose`, with no dependency on this
repository's dataset manifest. This page documents them as a library; for the
CLI workflow that drives them over a whole dataset, see [usage.md](usage.md).

Throughout: translations are millimetres, orientations are **rotation
vectors** (axis-angle, radians) — never Euler angles, which introduce an
orientation-dependent gain error that is worst exactly when you need
precision (see `vipose.geometry.transforms`) — and a rigid-body transform is
a `(4, 4)` homogeneous matrix, or `(n, 4, 4)` for a trajectory.

## 1. Temporal synchronization — `vipose.sync.stages`

The camera and the reference run on independent clocks with an unknown,
roughly-constant offset. Both are synchronized on their **geodesic angular
speed** — the rotation angle between successive orientations divided by the
interval — because that quantity does not depend on the (still unknown)
spatial alignment between the two trajectories:

```python
crosscorrelation_offset(track_time_ms, track_rotvec,
                         reference_time_ms, reference_rotvec) -> float
refine_offset_omega(track_time_ms, track_rotvec,
                     reference_time_ms, reference_rotvec,
                     initial_offset_ms) -> float
```

Both take plain `(n,)` timestamp arrays and `(n, 3)` rotation-vector arrays —
nothing pipeline- or dataset-specific. A positive offset means the reference
clock runs ahead of the track's.

- `crosscorrelation_offset` (stage 1) gets a coarse estimate from the peak of
  the cross-correlation between the two angular-speed series, refined to
  sub-sample precision.
- `refine_offset_omega` (stage 2) polishes that estimate by minimizing the RMS
  mismatch between the two speed series directly (Nelder–Mead).

**Know this before you use it:** stage 1 resamples the reference onto the
track's timestamps by linear interpolation; stage 2's objective
(`omega_mismatch_rms`) resamples by cubic spline instead. That asymmetry is
deliberate — the two stages were built to match different published
behaviour — and worth being aware of if you change either. `angular_speed`
(in `vipose.kinematics`) reports **radians per millisecond**, not radians per
second, because its inputs are in milliseconds; a scale mistake here degrades
this stage silently rather than raising.

If your own reference is already pose-level rather than marker-level (another
SLAM system, a robot arm), you can synchronize directly on its rotation-vector
stream — nothing here requires a motion-capture marker cluster.

## 2. Fitting a rigid body to a marker cluster — `vipose.geometry.rigid_body`

If your reference is a motion-capture system reporting individual marker
positions rather than poses, `fit_rigid_body` recovers the cluster's rigid
pose per frame by the Kabsch algorithm:

```python
fit_rigid_body(mocap: MocapData, *, min_markers: int = 4,
                reference_frame: int = 0) -> RigidBodyTrajectory
```

`mocap` is a `vipose.io.mocap.MocapData` — a frozen dataclass of a frame
index, a time array and a tuple of named `(n, 3)` marker trajectories; load
one from a Vicon Nexus export with `load_mocap`, or build one directly from
your own marker arrays (it's five lines: see `vipose/io/mocap.py`). Frames
with fewer than `min_markers` visible markers come back as NaN rather than a
fit from too little data — occlusion drops a frame, not a marker, and the
returned translation is corrected back to the *full* constellation's
centroid so a dropped marker doesn't displace the reported position.

The result, `RigidBodyTrajectory`, exposes `.matrices()` — `(n, 4, 4)`
transforms ready for the next step — and `.residual_mm`, the per-frame RMS
fit residual, which is itself a useful measure of your reference's own
precision.

## 3. Reference resampling — `vipose.sync.resample`

Once you have an offset, resample the reference onto the track's own
timestamps:

```python
resample_reference(reference: MocapData, target_time_ms: np.ndarray,
                    offset_ms: float) -> MocapData
```

This interpolates **each marker coordinate linearly**, then the rigid-body fit
is applied *afterwards* — markers are interpolated first and fitted second,
not the reverse, because the fit is nonlinear and interpolating already-fitted
poses does not give the same answer. It raises rather than silently
clamping if `target_time_ms` falls outside the reference's span at the given
offset — a `ValueError` here almost always means the offset or the window is
wrong, not that anything needs relaxing.

## 4. Hand-eye calibration — `vipose.calibration`

The camera reports its pose in its own SLAM world frame `W`; the reference
reports the marker cluster in a lab frame `V`. Relating them needs two
further, constant transforms — `T_V_W` (world to lab) and `T_M_C` (camera to
marker) — recovered by solving the robot-world/hand-eye problem:

```python
solve_hand_eye(camera_in_world: np.ndarray,   # (n, 4, 4), T_W_C(t)
               marker_in_vicon: np.ndarray,   # (n, 4, 4), T_V_M(t)
               *, method: int = SHAH) -> HandEyeResult
```

Both arguments are plain `(n, 4, 4)` stacks — build them from a translation
and a rotation-vector array with `vipose.to_matrices`. Frames with a NaN or
singular rotation are dropped automatically, so an occluded reference frame
removes a sample rather than poisoning the fit; it raises if fewer than three
frames remain.

**Know this before you use it:** OpenCV's `calibrateRobotWorldHandEye`
satisfies `A·Z = X·B`, not the more commonly assumed `A·X = Z·B` — getting
this backwards yields a calibration roughly 106° wrong for no obvious reason.
`solve_hand_eye` handles the inversion for you; `tests/test_conditioning.py`
pins the convention down empirically if you want to see it demonstrated.

`method` defaults to Shah's method (`SHAH`, i.e.
`cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH`). It was chosen because its
well-posedness can be measured directly on the matrix it factorizes, rather
than by comparing the residual it happens to produce — the next section
covers that measurement. OpenCV implements other methods behind the same
`method=` argument if you want to compare.

### Is the motion well-posed for this? — `vipose.conditioning`

A hand-eye solve is only determined if the reference rotates about more than
one axis:

```python
shah_margin(rotations: np.ndarray, *, tol: float = 1e-9) -> float   # (n, 3, 3)
```

Returns `1 - σ₂` of the matrix Shah's method factorizes, computed from the
reference rotations alone — `0` marks a degenerate, single-axis motion; a
single-axis synthetic motion reads `0.000`, two independent axes read
`0.0234`, and an isotropic random rotation reads `0.3270`. If you're
designing a recording protocol for your own rig, checking this on a trial
recording before committing to a full session is cheap and catches "the probe
was basically only pivoting" before it becomes an uninterpretable
calibration.

## 5. Residual metrics — `vipose.metrics`

Once both trajectories are expressed in the same frame, the per-frame error
is two pure functions on `(n, 4, 4)` pose stacks:

```python
translational_residual(a, b) -> np.ndarray   # mm, Euclidean distance
rotational_residual(a, b) -> np.ndarray       # deg, angle of the relative rotation
summarize(values) -> Summary                  # n, median, iqr, p95, mean, sd, maximum
```

`rotational_residual` deliberately goes through `Rotation.from_matrix(...).magnitude()`
rather than `arccos((tr R − 1) / 2)`: the two agree analytically, but arccos of
the trace loses precision near zero rotation, which matters when residuals are
a fraction of a degree. `summarize` raises on an empty or non-finite series
rather than reporting a misleading blank — in this kind of pipeline, an empty
window is a bug in the caller, not a result worth a NaN.

## Putting it together: your own trajectory against a marker reference

`calibrate_pipeline` (in `vipose.pipeline`, re-exported from `vipose`) chains
steps 3–5 for one already-synchronized trajectory:

```python
calibrate_pipeline(track: Track, reference: MocapData, offset_ms: float,
                    window_ms: tuple[float, float])
    -> (Residuals, HandEyeResult, MocapData, RigidBodyTrajectory)
```

It takes a `Track` (any pipeline's pose CSV — `vipose.io.tracks.load_track`,
or build one by hand) and a `MocapData` reference, and has no dependency on
this repository's dataset manifest or on any particular marker layout. The
following runs end-to-end against one of the recordings this repository
ships, treating it as an example recording rather than as a paper number to
reproduce — the same six lines are what you'd run against your own camera
trajectory and your own motion-capture export:

```python
from vipose import (
    crosscorrelation_offset, refine_offset_omega, fit_rigid_body, calibrate_pipeline,
)
from vipose.io.tracks import load_track
from vipose.io.mocap import load_mocap
from vipose.metrics import summarize

track = load_track("datasets/probe-tracking-2025-10-23/tracking/zed-sdk/pivot/poses.csv")
reference = load_mocap(
    "datasets/probe-tracking-2025-10-23/reference/pivot/mocap.csv",
    marker_prefix="ZED:", expected_rate_hz=240.0,
)

# 1. Synchronize, against the reference's own rigid-body fit.
reference_fit = fit_rigid_body(reference)
offset_ms = refine_offset_omega(
    track.time_ms, track.rotvec, reference_fit.time_ms, reference_fit.rotvec,
    crosscorrelation_offset(
        track.time_ms, track.rotvec, reference_fit.time_ms, reference_fit.rotvec
    ),
)

# 2-4. Crop, calibrate, measure -- over the track's own full span here; use a
# narrower window if you only trust part of the recording.
window_ms = (float(track.time_ms[0]), float(track.time_ms[-1]))
residuals, calibration, _, _ = calibrate_pipeline(track, reference, offset_ms, window_ms)

print(summarize(residuals.d_trans_mm))
print(summarize(residuals.d_rot_deg))
```

This reproduces stages 1 and 2 only. `vipose.pipeline.evaluate_recording` adds
a third refinement stage and, when more than one pipeline shares a recording,
iterates it against a shared evaluation window — see its docstring if you
have multiple trajectories to compare against the same reference and want
that refinement; it takes a `dataset` argument today only for a cosmetic
label, not because it needs one.

If your reference is already pose-level (no marker cluster to fit — another
SLAM system, a tracked robot arm), skip steps 2–3 above: synchronize directly
on its rotation-vector stream, then build its `(n, 4, 4)` pose stack with
`vipose.to_matrices` and pass both stacks straight to `solve_hand_eye`.
