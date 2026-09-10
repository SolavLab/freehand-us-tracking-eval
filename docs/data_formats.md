# Data formats

Every file `vipose` reads or writes, with its units, frames and the
assumptions the code makes about it. Use this page as the format spec if
you're pointing `vipose` at your own recording.

---

## 1. Marker-cluster reference — `datasets/*/reference/<recording>/mocap.csv`

A **Vicon Nexus 2.16 "Trajectories" export**, unmodified. It is not a plain
CSV: the first six lines are a header block.

```
1  (blank)
2  (blank)
3  Trajectories
4  240                                       <- sample rate, Hz
5  ,,ZED:TOP,,,ZED:BL,,,ZED:BR,,,ZED:FR,,,ZED:FL,,   <- marker names, one per 3 columns
6  Frame,Sub Frame,X,Y,Z,X,Y,Z,...           <- column roles
7  ,,mm,mm,mm,mm,mm,...                      <- units
8+ data
```

- **Markers**: only columns whose group name carries a configurable prefix
  (`vicon.marker_prefix` in `dataset.yaml`, `ZED:` for the four recordings
  shipped as example data) are read. The rigid-body fit needs at least four
  markers in a given frame; five ship here (`TOP`, `BL`, `BR`, `FR`, `FL`).
- **Units**: millimetres, declared on line 7 and checked on load.
- **Sample rate**: declared on line 4 and checked against the value in
  `dataset.yaml` (`vicon.nominal_rate_hz`) to within 0.1%, since the rate
  propagates into every timestamp, interpolation and reported lag — a
  recording whose file and manifest disagree fails loudly rather than
  guessing which one is right.
- **Gaps**: unlabelled frames appear as empty fields.

`load_mocap()` (`vipose.io.mocap`) parses this into a `MocapData`; see
[algorithms.md](algorithms.md) for what's built on top of it.

### Derived: rigid-body pose

`vipose.geometry.rigid_body.fit_rigid_body` fits the marker cluster with the
Kabsch algorithm, giving the marker cluster's pose `T_V_M(t)` and, per frame,
the RMS residual of the fit — a useful measure of the reference's own
precision (mean below 0.28 mm, max below 0.74 mm across the four recordings
shipped here).

---

## 2. Trajectory CSVs — `datasets/*/tracking/<pipeline>/<recording>/poses.csv`

The interchange format any pose-tracking pipeline is normalised to before
`vipose` compares it against a reference. The authoritative definition is
`datasets/probe-tracking-2025-10-23/schema/poses.schema.json`; `load_track()`
validates every file against it on load, so a malformed input surfaces as a
load-time schema error rather than a silently wrong number downstream.

| Column | Unit | Notes |
|---|---|---|
| `Frame` | — | Row index. Not reliable as a frame identifier across all trackers — see below |
| `Timestamp` | ms | Acquisition-platform clock. The authoritative time base |
| `Translation_X/Y/Z` | mm | Camera position in the tracker's own world frame |
| `Rotation_X/Y/Z` | rad | **Rotation vector** (axis–angle), not Euler angles |
| `Pose_Confidence` | — | Optional, 0–100 |
| `Tracking_State` | — | Optional, free-form |
| `Spatial_Memory` | — | Optional, tracker state or a provenance marker string |
| `Odometry`, `Tracking_Fusion` | — | Optional |

Only the first six columns are required (`REQUIRED_COLUMNS` in
`vipose/io/tracks.py`); everything else is carried through untouched as
diagnostic state and ignored by the evaluation.

Two things to get right when producing your own `poses.csv`:

- **`Rotation_*` are rotation vectors.** Read and write them with
  `scipy.spatial.transform.Rotation.from_rotvec`/`.as_rotvec()`. Reading them
  as Euler angles produces an orientation-dependent gain error of up to 20%
  on every rotational metric, which is the worst possible shape for a
  comparison across trackers that use different conventions. `vipose`
  forbids `from_euler` anywhere in `src/` by lint rule for this reason.
- **`Frame` is not always a reliable index for dropped frames.** Some
  trackers renumber their output sequentially, so a dropped input frame
  leaves no gap in `Frame` and index-continuity finds zero omissions.
  Omissions are better counted from `Timestamp`, as
  `round(dt / median(dt)) - 1` summed over the window; an exact repeat of all
  six pose components from the previous row indicates a stagnant estimate
  rather than a dropped one.

### Camera frame conventions of the pipelines shipped here

| Pipeline | Reported frame |
|---|---|
| `zed-sdk` | left optical sensor, ZED SDK right-handed Y-up convention |
| `zed-cuvslam` | configured base frame (`base_link`), ROS convention |
| `rs-cuvslam` | configured base frame (`camera_link`), ROS convention |

These differ by constant rigid transformations, which the hand–eye
calibration absorbs — they do not need to match for `vipose` to compare
trajectories reported in different frames.

---

## 3. Results — `results/<run-id>/`

```
run.yaml                                   code version, env, dataset hash, CLI args
recordings/<recording>/window.json         shared evaluation window, per-pipeline offsets
recordings/<recording>/reference_fit.json  reference rigid-body residual statistics
cells/<pipeline>/<recording>/residuals.csv
cells/<pipeline>/<recording>/calibration.json
cells/<pipeline>/<recording>/cell.json     status, n_frames, input hashes, summary stats
```

`residuals.csv` has five columns:

| Column | Unit | Meaning |
|---|---|---|
| `frame` | — | Position within the evaluation window, `0..n-1` |
| `source_frame` | — | The pipeline's own frame index, as it appears in `poses.csv` |
| `time_ms` | ms | Acquisition clock, absolute epoch milliseconds |
| `d_trans_mm` | mm | Euclidean distance between the two estimated camera positions, in the reference frame |
| `d_rot_deg` | deg | Angle of the relative rotation between the two estimated orientations |

### Two time bases

`calibration.json` reports the evaluation window twice, and they are **not**
in the same units:

- `vicon_window_ms` and `local_window_ms` are **relative** milliseconds,
  measured from the first timestamp of the pipeline's full (uncropped)
  trajectory.
- `time_ms` in `residuals.csv` is **absolute** epoch milliseconds.

`local_window_ms` is a continuous requested interval; the rows in
`residuals.csv` are the discrete frames that fall inside it, so the first and
last `time_ms` sit just within the requested bounds rather than exactly on
them. Both fields are labelled with their unit for this reason.

`calibration.json` holds `T_camera_to_marker`, `T_slamworld_to_vicon`, the
marker-intrinsic frame, the temporal offset, and every synchronization stage,
as JSON numbers — nothing here needs re-deriving from a text dump.

### Frames and the calibration

Four frames are used throughout:

| Symbol | Frame |
|---|---|
| `V` | reference laboratory frame |
| `M` | tracked rigid marker cluster |
| `C` | camera frame the pipeline reports |
| `W` | tracker's own world frame, initialised per recording when tracking begins |

The calibration solves `T_V_W · T_W_C(t) = T_V_M(t) · T_M_C` for the two
constant transforms, with OpenCV's `calibrateRobotWorldHandEye` and the Shah
method — see [algorithms.md § Hand-eye calibration](algorithms.md) for the
convention this satisfies and why it's easy to get backwards.

---

## 4. Authoring your own recording

To point `vipose` at a new recording rather than the shipped example:

1. Produce a `mocap.csv` in the Vicon Nexus layout above (or write your own
   loader against `vipose.io.mocap.MocapData` if your reference system isn't
   Vicon — it's a small frozen dataclass, not a parser you need to reuse).
2. Produce one `poses.csv` per tracker, matching
   `schema/poses.schema.json` — `load_track()` will reject anything that
   doesn't.
3. If you're using the `vipose` CLI end-to-end rather than calling the
   library directly (see [algorithms.md](algorithms.md)), add an entry under
   `recordings:` in a `dataset.yaml`, and a `sha256`/`rows` entry under
   `files:` for each new file — `Dataset.load()` verifies both on every run.
   `dataset.yaml` in `datasets/probe-tracking-2025-10-23/` is a worked
   example of the full manifest shape.

---

## 5. Raw recordings — not in this repository

The SVO2 and `.bag` files behind the shipped example dataset are catalogued
in `datasets/probe-tracking-2025-10-23/raw.yaml`; ~20 GiB across the five
sessions, several GB per recording — too large for this repository, and
planned for a separate Zenodo deposit. Resolved at runtime from
`$VIPOSE_RAW_ROOT`. Regenerating pose CSVs from these is covered in
[extract/README.md](../extract/README.md); it is not needed to use `vipose`
on data you already have as pose/mocap CSVs.

| Kind | Format | Consumed by |
|---|---|---|
| ZED stereo + IMU | `.svo2`, 1920×1200 @ 60 Hz, IMU 400 Hz | `zed-sdk`, `zed-cuvslam` |
| RealSense IR stereo + IMU | `.bag`, 1280×800 mono8 @ 30 Hz, IMU 400 Hz | `rs-cuvslam` |
| ZED spatial-memory map | `.area` | not consumed — listed for completeness |

### Intermediate: converted ROS 2 bags

`extract/convert/` turns each RealSense `.bag` into a rosbag2 (sqlite3)
directory publishing `/camera/{left,right}/image_raw`,
`/camera/{left,right}/camera_info` and `/imu`. Images are `mono8` at
1280×800; IMU keeps its native ~400 Hz, which is why the converter drives two
independent playback pipelines over the same file — enabling IMU in the
frameset pipeline would throttle it to camera rate.
