# Data formats

Every file this project reads or writes, with its units, frames and the
assumptions the code makes about it.

---

## 1. Vicon reference — `datasets/*/reference/<recording>/mocap.csv`

A **Vicon Nexus 2.16 "Trajectories" export**, unmodified. It is not a plain CSV:
the first six lines are a header block.

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

- **Markers**: only columns whose group name carries the prefix `ZED:` are read.
  The prefix is configurable in `dataset.yaml` (`vicon.marker_prefix`). Five
  markers form the rigid cluster: `TOP`, `BL`, `BR`, `FR`, `FL`.
- **Units**: millimetres, declared on line 7 and checked on load.
- **Sample rate**: 240 Hz, on line 4.

  > The original loader **hard-coded** 240.0 and never read line 4, behind a stale
  > comment saying 120 Hz. The rate propagates into every timestamp, every
  > interpolation and every reported lag, so changing where it comes from would
  > be a re-analysis rather than a refactor. This release keeps 240.0 as the
  > authoritative value in `dataset.yaml` and **asserts** that line 4 agrees with
  > it to within 0.1 %. The assumption is now checked rather than assumed.

- **Gaps**: unlabelled frames appear as empty fields. The rigid-body fit needs at
  least four of the five markers.

### Derived: rigid-body pose

`vipose.geometry.rigid_body` fits the marker cluster with the Kabsch algorithm,
giving the marker pose `T_V_M(t)` and, per frame, the RMS residual of the fit.
Those per-frame residuals are what the paper's Vicon-quality table reports
(mean below 0.28 mm, max below 0.74 mm across the four recordings).

---

## 2. Pipeline trajectories — `datasets/*/tracking/<pipeline>/<recording>/poses.csv`

The common interchange format all three pipelines are normalised to. The
authoritative definition, including per-column semantics, is
`datasets/probe-tracking-2025-10-23/schema/poses.schema.json`; it is validated on
every load.

| Column | Unit | Notes |
|---|---|---|
| `Frame` | — | Row index. **Not** a frame identifier for the cuVSLAM pipelines |
| `Timestamp` | ms | Acquisition-platform clock. The authoritative time base |
| `Translation_X/Y/Z` | mm | Camera position in the SLAM world frame `W` |
| `Rotation_X/Y/Z` | rad | **Rotation vector** (axis–angle), not Euler angles |
| `Pose_Confidence` | — | ZED SDK only, 0–100; empty for cuVSLAM |
| `Tracking_State` | — | ZED SDK only |
| `Spatial_Memory` | — | ZED SDK state, or the literal `cuVSLAM` as a provenance marker |
| `Odometry`, `Tracking_Fusion` | — | ZED SDK only |

Two traps are worth stating explicitly, because both have bitten this project:

- **`Rotation_*` are rotation vectors.** Read them with
  `scipy.spatial.transform.Rotation.from_rotvec`. Reading them as Euler angles
  produces an orientation-dependent gain error of up to 20 % on every rotational
  metric — and because the pipelines use different camera-frame conventions, the
  error is *pipeline-dependent*, which is the worst possible shape for a
  comparison. A lint rule forbids `from_euler` anywhere in `src/`.
- **`Frame` cannot be used to find dropped frames.** cuVSLAM renumbers its output
  sequentially, so an input frame it dropped leaves no gap in `Frame` and
  index-continuity finds zero omissions for every pipeline. Omissions must be
  counted from `Timestamp`, as `round(dt / median(dt)) - 1` summed over the
  window. Repeated poses are a different phenomenon: exact equality of all six
  pose components with the previous row.

### Camera frames differ by pipeline

| Pipeline | Reported frame |
|---|---|
| `zed-sdk` | left optical sensor, ZED SDK right-handed Y-up convention |
| `zed-cuvslam` | configured base frame (`base_link`), ROS convention |
| `rs-cuvslam` | configured base frame (`camera_link`), ROS convention |

These differ by constant rigid transformations, which the hand–eye calibration
absorbs. They do not affect the reported residuals.

---

## 3. Results — `results/<run-id>/`

```
run.yaml                                   code version, env, dataset hash, CLI args
recordings/<recording>/window.json         shared evaluation window, per-pipeline offsets
recordings/<recording>/reference_fit.json  Vicon rigid-body residual statistics
cells/<pipeline>/<recording>/residuals.csv frame, time_ms, d_trans_mm, d_rot_deg
cells/<pipeline>/<recording>/calibration.json
cells/<pipeline>/<recording>/cell.json     status, n_frames, input hashes, summary stats
```

`residuals.csv` carries the two per-frame error series:

- `d_trans_mm` — Euclidean distance between the SLAM-based and marker-based
  camera positions in the Vicon frame.
- `d_rot_deg` — the rotation angle of the relative rotation between the two
  estimated orientations.

`calibration.json` holds `T_camera_to_marker`, `T_slamworld_to_vicon`, the
marker-intrinsic frame, the temporal offset, and every synchronization stage, as
JSON numbers.

> The original wrote the transforms as a text dump of numpy `repr`, which nothing
> can read back. Downstream consumers therefore re-derived quantities that had
> already been computed. JSON ends that.

### Frames and the calibration

Four frames are used throughout, matching the manuscript:

| Symbol | Frame |
|---|---|
| `V` | Vicon laboratory frame |
| `M` | infrared marker cluster |
| `C` | camera frame the pipeline reports |
| `W` | SLAM world frame, initialised per recording when tracking begins |

The calibration solves `T_V_W · T_W_C(t) = T_V_M(t) · T_M_C` for the two constant
transforms, with OpenCV's `calibrateRobotWorldHandEye` and the Shah method.

> **Convention.** OpenCV's function satisfies `A·Z = X·B` and returns the pair
> `(Z, X)`. In this project's notation that is
> `T_C_W(t)·T_W_V = T_C_M·T_M_V(t)`. Assuming `A·X = Z·B` instead yields a
> calibration roughly 106° wrong. `tests/test_opencv_convention.py` pins this
> down empirically against six candidate laws.

---

## 4. Raw recordings — not shipped

Catalogued in `datasets/probe-tracking-2025-10-23/raw.yaml`; ~20 GiB across the
five sessions. Resolved at runtime from `$VIPOSE_RAW_ROOT`.

| Kind | Format | Consumed by |
|---|---|---|
| ZED stereo + IMU | `.svo2`, 1920×1200 @ 60 Hz, IMU 400 Hz | pipelines I and II |
| RealSense IR stereo + IMU | `.bag`, 1280×800 mono8 @ 30 Hz, IMU 400 Hz | pipeline III |
| ZED spatial-memory map | `.area` | **nothing** — see below |

The `.area` files are listed for completeness but are never loaded. Pipeline I
enables spatial memory and builds it fresh from each recording; the original
driver script computed an area-file path and then did not pass it to the
extractor.

### Intermediate: converted ROS 2 bags

`extract/convert/` turns each RealSense `.bag` into a rosbag2 (sqlite3)
directory publishing `/camera/{left,right}/image_raw`,
`/camera/{left,right}/camera_info` and `/imu`. Images are `mono8` at 1280×800;
IMU keeps its native ~400 Hz, which is why the converter drives two independent
playback pipelines over the same file — enabling IMU in the frameset pipeline
would throttle it to camera rate.
