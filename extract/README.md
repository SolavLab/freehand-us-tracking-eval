# Extraction stages

Everything here turns raw recordings into the pose CSVs under `datasets/`. It is
**rung R3** of the reproduction ladder: needed only to regenerate the pose
trajectories from scratch, not to reproduce any number in the paper.

`extract/` is deliberately separate from `src/vipose/`. It must not `import
vipose`, and `vipose` must not import it: they run under different interpreters,
and in one case inside a container. The only contract between them is
`datasets/*/schema/poses.schema.json`, which the evaluation validates on every
load — so a change in an extractor surfaces as a load-time schema error rather
than as a quietly different number.

| Directory | Environment (see `../docs/setup.md`) |
|---|---|
| `zed_sdk/` | conda `zed-sdk`: Python 3.12, pyzed 5.3, ZED SDK 5.3.0 |
| `convert/` | `ros-convert`: ROS 2 Humble sourced first, then the venv |
| `cuvslam/` | inside the Isaac ROS container |
| `common/` | any Python 3 with numpy and scipy |

## These stages are not bit-reproducible

cuVSLAM and the ZED SDK both run on the GPU and neither is deterministic. This
was measured rather than assumed.

Re-running `zed_sdk/zed_sdk_trajectory.py` on the pivot recording produced
**exactly the same 3369 rows, the same frame range, the same columns and
timestamps identical to the millisecond**, and the first **1812 frames were
bit-identical** to the published trajectory. Divergence begins at frame 1812,
30.2 s in — at precisely the frame where the SDK's own `Spatial_Memory` state
also differs, i.e. a spatial-memory event landed at a slightly different moment
and the internal SLAM state diverged from there. After that point the median
difference is 2.9 mm and the maximum 7.0 mm.

1812 consecutive bit-identical frames is not a coincidence, so the decode and
tracking path is the same; what differs is the SDK's internal non-determinism.
The divergence is comparable to the trial-to-trial variability the paper already
reports (0.05–0.94 mm, 0.019–0.037° between two acquisitions of one condition),
and it is why the paper's numbers are pinned to the shipped CSVs rather than
regenerated.

## Pipeline I — ZED SDK

```bash
conda activate zed-sdk
python extract/zed_sdk/zed_sdk_trajectory.py "$VIPOSE_RAW_ROOT/pivot/test2.svo2" \
    -o datasets/probe-tracking-2025-10-23/tracking/zed-sdk/pivot/poses.csv
```

Runs the SDK's positional-tracking module over the recording in offline
playback, writing one row per frame with a valid pose. `GEN_3`, spatial memory
and IMU fusion enabled, pose smoothing off, `DEPTH_MODE.NONE`, right-handed
Y-up; everything else at its default.

This script is **headless**. The original drove its grab loop from the render
callback of a Stereolabs OpenGL viewer and imported roughly 1,600 lines of
vendored sample code to do it. None of that affects the poses — the viewer only
displayed them, playback is offline, depth is disabled, and tracking runs inside
`grab()` — so it is gone, which also removes the redistribution question.

The `--area-file` option loads a prebuilt spatial-memory map. **The published
runs did not use one**; spatial memory was built fresh from each recording. The
original driver script computed an area-file path and then never passed it.

## Pipelines II and III — cuVSLAM

Three phases, all inside the container (`../environments/isaac_ros/`):

1. **Play back and track.** `launch/isaac_ros_visual_slam_zed_svo2.launch.py`
   for the ZED SVO2 path, `..._rosbag.launch.py` for the converted RealSense
   bags. Both preserve the acquisition timestamps (`use_sim_time: True`) and
   play at **40 % of real time** with enlarged QoS and cuVSLAM buffers, to stop
   the SLAM node dropping frames. The paper reports that rate as a limitation
   for any real-time claim, not as a tuning choice.

   Both launch files began as NVIDIA Apache-2.0 templates and were substantially
   rewritten. NVIDIA's notice is retained, as Apache-2.0 requires, and each
   carries a statement of what was changed, as section 4(b) requires.

2. **Capture the optimised trajectory.** When playback ends,
   `monitor_svo_end.py` (ZED) or an `OnProcessExit` handler (bag) calls the
   `GetAllPoses` service and writes the response to a text file.
   `export_poses.py` parses it into `poses.csv`.

3. **Convert to the common schema.** `common/tracks_to_common_csv.py` turns
   metres and quaternions into millimetres and rotation vectors, and stamps
   `Spatial_Memory = "cuVSLAM"` as a provenance marker.

```bash
python extract/cuvslam/export_poses.py /tmp/poses_response.txt -o /tmp/poses.csv
python extract/common/tracks_to_common_csv.py /tmp/poses.csv \
    -o datasets/probe-tracking-2025-10-23/tracking/rs-cuvslam/pivot/poses.csv
```

Unlike pipeline I, cuVSLAM returns a trajectory optimised over the whole
sequence. That asymmetry is a stated limitation of the comparison: part of the
measured difference between the pipelines follows from it, and neither could
apply full-sequence optimisation in a real-time setting.

## Stage B — RealSense bag conversion

`convert/realsense_bag_to_rosbag2.py` turns a native RealSense `.bag` into a
rosbag2 directory publishing `/camera/{left,right}/image_raw`,
`/camera/{left,right}/camera_info` and `/imu`. Images are `mono8` at
**1280×800**; IMU keeps its native ~400 Hz.

Keeping the IMU at its native rate is why the converter drives **two independent
playback pipelines over the same file** — one with only the infrared streams,
one with only accelerometer and gyroscope. Enabling IMU in the frameset pipeline
throttles it to the camera rate, which would have thrown away more than nine
tenths of the inertial data.

```bash
source /opt/ros/humble/setup.bash        # must come before the venv
source .venv-convert/bin/activate
python extract/convert/realsense_bag_to_rosbag2.py \
    "$VIPOSE_RAW_ROOT/pivot/test2.bag" -o /path/to/pivot_converted
```

### The IMU playback race — read this before re-converting

**On this machine today, conversion does not reliably reproduce the bags the
published results were built from, and the failure used to be silent.**

The two playback pipelines are opened over the same file concurrently. That is
what keeps the IMU at its native rate, and it is also timing dependent. Measured
on the pivot recording:

| | images/side | IMU | implied IMU rate |
|---|---|---|---|
| reference bag (April 2026) | 1677 | 22262 | 398 Hz — correct |
| original converter, today | 1677 | 1942 | 35 Hz |
| this converter, today | 1677 | 1635 | 29 Hz |

The camera stream is fine and reproduces exactly. The IMU is not: it comes back
at roughly the *camera* rate, which is precisely the throttling the two-pipeline
design exists to avoid. The original code and this port fail the same way, so
this is a librealsense or environment change rather than a regression in the
port — and the reference bags themselves are correct, so nothing in the paper is
affected.

What was silent: the driver script checked only that the output files existed.
A bag holding a tenth of its inertial data was therefore reported as a
successful conversion, and would have gone on to starve the visual-inertial SLAM
with no indication anything was wrong.

`_validate_output()` now checks the written bag's actual image and IMU rates
against the recording's span and **raises** if either falls below half the
expected rate. Re-converting today fails with:

```
Error: the converted bag does not contain the expected streams:
  - IMU at 29.3 Hz over 55.9 s, expected about 400 Hz (1635 written).
```

which is the correct outcome. If you need to regenerate these bags, expect to
investigate the playback race first; pinning `pyrealsense2` to the version that
produced the reference bags is the obvious first thing to try.

## Device calibration is baked in

The launch files carry **this rig's measured** extrinsics: the ZED stereo
baseline, the RealSense stereo baseline (−0.09495820850133896 m) and the
RealSense IMU offsets. They are not vendor defaults. Change them for other
hardware, or the SLAM will be solving a slightly wrong geometry.
