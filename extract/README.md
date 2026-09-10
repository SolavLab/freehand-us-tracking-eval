# Extraction stages

Everything here turns raw sensor recordings into the pose CSVs under
`datasets/`. It is not needed to use `vipose` on pose CSVs you already have —
see [`docs/usage.md`](../docs/usage.md) and [`docs/algorithms.md`](../docs/algorithms.md).
This is for regenerating those CSVs from scratch, or adapting the same
extraction path to a different rig.

`extract/` is deliberately separate from `src/vipose/`. It must not `import
vipose`, and `vipose` must not import it: they run under different
interpreters, and in one case inside a container. The only contract between
them is `datasets/*/schema/poses.schema.json`, which the evaluation validates
on every load — so a change in an extractor surfaces as a load-time schema
error rather than as a quietly different number.

| Directory | Environment (see `../docs/reproduction/setup-environments.md`) |
|---|---|
| `zed_sdk/` | conda `zed-sdk`: Python 3.12, pyzed 5.3, ZED SDK 5.3.0 |
| `convert/` | `ros-convert`: ROS 2 Humble sourced first, then the venv |
| `cuvslam/` | inside the Isaac ROS container |
| `common/` | any Python 3 with numpy and scipy |

## Pose estimation is not bit-reproducible

Both the ZED SDK and cuVSLAM run on the GPU, and neither is deterministic.
Re-running `zed_sdk/zed_sdk_trajectory.py` on the pivot recording reproduced
the same row count, frame range and timestamps, and the first 1812 frames
were bit-identical to the shipped trajectory; divergence began exactly where
the SDK's internal spatial-memory state also diverged, after which the median
difference was 2.9 mm. That is comparable to the trial-to-trial variability
between two acquisitions of the same condition (0.05–0.94 mm, 0.019–0.037°),
and it's why the shipped pose CSVs are pinned rather than regenerated for
every run — see `docs/reproduction/` for the full measurement if you're
auditing it.

## Pipeline I — ZED SDK

```bash
conda activate zed-sdk
python extract/zed_sdk/zed_sdk_trajectory.py "$VIPOSE_RAW_ROOT/pivot/test2.svo2" \
    -o datasets/probe-tracking-2025-10-23/tracking/zed-sdk/pivot/poses.csv
```

Runs the SDK's positional-tracking module over the recording in offline
playback, writing one row per frame with a valid pose: `GEN_3`, spatial
memory and IMU fusion enabled, pose smoothing off, `DEPTH_MODE.NONE`,
right-handed Y-up. The script is headless — no viewer is instantiated, since
tracking runs inside `grab()` regardless of whether anything is displayed.

The `--area-file` option loads a prebuilt spatial-memory map; the recordings
shipped here did not use one — spatial memory was built fresh each time.

## Pipelines II and III — cuVSLAM

```bash
# once
cd environments/isaac_ros
vcs import ../../src < repos.yaml && ./apply-patches.sh && ./run-container.sh
./build-workspace.sh

# per recording, from the host
extract/cuvslam/run_zed_svo2.sh      pivot /workspaces/isaac_ros-dev/test2.svo2
extract/cuvslam/run_realsense_bag.sh pivot /workspaces/isaac_ros-dev/test2_converted
```

### The three phases

1. **Play back and track.** `launch/isaac_ros_visual_slam_zed_svo2.launch.py`
   for the ZED SVO2 path, `..._rosbag.launch.py` for converted RealSense
   bags. Both preserve acquisition timestamps (`use_sim_time: true`) and play
   at **40% of real time** with enlarged QoS and cuVSLAM buffers, to stop the
   SLAM node dropping frames under load.

   Both launch files began as NVIDIA Apache-2.0 templates and were
   substantially rewritten; NVIDIA's notice is retained and each carries a
   statement of what changed, as Apache-2.0 requires.

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

cuVSLAM returns a trajectory optimised over the whole sequence, unlike
pipeline I's frame-by-frame estimate — part of the measured difference
between pipelines follows from that asymmetry, and neither could apply
full-sequence optimisation in a real-time setting.

## Stage B — RealSense bag conversion

`convert/realsense_bag_to_rosbag2.py` turns a native RealSense `.bag` into a
rosbag2 directory publishing `/camera/{left,right}/image_raw`,
`/camera/{left,right}/camera_info` and `/imu`. Images are `mono8` at
**1280×800**; IMU keeps its native ~400 Hz by driving **two independent
playback pipelines over the same file** — one with only the infrared
streams, one with only accelerometer and gyroscope. Enabling IMU in the
frameset pipeline throttles it to camera rate, discarding most of the
inertial data.

```bash
source /opt/ros/humble/setup.bash        # must come before the venv
source .venv-convert/bin/activate
python extract/convert/realsense_bag_to_rosbag2.py \
    "$VIPOSE_RAW_ROOT/pivot/test2.bag" -o /path/to/pivot_converted
```

If you adapt this converter: `librealsense` raises the same `RuntimeError`
for a genuine end-of-playback and for a transient read timeout, so a naive
per-pipeline timeout can silently truncate the IMU stream on one slow read
while the frameset stream keeps going and the run still reports success.
`_imu_wait()` here retries a timeout up to three times before concluding the
stream has actually ended, and `_validate_output()` measures the written
bag's image and IMU rates and raises if either is implausibly low, rather
than checking only that the output files exist.

## Device calibration is baked in

The launch files carry **this rig's measured** extrinsics: the ZED stereo
baseline, the RealSense stereo baseline, and the RealSense IMU offsets. They
are not vendor defaults — change them for other hardware, or the SLAM will be
solving a slightly wrong geometry.
