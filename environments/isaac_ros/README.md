# Isaac ROS container for the cuVSLAM pipelines

Pipelines II (`zed-cuvslam`) and III (`rs-cuvslam`) run cuVSLAM 3.2 inside a
Docker image built from NVIDIA's Isaac ROS Development Environment, modified to
carry the ZED and RealSense stacks in one image.

Upstream sources are **not** vendored here. `repos.yaml` pins them, the patches
and files below are the modifications, and `docs/licensing.md` explains why.

```bash
vcs import ../../src < repos.yaml
./apply-patches.sh
./build-image.sh          # -> ros2_humble.realsense.zed.custom
```

## What the modifications are, and why each is needed

| | |
|---|---|
| `patches/0001-zed-and-realsense-image-layers.patch` | The delta actually used to build the published image. Two files. |
| `patches/0002-harden-zed-sdk-download.patch` | Security fixes, kept separate so 0001 stays faithful. |
| `files/docker/Dockerfile.zed` | New ZED layer. |
| `files/docker/scripts/zed-entrypoint.sh` | New; runs at container start. |
| `files/docker/custom/Dockerfile.custom` | New; entrypoint-additions layer. |
| `files/docker/custom/01_my_setup.user.sh` | New; the entrypoint addition itself. |

**`docker/Dockerfile.realsense` (+15 lines).** Adds NVIDIA's Isaac ROS apt
repository and its GPG keyring to the RealSense layer, and sets
`Acquire::AllowReleaseInfoChange` so an upstream change to the repository's
Origin/Label/Suite does not stop a non-interactive build. Without this the layer
cannot install the Isaac ROS NITROS packages the visual-SLAM node needs.

**`docker/scripts/install-zed-x86_64.sh` (rewritten).** Upstream derives the ZED
SDK version, CUDA version and Ubuntu release at build time and installs SDK 4.2.
That is replaced by a pinned URL for **ZED SDK 5.1, cu12, Ubuntu 22**, plus
installation of `ros-humble-zed-msgs` and the other wrapper dependencies. Pinning
is the point: a build-time-derived version makes the image un-reproducible, and
the wrapper needs an SDK matching the one the recordings were made with.

> The SDK in this container is **5.1**, and it is used only to decode SVO2 files
> and publish stereo and IMU topics. Pipeline I is a separate thing entirely: it
> runs on the host under **ZED SDK 5.3.0** and uses that SDK's positional-tracking
> module. Both numbers are correct; they belong to different pipelines.

**`docker/Dockerfile.zed`** layers the ZED SDK onto the RealSense image, so one
container can play back both SVO2 and ROS 2 bags.

**`docker/scripts/zed-entrypoint.sh`** fixes ownership under `/usr/local/zed` and
builds the ZED wrapper with rosdep and colcon on first start.

**`docker/custom/`** adds an entrypoint-additions layer: scripts dropped into
`/usr/local/bin/scripts/entrypoint_additions/` are sourced in filename order at
container start. `01_my_setup.user.sh` sources ROS 2 Humble if it is not already
sourced and refreshes the rosdep index.

The resulting image key is `ros2_humble.realsense.zed.custom`.

## Notes for anyone rebuilding this

- **The image is x86-64 only.** The ZED SDK URL and the install script are
  architecture-specific. The recordings were *made* on a Jetson Orin NX, but all
  offline processing ran on an x86-64 workstation.
- **Two host mounts are required** so the SDK finds its calibration and
  resources: `/usr/local/zed/settings` and `/usr/local/zed/resources`.
- **Device-specific extrinsics are baked into the launch files** in
  `extract/cuvslam/launch/` — the ZED stereo baseline (`±0.02486265` m), the
  RealSense baseline (`-0.09495820850133896` m) and the RealSense IMU offsets.
  These are *this rig's* calibration, not defaults. Change them for other
  hardware.
- **Playback runs at 40 % of real time** (`svo.replay_rate: 0.4`, and
  `ros2 bag play --rate 0.4`), with `use_sim_time: true` so the original
  acquisition timestamps are preserved. This is to stop the visual-SLAM node
  dropping frames under load, and the paper reports it as a limitation for any
  real-time claim rather than as a tuning choice.
- One comment in patch 0001 is a working note recording where a fix came from.
  It is left in the patch because the patch is a record of what was run; it is
  not reproduced in this README.
