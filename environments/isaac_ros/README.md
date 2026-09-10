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

## No image is distributed

Only configuration is in this repository: `repos.yaml` pins the upstream
sources, `patches/` and `files/` are the modifications, and the image is built
locally by NVIDIA's own `build_image_layers.sh` from the Dockerfiles it composes.
Nothing here redistributes an NVIDIA image or an NVIDIA-derived layer, so the
NVIDIA EULA is between you and NVIDIA, as it should be.

`run-container.sh` is the interactive path and builds the image if it is
missing. `ensure-container.sh` is the scripted path and deliberately refuses to
build: if the image is absent it says so and points at `run-container.sh`,
rather than silently starting a 40 GB build inside a pipeline run.

## The container must be ready before the pipeline starts

Both entrypoint additions run a `colcon build` on **every** container start --
`01_my_setup.user.sh` builds `isaac_ros_visual_slam`, `zed-entrypoint.sh` builds
`zed_wrapper` -- so a freshly started container spends about a minute rebuilding
the workspace before it is usable.

Checking for `install/setup.bash` is not a readiness test. That file survives
from the previous build, so it exists immediately while the libraries beside it
are being replaced. Launching then fails with missing `.so` files and missing
Python modules, which looks like a broken image and is not.

`ensure-container.sh` instead waits for PID 1 to become the container command.
`workspace-entrypoint.sh` ends with `exec gosu admin "$@"`, so PID 1 *is* the
entrypoint until every addition has finished and *becomes* the command
afterwards. That is exact, and it needs no log scraping.

Note also that `docker exec` sessions do not inherit anything the entrypoint set
at runtime -- they start from the image's static `ENV`. This is why every
command run in the container sources `install/setup.bash` first, which chains to
the ROS underlay recorded at build time. The original scripts do the same, and
it is the reason they work.

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
