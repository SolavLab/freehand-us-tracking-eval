# Setup

There are **four** environments in this project, and they are mutually
incompatible. You almost certainly need only the first.

Read the [reproduction ladder](#reproduction-ladder) first and stop at the rung
you actually need.

## Reproduction ladder

| Rung | What it regenerates | Environment | Time |
|---|---|---|---|
| **R1** | Every table and figure, from the per-frame residuals that ship in `datasets/` | `analysis` | seconds |
| **R2** | The full evaluation — synchronization, hand–eye calibration, residuals — from the pose and mocap CSVs that ship | `analysis` | ~20 min |
| **R3a** | Pipeline-I poses, from the raw SVO2 recordings | `zed-sdk` | hours |
| **R3b** | ROS 2 bags, from the raw RealSense `.bag` recordings | `ros-convert` | ~1 h |
| **R3c** | Pipeline-II and -III poses, via cuVSLAM | `isaac-ros` container | hours |

**R1 and R2 are what the paper's claims rest on, and they need nothing but
Python 3.10 and this repository.** R3 needs the ~20 GiB raw archive (see
`datasets/probe-tracking-2025-10-23/raw.yaml`) and vendor SDKs, and is not
bit-reproducible — cuVSLAM and the ZED SDK both run on the GPU and are not
deterministic. R3 is included so the acquisition path is inspectable and
re-runnable, not so it can be replayed to identical output.

---

## 1. `analysis` — the evaluation package (R1, R2)

Python **3.10**. This is the only environment that is pinned exactly, because it
is the only one whose output the paper's numbers depend on.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Pins live in `pyproject.toml`: numpy 2.2.6, scipy 1.15.3, pandas 2.3.3,
matplotlib 3.10.8, seaborn 0.13.2, opencv-python 4.13.0.92, PyYAML 6.0.3.

`requires-python` is capped at `<3.11` on purpose. Two of the four environments
below are Python 3.12, and running the evaluation under one of them would not
fail — it would silently produce different numerics. The cap turns that into an
install-time error.

Check it:

```bash
vipose verify --dataset datasets/probe-tracking-2025-10-23
pytest tests/ -v
```

`opencv-python` is the one pin worth understanding. The hand–eye calibration
calls `cv2.calibrateRobotWorldHandEye`, whose internal SVD sign conventions can
differ between OpenCV builds. If the golden-master test fails on transforms
while the summary statistics still match the paper, an OpenCV mismatch is the
first thing to check.

---

## 2. `zed-sdk` — pipeline I pose extraction (R3a)

Python **3.12** with the Stereolabs ZED SDK and its Python bindings. In the
original work this was a conda environment named `ZED_env`.

- **ZED SDK 5.3.0** — install from Stereolabs; this is not pip-installable.
- **pyzed 5.3** — the binding matching that SDK, installed by the SDK's
  `get_python_api.py`.

```bash
conda create -n zed-sdk python=3.12
conda activate zed-sdk
python /usr/local/zed/get_python_api.py     # installs pyzed matching the SDK
```

Verify that the binding is loading the SDK you think it is — the extractor
prints it, and this is worth checking because the version *is* one of the
systems under comparison:

```bash
python -c "import pyzed.sl as sl; print(sl.Camera().get_sdk_version())"
# expected: 5.3.0
```

> The recordings themselves were made on a Jetson Orin NX with **ZED SDK
> 5.0.7** under JetPack 6.2.1. That is the acquisition-side version and is
> independent of the version used here for offline pose estimation.

---

## 3. `ros-convert` — RealSense bag conversion (R3b)

Python **3.10**, ROS 2 **Humble**, plus `pyrealsense2`.

`rosbag2_py`, `rclpy` and `cv_bridge` come from the ROS 2 installation, not from
pip. Only `pyrealsense2` and `numpy` come from the virtual environment.

**Source order matters.** ROS 2 must be sourced *before* the virtual environment
is activated:

```bash
source /opt/ros/humble/setup.bash        # must come first
python3.10 -m venv --system-site-packages .venv-convert
source .venv-convert/bin/activate
pip install "pyrealsense2==2.56.5.9235" "numpy==1.26.4" "opencv-python==4.12.0.88"
```

Sourcing ROS 2 puts its site-packages on `PYTHONPATH`, which Python honours
regardless of whether the virtual environment includes system site-packages. Do
it the other way round and the ROS imports fail. In the original tree this
ordering was undocumented and the environment had
`include-system-site-packages = false`, so the converter worked only in a shell
where ROS 2 happened to already be sourced.

Check all four imports at once:

```bash
python -c "import pyrealsense2, rosbag2_py, rclpy, cv_bridge; print('ok')"
```

`pyrealsense2` is pinned to 2.56.5 to match the RealSense Viewer version that
made the recordings.

---

## 4. `isaac-ros` — cuVSLAM pipelines (R3c)

A Docker container built on NVIDIA's Isaac ROS Development Environment, modified
to carry both the ZED and the RealSense stacks. Nothing runs on the host.

Upstream sources are **not vendored** in this repository. Fetch them at the
pinned revisions and apply the modifications:

```bash
cd environments/isaac_ros
vcs import ../../src < repos.yaml     # pins isaac_ros_common @ v3.2-8-7-ga40a017
./apply-patches.sh                    # Dockerfile + launch-file modifications
./build-image.sh                      # -> ros2_humble.realsense.zed.custom
```

What the patches do, and why each is needed, is in
`environments/isaac_ros/README.md`. In summary: add the Isaac ROS apt repository
to the RealSense image layer; pin and install ZED SDK 5.1 for cu12/Ubuntu 22 with
the ROS ZED message packages; add a ZED layer and its entrypoint; and add an
entrypoint-additions layer.

Requirements: an x86-64 host, an NVIDIA GPU with a recent driver, Docker with
the NVIDIA container runtime, and roughly 40 GB of disk for the image.

---

## Which environment runs which script

Every executable in this repository names its interpreter in a header comment.
The mapping is:

| Path | Environment |
|---|---|
| `src/vipose/**`, `vipose` CLI, `tests/**`, `investigations/**` | `analysis` |
| `extract/zed_sdk/**` | `zed-sdk` |
| `extract/convert/**` | `ros-convert` |
| `extract/cuvslam/**` | `isaac-ros`, inside the container |

Do not put `extract/` on `PYTHONPATH` for the analysis environment and do not
import `vipose` from `extract/`. They are deliberately separate trees that share
only a file format: `datasets/probe-tracking-2025-10-23/schema/poses.schema.json`,
which the evaluation package validates on every load. A change in an extractor
therefore shows up as a load-time schema error rather than as a quietly
different number.
