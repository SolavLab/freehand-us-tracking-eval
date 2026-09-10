# Setup

## Install `vipose`

Python **3.10**, pinned by `requires-python` in `pyproject.toml`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Dependency versions are pinned exactly in `pyproject.toml` (numpy 2.2.6,
scipy 1.15.3, pandas 2.3.3, matplotlib 3.10.8, seaborn 0.13.2,
opencv-python 4.13.0.92, PyYAML 6.0.3) because the numerics this package
produces are sensitive to them — an SVD-based fit or a solver can shift in
its last digits between library versions. Add `[dev]` for `pytest` and `ruff`
if you're running the test suite or contributing:

```bash
pip install -e ".[dev]"
```

`opencv-python` is the one pin most worth being aware of: the hand–eye
calibration calls `cv2.calibrateRobotWorldHandEye`, whose internal SVD sign
conventions can differ between OpenCV builds. If a calibration looks
qualitatively right but the transforms don't match another run bit-for-bit,
an OpenCV version mismatch is the first thing to check.

Check the install:

```bash
vipose verify            # against the shipped example dataset
pytest tests/ -v
```

## Everything else is optional

The install above is all you need to use `vipose` — as a library
(`docs/algorithms.md`) or through the CLI (`docs/usage.md`) — against pose
and mocap CSVs you already have, including the ones shipped in
`datasets/probe-tracking-2025-10-23/`.

Three more environments exist only to **regenerate those pose CSVs from raw
sensor recordings** (a ZED SVO2 file, a RealSense `.bag`), which most users of
this package will never need to do. They're mutually incompatible with each
other and with the environment above, they need vendor SDKs this project
cannot distribute, and — to regenerate the CSVs shipped here specifically —
the ~20 GiB raw archive behind them, which is too large for this repository
(see `datasets/probe-tracking-2025-10-23/raw.yaml`). If that's what you're
after, see
[`docs/reproduction/setup-environments.md`](reproduction/setup-environments.md).

## Which environment runs which script

| Path | Environment |
|---|---|
| `src/vipose/**`, the `vipose` CLI, `tests/**` | the install above |
| `extract/zed_sdk/**` | `zed-sdk` (see reproduction/setup-environments.md) |
| `extract/convert/**` | `ros-convert` |
| `extract/cuvslam/**` | `isaac-ros`, inside a container |

Every executable in this repository names its interpreter in a header
comment. Do not put `extract/` on `PYTHONPATH` for the `vipose` environment
and do not import `vipose` from `extract/` — they are deliberately separate
trees that share only a file format,
`datasets/probe-tracking-2025-10-23/schema/poses.schema.json`, which
`vipose` validates on every load. A change on the extraction side therefore
surfaces as a load-time schema error rather than as a quietly different
number.
