# Visual–inertial pose tracking for freehand 3D-ultrasound probe localization

Code and reference data for:

> Z. Oddes and D. Solav, *Accuracy of inside-out visual–inertial tracking
> pipelines for freehand 3D ultrasound probe localization*.

Three complete pose-tracking pipelines were compared against an eleven-camera
Vicon motion-capture reference, under identical physical conditions:

| Pipeline | Camera | Platform | Localization |
|---|---|---|---|
| `zed-sdk` | ZED X Mini | Jetson Orin NX | ZED SDK 5.3.0 positional tracking |
| `zed-cuvslam` | ZED X Mini | workstation | cuVSLAM 3.2 (Isaac ROS) |
| `rs-cuvslam` | RealSense D455 | workstation | cuVSLAM 3.2 (Isaac ROS) |

Both cameras and a five-marker infrared cluster were mounted on one rigid frame,
so all three pipelines saw the same motion, scene and lighting. Four recordings
cover three motion conditions, with the pivot condition acquired twice to
measure trial-to-trial variability.

## Reproduce the paper

Every number, table and figure in the manuscript is reproducible from this
repository alone — the raw recordings are not needed.

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

vipose verify            # 16 input files checked against their sha256
pytest tests/ -v         # golden master and manuscript checks
vipose evaluate          # synchronization, calibration, residuals  (~30 min)
vipose tables            # -> tables/residuals.tex
```

`vipose tables` regenerates the manuscript's tables and `vipose figures` its
appendix figures. Table 2 and Table 3 come out character-for-character identical
to the paper. `docs/paper_map.md` maps every published number to the command
that produces it, and is explicit about the three that do not yet regenerate.

## What is here

```
datasets/probe-tracking-2025-10-23/   16 input files, 15 MiB
  reference/<recording>/mocap.csv     Vicon marker trajectories, 240 Hz
  tracking/<pipeline>/<recording>/    per-pipeline pose CSVs
  dataset.yaml                        the manifest: ids, hashes, provenance
  schema/poses.schema.json            the pose-CSV contract
  raw.yaml                            catalogue of the ~20 GiB raw archive
src/vipose/                           the evaluation package
extract/                              acquisition-side code (see below)
environments/isaac_ros/               pinned upstream + patches for cuVSLAM
tests/golden/                         the published per-frame residuals
tests/tools/                          the provenance harness (see below)
docs/                                 setup, usage, data formats, licensing
```

Recordings are addressed by condition — `pivot`, `mixed`, `freehand`,
`pivot-repeat` — and pipelines by name. `dataset.yaml` records the `test1`..`test5`
identifiers the original analysis used, so results published before this release
remain traceable. It also records the fifth recording that was acquired and
excluded, with the reason.

## How the numbers were checked

`tests/golden/` holds the per-frame residual series for all twelve
pipeline × recording cells, as published. It was not taken on trust: the original
code was re-run and compared against the artifacts that produced the paper at
four tolerance tiers — frame counts and columns exact, per-frame residuals
elementwise, fitted transforms to 1e-9, temporal offsets to 1e-6 ms.

All twelve cells reproduced with a **maximum per-frame difference of exactly
zero**, and all seventy-two published statistics round-trip. `tests/tools/` holds
that harness, preserved as it was run, so the claim is auditable rather than
asserted.

`pytest` re-checks a fresh evaluation against that reference elementwise, which
is stronger than comparing medians: a median can survive substantial
rearrangement of the series beneath it.

## Reproduction ladder

| Rung | Regenerates | Needs |
|---|---|---|
| **R1** | tables and figures | this repository, Python 3.10 |
| **R2** | the full evaluation | as above, ~30 min |
| **R3** | pose estimation from raw recordings | vendor SDKs and the ~20 GiB archive |

Most readers need only R1 and R2. R3 exists so the acquisition path is
inspectable and re-runnable, not so it can be replayed to identical output:
cuVSLAM and the ZED SDK both run on the GPU and are not bit-deterministic. Pose
estimation reproduces to within the trial-to-trial variability the paper itself
reports (0.05–0.94 mm, 0.019–0.037°).

`docs/setup.md` covers the four environments R3 requires, and why they cannot be
combined.

## Implementation

`src/vipose/` is a complete reimplementation — about 2,200 lines across twenty
modules, replacing roughly 4,700 lines of the original four. Each module was
ported against the code it replaced and checked elementwise before the original
was removed, and the resulting pipeline reproduces the published per-frame
residuals to 1.4e-08 mm and 2.5e-09 degrees. `docs/porting-notes.md` records
what changed and what was deliberately preserved.

```
recordings.py    the dataset manifest; Dataset and Cell
io/mocap.py      Vicon Nexus export -> MocapData
io/tracks.py     pose CSV -> Track, schema-validated
geometry/        SE(3) helpers, and the Kabsch marker-cluster fit
kinematics.py    rotational increments and geodesic angular speed
sync/            stages 1 and 2, reference resampling
calibration.py   robot-world/hand-eye (Shah), marker-intrinsic frame
metrics.py       residuals and summary statistics -- pure, no plotting
pipeline.py      orchestration, including stage 3 and the fixed-point loop
report/          tables
evaluate.py      result-store writer and the per-recording driver
cli.py           the `vipose` command
```

Nothing outside `report/` imports pyplot. In the original, the residual values
were returned *by* the plotting functions, so results could not be computed
without drawing figures — which is why a matplotlib error could destroy a
twelve-cell run and why the figures leaked until the process ran out of memory.

## Licence

MIT, see [`LICENSE`](LICENSE). `docs/licensing.md` explains why no
third-party source is redistributed, and what still needs attention before publication.

## Citation

`CITATION.cff` — pending a DOI for the raw archive.
