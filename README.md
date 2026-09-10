# vipose — visual–inertial pose tracking evaluation

`vipose` compares a camera's estimated trajectory against a motion-capture
reference: it synchronizes the two clocks, calibrates the constant transform
between the camera and the reference rig, and reports the per-frame
translational and rotational residual. The algorithms are plain functions
over numpy arrays — see [`docs/algorithms.md`](docs/algorithms.md) if you
want to call them directly against your own trajectories — with a CLI on top
for running the whole thing over a directory of recordings.

It was built to compare three real-time pose-tracking pipelines against an
eleven-camera Vicon reference, for the paper this repository accompanies:

> Z. Oddes and D. Solav, *Accuracy of inside-out visual–inertial tracking
> pipelines for freehand 3D ultrasound probe localization*.

| Pipeline | Camera | Platform | Localization |
|---|---|---|---|
| `zed-sdk` | ZED X Mini | Jetson Orin NX | ZED SDK 5.3.0 positional tracking |
| `zed-cuvslam` | ZED X Mini | workstation | cuVSLAM 3.2 (Isaac ROS) |
| `rs-cuvslam` | RealSense D455 | workstation | cuVSLAM 3.2 (Isaac ROS) |

Both cameras and a five-marker infrared cluster were mounted on one rigid
frame, so all three pipelines saw the same motion, scene and lighting. The raw
recordings themselves (SVO2 and `.bag` files, several GB per recording) are
not in this repository, but the extracted pose and marker-trajectory CSVs for
four recordings ship here (~15 MiB total) as example data — enough to run the
whole workflow without acquiring or extracting anything yourself.

## Quickstart

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -e .

vipose verify             # check the example dataset against its manifest
vipose evaluate            # synchronize, calibrate, measure -- a few minutes
vipose tables               # summary tables from the run just written
cat results/*/cells/zed-sdk/pivot/cell.json
```

That runs the full workflow on the shipped example recordings. Point
`--dataset` at a directory built from your own recordings to run it on those
instead — see [`docs/usage.md`](docs/usage.md) for the CLI, and
[`docs/data_formats.md`](docs/data_formats.md#4-authoring-your-own-recording)
for what a dataset directory needs to contain.

## What is here

```
datasets/probe-tracking-2025-10-23/   the shipped example: 16 input files, 15 MiB
  reference/<recording>/mocap.csv     Vicon marker trajectories, 240 Hz
  tracking/<pipeline>/<recording>/    per-pipeline pose CSVs
  dataset.yaml                        the manifest: ids, hashes, provenance
  schema/poses.schema.json            the pose-CSV contract
  raw.yaml                            catalogue of the ~20 GiB raw archive (too large for GitHub; planned for a Zenodo deposit)
src/vipose/                           the library and CLI
extract/                              acquisition-side code: raw recordings -> pose CSVs
environments/isaac_ros/               pinned upstream + patches for cuVSLAM
tests/golden/                         the published per-frame residuals, for regression checks
tests/tools/                          the provenance harness (see docs/reproduction/)
docs/                                 algorithms, usage, data formats, setup, licensing
```

## Documentation

- **[`docs/algorithms.md`](docs/algorithms.md)** — the synchronization,
  calibration and metrics functions as a library, with signatures and a
  worked example against a plain trajectory and mocap file.
- **[`docs/usage.md`](docs/usage.md)** — the `vipose` CLI: verify a dataset,
  run an evaluation, read the results.
- **[`docs/data_formats.md`](docs/data_formats.md)** — every file format
  involved, and what your own recording needs to look like.
- **[`docs/setup.md`](docs/setup.md)** — installing `vipose`.
- **[`docs/licensing.md`](docs/licensing.md)** — why no third-party source is
  redistributed here, and what that means for reuse.
- **[`docs/reproduction/`](docs/reproduction/README.md)** — how this
  repository's own numbers were produced and checked against the manuscript.
  Not needed to use `vipose`; useful if you're reviewing the paper or
  auditing the shipped results.

## Implementation

```
recordings.py    the dataset manifest; Dataset and Cell
io/mocap.py      Vicon Nexus export -> MocapData
io/tracks.py     pose CSV -> Track, schema-validated
geometry/        SE(3) helpers, and the Kabsch marker-cluster fit
kinematics.py    rotational increments and geodesic angular speed
sync/            temporal synchronization stages, reference resampling
calibration.py   robot-world/hand-eye (Shah), marker-intrinsic frame
conditioning.py  well-posedness of the hand-eye solve
metrics.py       residuals and summary statistics -- pure, no plotting
pipeline.py      the per-recording driver: synchronize, crop, calibrate, measure
report/          tables and figures, built for this manuscript's structure
evaluate.py      result-store writer and the per-recording driver
cli.py           the `vipose` command
```

Nothing outside `report/` imports pyplot, so results never depend on drawing
a figure.

## Licence

MIT, see [`LICENSE`](LICENSE). [`docs/licensing.md`](docs/licensing.md)
explains why no third-party source is redistributed, and what that means if
you build on this.

## Citation

See the manuscript citation above. Machine-readable citation metadata
(`CITATION.cff`) will be added once the raw-recording archive has a DOI.
