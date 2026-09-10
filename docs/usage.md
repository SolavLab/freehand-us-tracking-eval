# Usage

This page covers the `vipose` command-line workflow: check a dataset,
evaluate it, and read the results. It uses the example dataset shipped with
this repository throughout — the extracted pose and marker-trajectory CSVs
for four recordings, not the multi-gigabyte raw sensor recordings themselves.
Nothing here is specific to that dataset, though: the same commands run
against a directory built from your own recordings (see
[data_formats.md § Authoring your own recording](data_formats.md#4-authoring-your-own-recording)).
For the underlying algorithms as a library you can call directly — useful if
you'd rather script this than go through a `dataset.yaml` — see
[algorithms.md](algorithms.md).

Install the package first (see [setup.md](setup.md)):

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Verify a dataset

```bash
vipose verify
```

Checks every file a dataset's manifest lists against its recorded sha256 and
row count, then — if a result store or golden master is given — summarises
its residuals per pipeline × recording. It fails on the first discrepancy
rather than proceeding with a partial dataset, so a missing or altered input
is a hard error rather than a silently dropped cell.

```bash
vipose verify --dataset path/to/your-dataset      # your own dataset directory
vipose verify --no-hashes                         # skip the sha256 pass (faster)
```

## Run an evaluation

```bash
vipose evaluate
```

For each recording in the dataset, and each pipeline within it: synchronize
against the reference, crop to the span they share, calibrate the constant
transform between them, and measure the per-frame residual (the four steps
`docs/algorithms.md` documents at the function level). Against the ~15 MiB
example dataset shipped here this takes a few minutes; against a larger
dataset of your own, scale accordingly.

Writes `results/<run-id>/`, where the run id is a UTC timestamp and the
dataset id:

```
results/<run-id>/
  run.yaml                                   versions, dataset, arguments
  recordings/<recording>/window.json         shared window and per-pipeline offsets
  cells/<pipeline>/<recording>/residuals.csv per-frame errors
  cells/<pipeline>/<recording>/calibration.json
  cells/<pipeline>/<recording>/cell.json     summary statistics
  <recording>.log                            full output of that recording's worker
```

Useful options:

```bash
vipose evaluate --dataset path/to/your-dataset
vipose evaluate --recordings pivot            # one recording
vipose evaluate --recordings pivot,freehand   # a subset
vipose evaluate --out /tmp/scratch-run        # somewhere other than results/
```

### Why one process per recording

`vipose evaluate` spawns a separate process per recording rather than
evaluating a whole dataset in one process. Stage 3 of the synchronization
performs on the order of a hundred calibrations per recording, and each one
is SVD-based; keeping that work in short-lived worker processes bounds peak
memory regardless of how many recordings a dataset has, and keeps one
recording's failure from aborting the rest of the run.

Each worker also runs with `DISPLAY` removed and BLAS thread counts pinned to
one, so a headless machine behaves the same as an interactive one and thread
count cannot perturb the last digits of an SVD-based fit.

## Read the results

```bash
cat results/<run-id>/cells/zed-sdk/pivot/cell.json
```

`cell.json` has the summary statistics for one pipeline × recording cell;
`residuals.csv` has the full per-frame series if you want to plot or
re-aggregate it yourself. `docs/data_formats.md` specifies every column and
unit.

## Regenerate summary tables and figures from a result store

```bash
vipose tables --store results/<run-id>
vipose figures --store results/<run-id>
```

Writes `tables/residuals.tex`, `tables/motion.tex`, `tables/reference_fit.tex`
(and their `.json` equivalents), plus two diagnostic figures, from any result
store — your own run or `tests/golden` (the default). These were built to
typeset this project's own manuscript, so the pipeline and condition
groupings (`report/tables.py`, `report/figures.py`) reflect that paper's
structure rather than being fully generic; they're included because the
grouping logic and the plots themselves (a synchronization angular-speed
overlay, and the stage-3 offset-search objective) are a reasonable starting
point if you want the same kind of summary over your own dataset.

```bash
vipose tables --store results/<run-id> --out tables
```

## Verify against a reference run

```bash
pytest tests/ -v
```

The comparison against `tests/golden` is elementwise and exact: there is no
random number generator anywhere in the pipeline, so a non-zero difference on
the shipped dataset is a real change rather than floating-point noise. This
repository's own provenance record of how that golden master was produced —
useful if you're auditing the shipped results rather than running your own —
is under [`docs/reproduction/`](reproduction/README.md).

## Regenerating pose trajectories from raw sensor recordings

Producing a `poses.csv` from raw camera/IMU data (rather than starting from
one, as everything above does) needs vendor SDKs and — for the recordings
shipped here — the raw archive catalogued in `raw.yaml`. See
[`extract/README.md`](../extract/README.md) and
[`docs/reproduction/setup-environments.md`](reproduction/setup-environments.md).
