# Changelog

Notable changes between tagged releases. Versions follow [semantic
versioning](https://semver.org/): the minor bump below reflects new public API,
added compatibly.

## v1.1.0

### Added

- **Block-bootstrap confidence intervals** for the residual statistics.
  `bootstrap_standard_error` resamples a statistic of an autocorrelated series
  by the moving-block bootstrap; `bootstrap_difference_standard_error`
  resamples the difference between two series, drawing one set of block start
  times for both where they cover a common window. The manuscript's residuals
  table now carries an interval on every median and 95th percentile, and its
  separation marks are decided by the interval for the difference rather than
  by whether two printed intervals overlap.
- **Correlation time and effective sample size** — `correlation_time`
  integrates the normalized autocorrelation over the initial run of positive
  lags; `effective_sample_size` divides the recording duration by it.
  `median_standard_error` gives the asymptotic alternative for a median.
- **Reference orientation uncertainty** — `reference_uncertainty` propagates a
  rigid-body fit residual to the orientation precision of the marker cluster
  that produced it, reporting the per-coordinate and per-marker noise
  separately alongside the uncertainty about each principal axis;
  `cluster_rms_radius` gives the lever arm it needs.
- `CITATION.cff`, and this changelog.

### Changed

- The raw-recording archive's DOI is corrected throughout from the reserved
  `10.5281/zenodo.22690903` to the published `10.5281/zenodo.22690904`
  (CC-BY-4.0). **Anyone working from the v1.0.0 tag has the reserved DOI,
  which does not resolve.**
- The archive size is stated consistently as ~17 GiB (18 GB) across the eight
  files of the four published recordings, computed from the byte counts in
  `raw.yaml`. Earlier documentation said ~16 GiB in two places and ~20 GiB in
  three.
- `project.urls.Repository` now points at the public repository.
- The README states up front that this package re-implements the paper's
  analysis and reproduces its published per-frame results, rather than leaving
  that in `docs/reproduction/`.

### Fixed

- `raw.yaml` carried `sha256: pending  # to be filled in when the archive is
  deposited` alongside a DOI marked published — the two contradicted each
  other. The archive is deposited and Zenodo publishes its own per-file
  checksums.

### Tests

189 pass, up from 132 at v1.0.0. The additions pin the new statistics to the
values the manuscript prints, and check the orientation propagation against a
Monte Carlo of the rigid-body fit it describes.

## v1.0.0

First public release: the evaluation library and CLI, the shipped example
dataset, the golden-master regression suite, and the acquisition-side code.
