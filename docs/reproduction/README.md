# Reproduction record

**You don't need anything in this folder to use `vipose`.** It documents how
the manuscript

> Z. Oddes and D. Solav, *Accuracy of inside-out visual–inertial tracking
> pipelines for freehand 3D ultrasound probe localization*

was produced and checked against this codebase, for reviewers or anyone
auditing the published numbers rather than running their own evaluation. If
you arrived here wanting to evaluate your own recording, start at
[`README.md`](../../README.md) and [`docs/usage.md`](../usage.md) instead.

- **[`paper_map.md`](paper_map.md)** — every table, figure and quoted number
  in the manuscript, mapped to the command that reproduces it, with the two
  that don't yet regenerate stated explicitly.
- **[`porting-notes.md`](porting-notes.md)** — this package is a from-scratch
  reimplementation of the analysis code that produced the manuscript. This
  records what was verified, what was deliberately preserved even where it
  looks questionable, and what was changed and why.
- **[`setup-environments.md`](setup-environments.md)** — the three
  environments needed only to regenerate raw sensor recordings into pose
  CSVs, and the reproduction ladder (R1/R2/R3) they belong to.

## How the numbers were checked

`tests/golden/` holds the per-frame residual series for all twelve
pipeline × recording cells, as published. It was not taken on trust: the
original analysis code was re-run and compared against the artifacts that
produced the paper at four tolerance tiers — frame counts and columns exact,
per-frame residuals elementwise, fitted transforms to 1e-9, temporal offsets
to 1e-6 ms.

All twelve cells reproduced with a **maximum per-frame difference of exactly
zero**, and all seventy-two published statistics round-trip. `tests/tools/`
holds that harness, preserved as it was run, so the claim is auditable rather
than asserted. `pytest tests/` re-checks a fresh evaluation against that
reference elementwise, which is stronger than comparing medians: a median can
survive substantial rearrangement of the series beneath it.
