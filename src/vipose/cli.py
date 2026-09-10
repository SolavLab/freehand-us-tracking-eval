"""Command line interface to the vipose evaluation.

    vipose verify     check a dataset matches its manifest, and summarise any
                      golden master alongside it
    vipose evaluate   run synchronization, calibration and residuals for every
                      recording in a dataset
    vipose tables     write summary tables from a result store
    vipose figures    write diagnostic figures from a result store

This is the only module permitted to configure matplotlib.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .metrics import summarize
from .recordings import Dataset, DatasetError
from .report.tables import (
    PIPELINE_ORDER,
    write_motion_table,
    write_reference_fit_table,
    write_residuals_table,
)
from .results import read_residuals

DEFAULT_DATASET = "datasets/probe-tracking-2025-10-23"
DEFAULT_GOLDEN = "tests/golden"


def _cmd_verify(args: argparse.Namespace) -> int:
    try:
        ds = Dataset.load(args.dataset, verify_hashes=not args.no_hashes)
    except DatasetError as exc:
        print(f"dataset FAILED\n{exc}", file=sys.stderr)
        return 1

    n_files = len(ds.manifest["files"])
    print(f"dataset {ds.id}")
    print(f"  {n_files} files verified" + ("" if not args.no_hashes else " (hashes skipped)"))
    print(f"  pipelines  : {', '.join(ds.pipelines)}")
    print(f"  recordings : {', '.join(ds.recordings())}")
    for rid, meta in ds.excluded().items():
        print(f"  excluded   : {rid} -- {meta['exclusion'].strip()}")

    golden = Path(args.golden)
    if not golden.is_dir():
        print(f"\nno golden master at {golden}; skipping residual checks")
        return 0

    print(f"\ngolden master at {golden}")
    problems = []
    for cell in ds.cells():
        path = golden / cell.pipeline / cell.recording / "residuals.csv"
        if not path.is_file():
            problems.append(f"{cell}: missing {path}")
            continue
        r = read_residuals(path)
        t, a = summarize(r.d_trans_mm), summarize(r.d_rot_deg)
        expected = float(ds.manifest["recordings"][cell.recording]["duration_s"])
        drift = abs(r.duration_s - expected)
        flag = (
            "" if drift < 0.6
            else f"  <-- span {r.duration_s:.1f}s vs manifest {expected:.1f}s"
        )
        if flag:
            problems.append(
                f"{cell}: evaluation window spans {r.duration_s:.1f}s, "
                f"manifest says {expected:.1f}s"
            )
        print(
            f"  {str(cell):26s} n={t.n:5d}  "
            f"d {t.median:5.2f} ({t.iqr:4.2f}) {t.p95:5.2f} mm   "
            f"phi {a.median:.3f} ({a.iqr:.3f}) {a.p95:.3f} deg{flag}"
        )
    if problems:
        print(f"\nFAILED: {len(problems)} problem(s)", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    print("\nOK")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluate import evaluate_all, run_id

    ds = Dataset.load(args.dataset, verify_hashes=not args.no_hashes)
    out = Path(args.out) if args.out else Path("results") / run_id(ds)
    recordings = args.recordings.split(",") if args.recordings else None
    print(f"evaluating {ds.id} -> {out}")
    evaluate_all(ds, out, recordings=recordings)
    print(f"\nresults in {out}")
    print(f"next: vipose tables --store {out}")
    return 0


def _cmd_tables(args: argparse.Namespace) -> int:
    ds = Dataset.load(args.dataset, verify_hashes=False)
    table = write_residuals_table(ds, args.store, args.out)
    motion = write_motion_table(ds, args.store, args.out)
    write_reference_fit_table(ds, args.store, args.out)
    print(f"wrote residuals.tex, motion.tex, reference_fit.tex and their .json "
          f"into {args.out}/")
    print(f"  motion: {len(motion['rows'])} recordings")
    print(
        f"  {len(table['rows'])} rows, "
        f"pipeline order: {', '.join(PIPELINE_ORDER)}"
    )
    return 0


def _cmd_figures(args: argparse.Namespace) -> int:
    from .report.figures import write_appendix_figures

    ds = Dataset.load(args.dataset, verify_hashes=False)
    print(f"generating figures from {args.store} (this evaluates the stage-3 objective)")
    written = write_appendix_figures(ds, args.store, args.out)
    for path in written:
        print(f"  {path} ({path.stat().st_size // 1024} kB)")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vipose", description=__doc__.splitlines()[0])
    p.add_argument(
        "--dataset", default=DEFAULT_DATASET,
        help=f"dataset directory (default: {DEFAULT_DATASET})",
    )
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("verify", help="check the dataset and golden master")
    v.add_argument("--golden", default=DEFAULT_GOLDEN)
    v.add_argument(
        "--no-hashes", action="store_true", help="skip sha256 verification (faster)"
    )
    v.set_defaults(func=_cmd_verify)

    e = sub.add_parser("evaluate", help="run the full evaluation")
    e.add_argument("--out", default=None, help="output directory (default: results/<run-id>)")
    e.add_argument(
        "--recordings", default=None,
        help="comma-separated recording ids (default: all published)",
    )
    e.add_argument("--no-hashes", action="store_true")
    e.set_defaults(func=_cmd_evaluate)

    t = sub.add_parser("tables", help="write summary tables from a result store")
    t.add_argument("--store", default=DEFAULT_GOLDEN, help="result store or golden master")
    t.add_argument("--out", default="tables", help="output directory")
    t.set_defaults(func=_cmd_tables)

    f = sub.add_parser("figures", help="write diagnostic figures from a result store")
    f.add_argument("--store", default=DEFAULT_GOLDEN, help="result store or golden master")
    f.add_argument("--out", default="figures", help="output directory")
    f.set_defaults(func=_cmd_figures)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
