"""Phase-0 gate: does today's code reproduce the 2026-08-18 published artifacts?

Compares a fresh run's `comparison/` tree against the one on disk in ZED_Project,
cell by cell, at the tolerance tiers defined in the plan:

  tier 0  frame counts, column sets, row order      exact
  tier 1  per-frame residuals                       atol=0 first, then measured floor
  tier 2  transforms from the .txt                  atol=1e-9
  tier 3  offsets / windows from metadata JSON      atol=1e-6 ms
  tier 4  median / IQR / p95                        must round-trip to Table 3

Usage:  python3 gate.py <fresh_comparison_dir> [<reference_comparison_dir>]
"""
import sys, os, json, re
import numpy as np
import pandas as pd

REF_DEFAULT = os.environ.get("VIPOSE_LEGACY_COMPARISON_DIR", "")
PIPE = {1: "pipeline1_zed_sdk", 2: "pipeline2_isaac_svo2", 3: "pipeline3_isaac_realsense"}
COND = {"test2": "Pivot", "test3": "Mixed", "test5": "Freehand", "test1": "Pivot(repeat)"}
TESTS = ["test2", "test3", "test5", "test1"]

# paper Table 3: (n, d_median, d_iqr, d_p95, a_median, a_iqr, a_p95)
TABLE3 = {
    (1, "test2"): (3304, 5.26, 3.40, 9.13, 0.193, 0.142, 0.390),
    (1, "test3"): (3583, 4.08, 3.14, 9.77, 0.144, 0.109, 0.318),
    (1, "test5"): (2812, 4.77, 3.88, 9.55, 0.148, 0.108, 0.304),
    (1, "test1"): (2243, 4.32, 2.29, 9.01, 0.156, 0.106, 0.326),
    (2, "test2"): (3253, 2.63, 1.76, 5.13, 0.100, 0.076, 0.213),
    (2, "test3"): (3520, 2.51, 1.95, 5.74, 0.115, 0.089, 0.237),
    (2, "test5"): (2757, 4.09, 2.85, 7.38, 0.117, 0.078, 0.223),
    (2, "test1"): (2217, 3.11, 2.08, 5.95, 0.121, 0.074, 0.224),
    (3, "test2"): (1652, 3.08, 2.44, 7.78, 0.149, 0.104, 0.307),
    (3, "test3"): (1792, 4.10, 2.97, 8.44, 0.156, 0.131, 0.309),
    (3, "test5"): (1406, 3.81, 2.86, 7.66, 0.190, 0.111, 0.402),
    (3, "test1"): (1122, 3.03, 3.11, 9.39, 0.130, 0.115, 0.412),
}

MAT = re.compile(r"-?\d+\.\d+e[+-]\d+|-?\d+\.\d+|-?\d+")
failures, notes = [], []


def fail(cell, tier, msg):
    failures.append((cell, tier, msg))
    print(f"    FAIL [tier {tier}] {msg}")


def load_resid(base, p, t):
    d = f"{base}/{PIPE[p]}/{t}/pipeline{p}_{t}_calibration_results_unified_distances.csv"
    a = f"{base}/{PIPE[p]}/{t}/pipeline{p}_{t}_calibration_results_unified_angles.csv"
    if not (os.path.exists(d) and os.path.exists(a)):
        return None, None
    return pd.read_csv(d), pd.read_csv(a)


def numbers(path):
    """Every float in a calibration .txt, in order — order is stable, so this
    compares the transforms without parsing numpy repr."""
    if not os.path.exists(path):
        return None
    return np.array([float(x) for x in MAT.findall(open(path).read())])


def stats(v):
    return (len(v), float(np.median(v)),
            float(np.percentile(v, 75) - np.percentile(v, 25)),
            float(np.percentile(v, 95)))


def main():
    new = sys.argv[1].rstrip("/")
    ref = (sys.argv[2].rstrip("/") if len(sys.argv) > 2 else REF_DEFAULT)
    print(f"fresh : {new}\nref   : {ref}\n")

    max_resid_delta = 0.0
    for p in (1, 2, 3):
        for t in TESTS:
            cell = f"p{p}/{COND[t]}"
            print(f"{cell}")
            dn, an = load_resid(new, p, t)
            dr, ar = load_resid(ref, p, t)
            if dn is None:
                fail(cell, 0, "fresh outputs missing")
                continue
            if dr is None:
                fail(cell, 0, "reference outputs missing")
                continue

            # tier 0
            if list(dn.columns) != list(dr.columns):
                fail(cell, 0, f"distance columns {list(dn.columns)} vs {list(dr.columns)}")
            if list(an.columns) != list(ar.columns):
                fail(cell, 0, f"angle columns {list(an.columns)} vs {list(ar.columns)}")
            if len(dn) != len(dr) or len(an) != len(ar):
                fail(cell, 0, f"row counts d {len(dn)}/{len(dr)} a {len(an)}/{len(ar)}")
                continue
            if not dn["Frame"].equals(dr["Frame"]) or not an["Frame"].equals(ar["Frame"]):
                fail(cell, 0, "Frame column differs")

            # tier 1
            dcol = [c for c in dn.columns if c != "Frame"][0]
            acol = [c for c in an.columns if c != "Frame"][0]
            for lbl, x, y in (("d", dn[dcol].values, dr[dcol].values),
                              ("phi", an[acol].values, ar[acol].values)):
                delta = float(np.max(np.abs(x - y))) if len(x) else 0.0
                max_resid_delta = max(max_resid_delta, delta)
                if delta != 0.0:
                    rel = delta / max(1e-30, float(np.max(np.abs(y))))
                    fail(cell, 1, f"{lbl} max|Δ| = {delta:.3e} (rel {rel:.2e})")

            # tier 2
            nn = numbers(f"{new}/{PIPE[p]}/{t}/pipeline{p}_{t}_calibration_results_unified.txt")
            nr = numbers(f"{ref}/{PIPE[p]}/{t}/pipeline{p}_{t}_calibration_results_unified.txt")
            if nn is None or nr is None:
                fail(cell, 2, "calibration .txt missing")
            elif len(nn) != len(nr):
                fail(cell, 2, f"calibration .txt has {len(nn)} numbers vs {len(nr)}")
            else:
                dmax = float(np.max(np.abs(nn - nr)))
                if dmax > 1e-9:
                    fail(cell, 2, f"transforms max|Δ| = {dmax:.3e}")

            # tier 3
            jn_p = f"{new}/{PIPE[p]}/{t}/pipeline{p}_{t}_metadata_unified.json"
            jr_p = f"{ref}/{PIPE[p]}/{t}/pipeline{p}_{t}_metadata_unified.json"
            if os.path.exists(jn_p) and os.path.exists(jr_p):
                jn, jr = json.load(open(jn_p)), json.load(open(jr_p))
                for k in ("temporal_offset_ms",):
                    if abs(jn[k] - jr[k]) > 1e-6:
                        fail(cell, 3, f"{k} {jn[k]!r} vs {jr[k]!r}")
                for k, v in jr["offset_refinement_stages"].items():
                    w = jn["offset_refinement_stages"].get(k)
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        if w is None or abs(w - v) > 1e-6:
                            fail(cell, 3, f"stage {k} {w!r} vs {v!r}")
                    elif w != v:
                        fail(cell, 3, f"stage {k} {w!r} vs {v!r}")
                for k in ("vicon_window", "local_window"):
                    if np.max(np.abs(np.array(jn[k]) - np.array(jr[k]))) > 1e-6:
                        fail(cell, 3, f"{k} {jn[k]} vs {jr[k]}")
            else:
                fail(cell, 3, "metadata json missing")

            # tier 4 — against the manuscript, not the reference files
            n, dm, di, dp, am, ai, ap = TABLE3[(p, t)]
            sn, smd, sid, spd = stats(dn[dcol].values)
            _, sma, sia, spa = stats(an[acol].values)
            if sn != n:
                fail(cell, 4, f"frame count {sn} vs Table 3 {n}")
            for lbl, got, exp, dp_ in (("d_med", smd, dm, 2), ("d_iqr", sid, di, 2),
                                       ("d_p95", spd, dp, 2), ("a_med", sma, am, 3),
                                       ("a_iqr", sia, ai, 3), ("a_p95", spa, ap, 3)):
                if round(got, dp_) != exp:
                    fail(cell, 4, f"{lbl} {got:.5f} rounds to {round(got, dp_)}, Table 3 says {exp}")
            print(f"    ok  n={sn}  d {smd:.2f} ({sid:.2f}) {spd:.2f}   "
                  f"phi {sma:.3f} ({sia:.3f}) {spa:.3f}")

    print("\n" + "=" * 72)
    print(f"max per-frame residual |Δ| across all 24 series: {max_resid_delta:.3e}")
    if not failures:
        print("GATE PASSED — today's code reproduces the published artifacts exactly,")
        print("and all 12 cells round-trip to Table 3.")
        return 0
    print(f"GATE FAILED — {len(failures)} problem(s):")
    by_tier = {}
    for cell, tier, msg in failures:
        by_tier.setdefault(tier, []).append((cell, msg))
    for tier in sorted(by_tier):
        print(f"  tier {tier}: {len(by_tier[tier])}")
        for cell, msg in by_tier[tier][:12]:
            print(f"     {cell:18s} {msg}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
