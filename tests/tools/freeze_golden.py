"""Freeze the Phase-0 baseline into the release as tests/golden/.

Run only after gate.py reports PASS. Converts the original tool's output for the
12 published cells into the release's layout and records everything needed to
detect a regression later:

  tests/golden/<pipeline>/<recording>/residuals.csv    frame, time_ms, d_trans_mm, d_rot_deg
  tests/golden/<pipeline>/<recording>/calibration.json transforms, marker frame, lag, stages
  tests/golden/offsets.json                            the 12 stage-1/2/3 lags, pinnable
  tests/golden/manifest.json                           hashes + interpreter/library versions

`offsets.json` exists because the stage-2 objective is multi-modal (five
near-equal troughs spanning ~12 ms). Pinning the lag lets the calibration be
regression-tested independently of the optimiser, so a refactor that perturbs
floating-point ordering is diagnosed rather than merely detected.
"""
import sys, os, re, json, glob, hashlib, platform
import numpy as np
import pandas as pd

PIPE = {"zed-sdk": (1, "pipeline1_zed_sdk"),
        "zed-cuvslam": (2, "pipeline2_isaac_svo2"),
        "rs-cuvslam": (3, "pipeline3_isaac_realsense")}
REC = {"pivot": "test2", "mixed": "test3", "freehand": "test5", "pivot-repeat": "test1"}

NUM = re.compile(r"-?\d+\.\d+e[+-]\d+|-?\d+\.\d+|-?\d+")


def parse_calib_txt(path):
    """Pull the named 4x4 matrices, axes and centroid out of the original's
    human-readable dump. The dump prints numpy repr, which nothing can read
    back; this is the last time that format is parsed."""
    text = open(path).read()
    out = {}
    for key, label in (("T_camera_to_marker", "T_zed2rb"),
                       ("T_slamworld_to_vicon", "T_zed02vicon"),
                       ("T_vicon_to_slamworld", "T_vicon2zed0"),
                       ("T_camera_to_marker_intrinsic", "T_zed2marker_intrinsic")):
        m = re.search(re.escape(label) + r"[^\[]*(\[\[.*?\]\])", text, re.S)
        if m:
            v = [float(x) for x in NUM.findall(m.group(1))]
            if len(v) == 16:
                out[key] = np.array(v).reshape(4, 4).tolist()
    for key, label in (("marker_centroid", "Marker centroid"),
                       ("marker_x_axis", "Marker-intrinsic X-axis"),
                       ("marker_y_axis", "Marker-intrinsic Y-axis"),
                       ("marker_z_axis", "Marker-intrinsic Z-axis")):
        m = re.search(re.escape(label) + r"[^\[]*\[([^\]]*)\]", text, re.S)
        if m:
            out[key] = [float(x) for x in NUM.findall(m.group(1))]
    m = re.search(r"Optimized temporal lag:\s*(-?[\d.]+)", text)
    if m:
        out["temporal_offset_ms"] = float(m.group(1))
    return out


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    src = sys.argv[1].rstrip("/")          # a comparison/ dir
    dst = sys.argv[2].rstrip("/")          # tests/golden
    os.makedirs(dst, exist_ok=True)
    offsets, manifest_files = {}, {}

    for pid, (pnum, pdir) in PIPE.items():
        for rid, t in REC.items():
            base = f"{src}/{pdir}/{t}/pipeline{pnum}_{t}_calibration_results_unified"
            d = pd.read_csv(base + "_distances.csv")
            a = pd.read_csv(base + "_angles.csv")
            dcol = [c for c in d.columns if c != "Frame"][0]
            acol = [c for c in a.columns if c != "Frame"][0]
            assert d["Frame"].equals(a["Frame"]), f"{pid}/{rid}: frame columns differ"

            meta = json.load(open(f"{src}/{pdir}/{t}/pipeline{pnum}_{t}_metadata_unified.json"))
            calib = parse_calib_txt(base + ".txt")
            calib["offset_refinement_stages"] = meta["offset_refinement_stages"]
            calib["temporal_offset_ms"] = meta["temporal_offset_ms"]
            calib["vicon_window_ms"] = meta["vicon_window"]
            calib["local_window_ms"] = meta["local_window"]
            calib["provenance"] = {
                "source": "2026-08-18 published run, reproduced by Phase-0 baseline",
                "legacy_pipeline_dir": pdir, "legacy_test": t,
            }

            # The legacy residual CSVs carry only a positional row counter, no
            # time. Recover it from the cropped trajectory written alongside
            # them, which aligns row-for-row (verified: equal lengths, the
            # residual Frame column is exactly range(n), timestamps monotonic).
            # `source_frame` is the pipeline's own frame index, which is more
            # informative than the counter, and `time_ms` is the authoritative
            # acquisition clock. Without these, a residual series cannot be
            # placed in time or joined back to the trajectory.
            traj = glob.glob(f"{src}/{pdir}/{t}/*_unified.csv")
            assert len(traj) == 1, f"{pid}/{rid}: expected one cropped trajectory, got {traj}"
            tr = pd.read_csv(traj[0])
            assert len(tr) == len(d), f"{pid}/{rid}: trajectory {len(tr)} vs residuals {len(d)}"
            assert list(d["Frame"]) == list(range(len(d))), f"{pid}/{rid}: Frame not positional"

            out = f"{dst}/{pid}/{rid}"
            os.makedirs(out, exist_ok=True)
            pd.DataFrame({
                "frame": d["Frame"].astype(int),
                "source_frame": tr["Frame"].astype(int).values,
                "time_ms": tr["Timestamp"].astype("int64").values,
                "d_trans_mm": d[dcol].astype(float).values,
                "d_rot_deg": a[acol].astype(float).values,
            }).to_csv(f"{out}/residuals.csv", index=False, lineterminator="\n")
            with open(f"{out}/calibration.json", "w") as f:
                json.dump(calib, f, indent=2, sort_keys=True)
                f.write("\n")

            offsets.setdefault(pid, {})[rid] = meta["temporal_offset_ms"]
            for rel in (f"{pid}/{rid}/residuals.csv", f"{pid}/{rid}/calibration.json"):
                manifest_files[rel] = sha256(f"{dst}/{rel}")
            print(f"  {pid:12s} {rid:13s} n={len(d):5d}  "
                  f"med d={np.median(d[dcol]):6.3f} mm  med phi={np.median(a[acol]):.4f} deg")

    with open(f"{dst}/offsets.json", "w") as f:
        json.dump(offsets, f, indent=2, sort_keys=True); f.write("\n")

    import scipy, matplotlib, cv2
    with open(f"{dst}/manifest.json", "w") as f:
        json.dump({
            "description": "Golden master for the 12 published pipeline x recording cells. "
                           "Reproduced from the original code in Phase 0 and verified "
                           "elementwise against the 2026-08-18 published artifacts and "
                           "against Table 3 of the manuscript.",
            "environment": {
                "python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "pandas": pd.__version__,
                "matplotlib": matplotlib.__version__, "opencv": cv2.__version__,
            },
            "files": manifest_files,
        }, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"\nfroze {len(manifest_files)//2} cells -> {dst}")


if __name__ == "__main__":
    main()
