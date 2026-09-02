#!/usr/bin/env python3
"""Convert a cuVSLAM `poses.csv` into the common pose schema.

ENV: any Python 3 with numpy and scipy. Runs inside the Isaac ROS container in
     the normal flow, but has no ROS dependency.

The cuVSLAM pipelines report poses as metres and quaternions; pipeline I and the
evaluation use millimetres and rotation vectors. This is the adapter, and it is
the last step of stages II and III.

Three conversions, each of which matters:

* **metres to millimetres**, matching the Vicon reference.
* **quaternion to rotation vector** (axis-angle). The common schema stores
  orientations as rotation vectors; reading those columns as Euler angles
  instead introduces an orientation-dependent gain error of up to 20 percent on
  every rotational metric, and this project has actually suffered that.
* **seconds to integer milliseconds**, matching the acquisition clocks.

`Spatial_Memory` is set to the literal string `cuVSLAM`. That is a provenance
marker, not a state: it is how a reader tells a cuVSLAM trajectory from a ZED
SDK one from the data alone, and the evaluation relies on nothing else to do so.

This file existed as two byte-identical copies in two separate repositories,
each reached by a different driver script. There is one now.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

log = logging.getLogger("tracks_to_common_csv")

INPUT_COLUMNS = ["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"]
OUTPUT_COLUMNS = [
    "Frame", "Timestamp",
    "Translation_X", "Translation_Y", "Translation_Z",
    "Rotation_X", "Rotation_Y", "Rotation_Z",
    "Pose_Confidence", "Tracking_State",
    "Spatial_Memory", "Odometry", "Tracking_Fusion",
]
M_TO_MM = 1000.0
PROVENANCE_MARKER = "cuVSLAM"


def convert(rows: list[dict]) -> list[list]:
    """Convert parsed `poses.csv` rows to the common schema."""
    if not rows:
        raise ValueError("no poses to convert")

    quaternions = np.array(
        [[float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])] for r in rows]
    )
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms < 1e-9):
        raise ValueError(f"{int((norms < 1e-9).sum())} zero-length quaternions in the input")
    if np.max(np.abs(norms - 1.0)) > 1e-3:
        # scipy renormalises silently; a large deviation means the upstream
        # export is wrong and worth knowing about rather than absorbing.
        log.warning(
            "quaternions deviate from unit norm by up to %.2e; scipy will renormalise",
            float(np.max(np.abs(norms - 1.0))),
        )
    rotvecs = Rotation.from_quat(quaternions).as_rotvec()

    out = []
    for index, (row, rotvec) in enumerate(zip(rows, rotvecs, strict=True)):
        out.append([
            index,
            int(round(float(row["timestamp"]) * 1000.0)),
            float(row["x"]) * M_TO_MM,
            float(row["y"]) * M_TO_MM,
            float(row["z"]) * M_TO_MM,
            rotvec[0], rotvec[1], rotvec[2],
            "",                      # no confidence equivalent in cuVSLAM
            "",                      # no tracking-state equivalent
            PROVENANCE_MARKER,
            "",
            "",
        ])
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("input", type=Path, help="poses.csv from export_poses.py")
    p.add_argument("-o", "--output", type=Path, required=True, help="output pose CSV")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not args.input.is_file():
        p.error(f"no such file: {args.input}")

    with open(args.input, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in INPUT_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            p.error(f"{args.input}: missing column(s) {missing}; found {reader.fieldnames}")
        rows = list(reader)

    converted = convert(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(OUTPUT_COLUMNS)
        writer.writerows(converted)

    span = (converted[-1][1] - converted[0][1]) / 1000.0
    log.info("wrote %d poses spanning %.1f s to %s", len(converted), span, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
