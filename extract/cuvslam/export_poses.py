#!/usr/bin/env python3
"""Parse a cuVSLAM `GetAllPoses` response into a CSV trajectory.

ENV: inside the Isaac ROS container (`python3`, ROS 2 Humble). Needs no ROS
     imports itself -- see below.

When playback of a recording finishes, the visual-SLAM node is asked for its
optimised trajectory via the `GetAllPoses` service, and the response is captured
with `ros2 service call`, which prints Python `repr` of the message. This turns
that text into `poses.csv` with columns
`timestamp,x,y,z,qx,qy,qz,qw,frame_id`.

Parsing `repr` is unpleasant, and it is worth saying why it is done this way:
`ros2 service call` is the only route to that service that works from a shell
inside the container without building a client node, and the response is large
(up to 20000 poses). The alternative -- a small rclpy client -- would be
cleaner, but the published results were produced through this path and the
parser is the part that must not change.

The trajectory this returns is optimised over the whole sequence, unlike
pipeline I's causal per-frame poses. That asymmetry is a stated limitation of
the comparison.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path

log = logging.getLogger("export_poses")

COLUMNS = ["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw", "frame_id"]

_POSE_START = "geometry_msgs.msg.PoseStamped("
_HEADER = re.compile(
    r"header=std_msgs\.msg\.Header\("
    r"stamp=builtin_interfaces\.msg\.Time\(sec=(\d+),\s*nanosec=(\d+)\)"
)
_FRAME_ID = re.compile(r"frame_id='([^']*)'")
_POSITION = re.compile(
    r"position=geometry_msgs\.msg\.Point\("
    r"x=([-\d.eE+]+),\s*y=([-\d.eE+]+),\s*z=([-\d.eE+]+)\)"
)
_ORIENTATION = re.compile(
    r"orientation=geometry_msgs\.msg\.Quaternion\("
    r"x=([-\d.eE+]+),\s*y=([-\d.eE+]+),\s*z=([-\d.eE+]+),\s*w=([-\d.eE+]+)\)"
)


def _matching_bracket(text: str, start: int, open_ch: str, close_ch: str) -> int:
    """Index of the bracket closing the one at `start`, or -1.

    Needed because the message `repr` nests parentheses several deep, so the
    end of a pose cannot be found by searching for the next `)`.
    """
    depth = 0
    for i in range(start, len(text)):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return -1


def parse_poses(content: str) -> list[dict]:
    """Extract every pose from a `GetAllPoses` response."""
    success = re.search(r"success=(\w+)", content)
    if success is None:
        raise ValueError("no 'success' field in the response; is this a GetAllPoses reply?")
    if success.group(1).lower() not in ("true", "1"):
        raise ValueError(f"the service reported success={success.group(1)}")

    start = content.find("poses=[")
    if start == -1:
        raise ValueError("no 'poses' field in the response")
    end = _matching_bracket(content, start + len("poses="), "[", "]")
    if end == -1:
        raise ValueError("unterminated 'poses' list in the response")
    body = content[start + len("poses=[") : end]

    offsets = []
    at = body.find(_POSE_START)
    while at != -1:
        offsets.append(at)
        at = body.find(_POSE_START, at + 1)
    if not offsets:
        raise ValueError("the response contains no PoseStamped entries")
    log.info("found %d PoseStamped entries", len(offsets))

    poses, skipped = [], 0
    for index, at in enumerate(offsets):
        close = _matching_bracket(body, at + len(_POSE_START) - 1, "(", ")")
        if close == -1:
            log.warning("pose %d: unterminated entry, skipped", index)
            skipped += 1
            continue
        chunk = body[at : close + 1]

        header, position, orientation = (
            _HEADER.search(chunk), _POSITION.search(chunk), _ORIENTATION.search(chunk)
        )
        if not (header and position and orientation):
            missing = [
                name for name, m in (("header", header), ("position", position),
                                     ("orientation", orientation)) if not m
            ]
            log.warning("pose %d: missing %s, skipped", index, ", ".join(missing))
            skipped += 1
            continue
        frame = _FRAME_ID.search(chunk)
        poses.append({
            "timestamp": int(header.group(1)) + int(header.group(2)) / 1e9,
            "x": float(position.group(1)),
            "y": float(position.group(2)),
            "z": float(position.group(3)),
            "qx": float(orientation.group(1)),
            "qy": float(orientation.group(2)),
            "qz": float(orientation.group(3)),
            "qw": float(orientation.group(4)),
            "frame_id": frame.group(1) if frame else "",
        })

    if skipped:
        # Loud, because a silently short trajectory would still calibrate and
        # still produce plausible residuals.
        log.error("%d of %d poses could not be parsed", skipped, len(offsets))
    return poses


def write_csv(poses: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(poses)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("response", type=Path,
                   help="text file holding the ros2 service call output")
    p.add_argument("-o", "--output", type=Path, required=True, help="output poses.csv")
    p.add_argument("--min-poses", type=int, default=1,
                   help="fail if fewer than this many poses were parsed")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not args.response.is_file():
        p.error(f"no such file: {args.response}")

    poses = parse_poses(args.response.read_text())
    if len(poses) < args.min_poses:
        print(f"error: parsed only {len(poses)} poses, expected at least {args.min_poses}",
              file=sys.stderr)
        return 1
    write_csv(poses, args.output)
    span = poses[-1]["timestamp"] - poses[0]["timestamp"]
    log.info("wrote %d poses spanning %.1f s to %s", len(poses), span, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
