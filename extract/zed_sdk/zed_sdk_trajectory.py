#!/usr/bin/env python3
"""Pipeline I: camera poses from an SVO2 recording, via the ZED SDK.

ENV: conda `zed-sdk` -- Python 3.12 with pyzed 5.3 and ZED SDK 5.3.0.
     See ../../docs/setup.md. This script must not import `vipose`.

Decodes an SVO2 recording in offline playback mode and runs the ZED SDK's own
positional-tracking module over it, writing one row per frame for which tracking
reported a valid pose. Output conforms to
`datasets/*/schema/poses.schema.json`.

Headless by design. The original version of this script drove its grab loop from
the render callback of a Stereolabs OpenGL viewer, and imported roughly 1,600
lines of vendored sample code to do it. None of that affects the poses: the
viewer only displayed them, playback is offline (`svo_real_time_mode = False`),
depth is disabled, and positional tracking runs inside `grab()`. Removing it
makes the script self-contained, runnable over ssh, and free of redistributed
third-party source.

Poses are written as the tracking module produces them. The ZED SDK Python API
offers no way to retrieve a globally optimised trajectory once playback has
finished, so unlike the cuVSLAM pipelines these poses are causal -- not refined
using later observations. That asymmetry is a stated limitation of the
comparison, not an oversight here.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

try:
    import pyzed.sl as sl
except ImportError:  # pragma: no cover
    sys.exit(
        "pyzed is not installed. This script needs the ZED SDK and its Python "
        "bindings; see docs/setup.md, section 'zed-sdk'."
    )

log = logging.getLogger("zed_sdk_trajectory")

COLUMNS = [
    "Frame", "Timestamp",
    "Translation_X", "Translation_Y", "Translation_Z",
    "Rotation_X", "Rotation_Y", "Rotation_Z",
    "Pose_Confidence", "Tracking_State",
    "Spatial_Memory", "Odometry", "Tracking_Fusion",
]

#: Metres to millimetres. The SDK reports metres; the common schema is
#: millimetres, matching the Vicon reference.
M_TO_MM = 1000.0


def build_init_parameters(svo: Path) -> sl.InitParameters:
    """Initialisation parameters, as used for the published results.

    `DEPTH_MODE.NONE` is deliberate: GEN_3 positional tracking does not consume
    a depth map, so computing one only costs GPU time. (One superseded variant
    of this script set `NEURAL_PLUS` here, which is slower for no benefit.)

    The coordinate system is the SDK's right-handed Y-up convention and the
    reported frame is the left optical sensor. That differs from the cuVSLAM
    pipelines, which report their configured base frame in the ROS convention;
    the difference is a constant rigid transformation, which the hand-eye
    calibration absorbs.
    """
    params = sl.InitParameters()
    params.set_from_svo_file(str(svo))
    params.svo_real_time_mode = False        # decode as fast as possible
    params.depth_mode = sl.DEPTH_MODE.NONE
    params.coordinate_units = sl.UNIT.METER
    params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    return params


def build_tracking_parameters(area_file: Path | None = None) -> sl.PositionalTrackingParameters:
    """Tracking parameters, as used for the published results.

    Spatial memory and IMU fusion enabled, pose smoothing disabled, GEN_3 mode;
    everything else at its default.

    `area_file` loads a prebuilt spatial-memory map. The published runs did
    **not** use one -- spatial memory was built fresh from each recording -- so
    this stays None unless asked for. It is offered because the SDK supports it
    and someone re-running this may want to.
    """
    params = sl.PositionalTrackingParameters()
    params.mode = sl.POSITIONAL_TRACKING_MODE.GEN_3
    params.enable_area_memory = True
    params.enable_imu_fusion = True
    params.enable_pose_smoothing = False
    params.set_floor_as_origin = False
    params.set_gravity_as_origin = False
    params.set_as_static = False
    params.depth_min_range = -1
    if area_file is not None:
        params.area_file_path = str(area_file)
        log.info("loading spatial-memory map from %s", area_file)
    return params


def extract(svo: Path, out_csv: Path, *, area_file: Path | None = None,
            confidence_threshold: int = 30) -> int:
    """Run tracking over `svo`, writing poses to `out_csv`. Returns rows written."""
    log.info("ZED SDK %s", sl.Camera().get_sdk_version())
    log.info("input  %s", svo)
    log.info("output %s", out_csv)

    camera = sl.Camera()
    status = camera.open(build_init_parameters(svo))
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"could not open {svo}: {status}")

    try:
        status = camera.enable_positional_tracking(build_tracking_parameters(area_file))
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"could not enable positional tracking: {status}")

        runtime = sl.RuntimeParameters()
        runtime.confidence_threshold = confidence_threshold
        pose = sl.Pose()

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        frame = written = 0
        with open(out_csv, "w", newline="") as fh:
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow(COLUMNS)

            while True:
                status = camera.grab(runtime)
                if status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    break
                if status != sl.ERROR_CODE.SUCCESS:
                    raise RuntimeError(f"grab failed at frame {frame}: {status}")

                # WORLD is the SLAM world frame, initialised at the pose where
                # tracking began.
                state = camera.get_position(pose, sl.REFERENCE_FRAME.WORLD)
                tracking = camera.get_positional_tracking_status()

                if state == sl.POSITIONAL_TRACKING_STATE.OK:
                    t = pose.get_translation().get()
                    r = pose.get_rotation_vector()
                    writer.writerow([
                        frame,
                        pose.timestamp.get_milliseconds(),
                        t[0] * M_TO_MM, t[1] * M_TO_MM, t[2] * M_TO_MM,
                        r[0], r[1], r[2],
                        pose.pose_confidence,
                        state.name,
                        tracking.spatial_memory_status.name,
                        tracking.odometry_status.name,
                        tracking.tracking_fusion_status.name,
                    ])
                    written += 1

                # Counts every decoded frame, not only the written ones, so
                # Frame remains the SVO frame index and a gap in it means
                # tracking was not valid there.
                frame += 1
                if frame % 500 == 0:
                    log.info("  %d frames decoded, %d poses written", frame, written)
    finally:
        camera.close()

    log.info("done: %d frames decoded, %d poses written (%d without a valid pose)",
             frame, written, frame - written)
    if written == 0:
        raise RuntimeError("tracking never reported a valid pose; output is empty")
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("svo", type=Path, help="input .svo2 recording")
    p.add_argument("-o", "--output", type=Path, required=True, help="output pose CSV")
    p.add_argument("--area-file", type=Path, default=None,
                   help="prebuilt spatial-memory map (not used for the published results)")
    p.add_argument("--confidence-threshold", type=int, default=30)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if not args.svo.is_file():
        p.error(f"no such file: {args.svo}")
    extract(args.svo, args.output, area_file=args.area_file,
            confidence_threshold=args.confidence_threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
