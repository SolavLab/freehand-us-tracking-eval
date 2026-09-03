"""The manuscript's figures regenerate.

Not pixel comparisons against the published PNGs: matplotlib is not
byte-reproducible across versions, and a test that fails on a font-rendering
change would be noise. These check that each figure generates, has the expected
geometry, and -- for the stage-3 panel -- that the underlying curve still has
the shape the appendix describes, which is the part that carries meaning.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from vipose import Dataset  # noqa: E402
from vipose.io.mocap import load_mocap  # noqa: E402
from vipose.io.tracks import load_track  # noqa: E402
from vipose.recordings import Cell  # noqa: E402
from vipose.results import read_calibration  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden"
DATASET = Dataset.load(ROOT / "datasets" / "probe-tracking-2025-10-23", verify_hashes=False)


def test_appendix_figures_generate(tmp_path):
    from vipose.report.figures import write_appendix_figures

    written = write_appendix_figures(DATASET, GOLDEN, tmp_path)
    assert len(written) == 2
    for path in written:
        assert path.is_file() and path.stat().st_size > 20_000, f"{path} looks empty"


def test_no_pyplot_outside_the_report_package():
    """Numerical modules must not import pyplot, even transitively.

    In the code this replaced, the residual values were returned *by* the
    plotting functions, so results could not be computed without drawing
    figures -- which is why a matplotlib error could destroy a twelve-cell run
    and why the figures leaked until the process ran out of memory.
    """
    import subprocess
    import sys

    probe = (
        "import sys; import vipose.pipeline, vipose.calibration, vipose.metrics, "
        "vipose.motion, vipose.conditioning, vipose.sync; "
        "print('matplotlib.pyplot' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                         env={**os.environ, "MPLBACKEND": "nonexistent-backend"})
    assert out.returncode == 0, f"importing the numeric modules failed:\n{out.stderr}"
    assert out.stdout.strip() == "False", "a numeric module pulled in pyplot"


@pytest.mark.parametrize("recording", ["pivot", "mixed", "freehand"])
def test_stage3_objective_curve_has_a_minimum_near_the_refined_offset(recording):
    """The curve's minimum should sit where stage 3 said it does.

    This is what the appendix's figure shows, and it is the substantive claim:
    the scalar angular-speed criterion of stages 1 and 2 lands away from the
    pose-alignment optimum, by up to about 12 ms for ZED-SDK.
    """
    from vipose.pipeline import stage3_objective_curve

    mocap = load_mocap(
        DATASET.reference_csv(recording), marker_prefix=DATASET.marker_prefix,
        expected_rate_hz=DATASET.vicon_rate_hz,
    )
    for pipeline in DATASET.pipelines:
        calib = read_calibration(GOLDEN / pipeline / recording / "calibration.json")
        stages = calib["offset_refinement_stages"]
        track = load_track(DATASET.tracking_csv(Cell(pipeline, recording)))

        offsets, values = stage3_objective_curve(
            track, mocap, stages["stage2_optimized_ms"], tuple(calib["local_window_ms"])
        )
        assert len(offsets) == len(values) > 5
        assert np.all(np.isfinite(values))

        # The grid minimum must agree with the refined offset to within the grid
        # step; stage 3 then refines inside that bracket.
        grid_best = float(offsets[int(np.argmin(values))])
        refined = stages["stage3_offset_ms"] - stages["stage2_optimized_ms"]
        assert abs(grid_best - refined) <= 4.0 + 1e-6, (
            f"{pipeline}/{recording}: grid minimum at {grid_best:+.1f} ms but stage 3 "
            f"refined to {refined:+.1f} ms"
        )


def test_zed_sdk_objective_is_the_one_that_moves_most():
    """The appendix's substantive finding, checked rather than asserted.

    Stage 3 shifts ZED-SDK's pivot offset by about 12 ms while both cuVSLAM
    pipelines stay within 4 ms. That asymmetry is the sole surviving evidence
    that the ZED SDK stream is the hardest to align to an external reference.
    """
    shifts = {}
    for pipeline in DATASET.pipelines:
        stages = read_calibration(
            GOLDEN / pipeline / "pivot" / "calibration.json"
        )["offset_refinement_stages"]
        shifts[pipeline] = abs(stages["stage3_delta_ms"])

    assert shifts["zed-sdk"] > 8.0, f"expected a large ZED-SDK shift, got {shifts}"
    assert shifts["zed-cuvslam"] < 4.0 and shifts["rs-cuvslam"] < 4.0, shifts
    assert shifts["zed-sdk"] > 2 * max(shifts["zed-cuvslam"], shifts["rs-cuvslam"]), shifts
