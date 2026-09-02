"""Reading pipeline pose trajectories.

The three pipelines are normalised to one CSV schema, specified in
``datasets/*/schema/poses.schema.json`` and validated here on every load. That
validation is the contract between the extraction stages -- which run under
different interpreters, and in one case inside a container -- and this package:
a change in an extractor surfaces as a load-time error rather than as a quietly
different number.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["Track", "load_track", "REQUIRED_COLUMNS"]

REQUIRED_COLUMNS = (
    "Frame",
    "Timestamp",
    "Translation_X",
    "Translation_Y",
    "Translation_Z",
    "Rotation_X",
    "Rotation_Y",
    "Rotation_Z",
)


@dataclass(frozen=True)
class Track:
    """One pipeline's estimated camera trajectory in its SLAM world frame."""

    frame: np.ndarray        # (n,) the pipeline's own frame index
    timestamp_ms: np.ndarray  # (n,) absolute acquisition clock
    translation: np.ndarray  # (n, 3) millimetres
    rotvec: np.ndarray       # (n, 3) radians, axis-angle
    state: pd.DataFrame      # diagnostic columns, kept as-is

    def __len__(self) -> int:
        return int(self.frame.size)

    @property
    def time_ms(self) -> np.ndarray:
        """Milliseconds from this track's first sample."""
        return self.timestamp_ms - self.timestamp_ms[0]

    def cropped(self, t_min_ms: float, t_max_ms: float) -> Track:
        """A copy restricted to a window in *relative* time. Never mutates."""
        keep = (self.time_ms >= t_min_ms) & (self.time_ms <= t_max_ms)
        if not keep.any():
            raise ValueError(
                f"no samples in [{t_min_ms}, {t_max_ms}] ms; track spans "
                f"[0, {self.time_ms[-1]}]"
            )
        return replace(
            self,
            frame=self.frame[keep],
            timestamp_ms=self.timestamp_ms[keep],
            translation=self.translation[keep],
            rotvec=self.rotvec[keep],
            state=self.state.loc[keep].reset_index(drop=True),
        )

    def repeated_poses(self) -> np.ndarray:
        """Indices where all six pose components equal the previous row.

        A pipeline that returns the previous pose unchanged is reporting a
        stagnant estimate, which contributes a residual growing with whatever
        motion occurred during the repetition. The paper reports these counts
        per pipeline.
        """
        pose = np.hstack([self.translation, self.rotvec])
        same = np.all(pose[1:] == pose[:-1], axis=1)
        return np.flatnonzero(same) + 1

    def omitted_frames(self) -> int:
        """Count of expected frames for which no pose was returned.

        Derived from the timestamps, **not** from ``Frame``: cuVSLAM renumbers
        its output sequentially, so a dropped input frame leaves no gap in the
        frame column and index continuity finds zero omissions for every
        pipeline. Counted as ``round(dt / median(dt)) - 1`` summed over the
        track.
        """
        dt = np.diff(self.timestamp_ms.astype(float))
        if dt.size == 0:
            return 0
        step = float(np.median(dt))
        if step <= 0:
            raise ValueError("median sample interval is not positive")
        return int(np.sum(np.maximum(np.round(dt / step) - 1, 0)))


def load_track(path: str | Path) -> Track:
    """Load a pose CSV, validating the schema."""
    path = Path(path)
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: missing required column(s) {missing}. Expected the common "
            f"pose schema; found {list(df.columns)}"
        )
    if df.empty:
        raise ValueError(f"{path}: no data rows")

    timestamp = df["Timestamp"].to_numpy(dtype="int64")
    if np.any(np.diff(timestamp) < 0):
        raise ValueError(f"{path}: Timestamp is not monotonically increasing")

    diagnostics = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    return Track(
        frame=df["Frame"].to_numpy(dtype="int64"),
        timestamp_ms=timestamp,
        translation=df[["Translation_X", "Translation_Y", "Translation_Z"]].to_numpy(float),
        rotvec=df[["Rotation_X", "Rotation_Y", "Rotation_Z"]].to_numpy(float),
        state=df[diagnostics].reset_index(drop=True),
    )
