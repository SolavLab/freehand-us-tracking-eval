"""Reading Vicon Nexus marker-trajectory exports.

The export is not a plain CSV: six header lines precede the data, carrying the
sample rate, the marker names and the length unit. See ``docs/data_formats.md``.

Three things here differ from the original loader, all deliberate:

* The sample rate is **taken from the dataset manifest and checked against the
  file**, rather than hard-coded. The original hard-coded ``240.0`` and never
  read line 4, behind a comment that said 120 Hz. Because the rate propagates
  into every timestamp, interpolation and reported lag, silently reading it from
  the file instead would be a re-analysis rather than a refactor -- so the
  manifest value stays authoritative and the file is used to verify it.
* Marker columns are **located by matching the header**, not by assuming they
  start at index 2 and run contiguously. The original parsed the marker names
  and then never used their positions, so a renamed Vicon subject would quietly
  yield fewer markers and the rigid-body fit would happily fit three of them.
* An unrecognised length unit **raises**. The original left the conversion
  factor unbound and failed later with a ``NameError``.

:class:`MocapData` is immutable. In the original, resampling the reference
mutated the caller's ``Time_ms`` in place; that was safe only because the CSV
was re-read for every pipeline, so the obvious optimisation of loading it once
per recording and sharing it across the three pipelines would have shifted the
second pipeline's clock by the first pipeline's lag and changed every published
number without raising anything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["Marker", "MocapData", "load_mocap"]

log = logging.getLogger(__name__)

_HEADER_LINES = 7
_RATE_LINE = 3          # 0-based: line 4 of the file
_NAME_LINE = 4          # marker names, one per group of three columns
_UNIT_LINE = 6
_UNIT_TO_MM = {"mm": 1.0, "m": 1000.0}


@dataclass(frozen=True)
class Marker:
    """One reflective marker's trajectory, in millimetres."""

    name: str
    xyz: np.ndarray  # (n, 3)

    @property
    def X(self) -> np.ndarray:
        return self.xyz[:, 0]

    @property
    def Y(self) -> np.ndarray:
        return self.xyz[:, 1]

    @property
    def Z(self) -> np.ndarray:
        return self.xyz[:, 2]


@dataclass(frozen=True)
class MocapData:
    """A marker-cluster recording: frame indices, times and marker trajectories."""

    frame: np.ndarray          # (n,) Vicon frame numbers as exported
    time_ms: np.ndarray        # (n,) milliseconds from the first frame
    markers: tuple[Marker, ...]
    rate_hz: float

    def __len__(self) -> int:
        return int(self.frame.size)

    @property
    def positions(self) -> np.ndarray:
        """``(n_frames, n_markers, 3)`` marker positions in millimetres."""
        return np.stack([m.xyz for m in self.markers], axis=1)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(m.name for m in self.markers)

    def shifted(self, delta_ms: float) -> MocapData:
        """A copy with the clock shifted. Never mutates ``self``."""
        return replace(self, time_ms=self.time_ms + delta_ms)

    def cropped(self, t_min_ms: float, t_max_ms: float) -> MocapData:
        """A copy restricted to ``[t_min_ms, t_max_ms]``. Never mutates ``self``."""
        keep = (self.time_ms >= t_min_ms) & (self.time_ms <= t_max_ms)
        if not keep.any():
            raise ValueError(
                f"no reference samples in [{t_min_ms}, {t_max_ms}] ms; "
                f"recording spans [{self.time_ms[0]}, {self.time_ms[-1]}]"
            )
        return replace(
            self,
            frame=self.frame[keep],
            time_ms=self.time_ms[keep],
            markers=tuple(Marker(m.name, m.xyz[keep]) for m in self.markers),
        )


def load_mocap(
    path: str | Path,
    *,
    marker_prefix: str = "ZED:",
    expected_rate_hz: float | None = None,
    expected_markers: int | None = None,
    rate_tolerance: float = 1e-3,
) -> MocapData:
    """Load a Vicon Nexus trajectory export.

    ``expected_rate_hz`` comes from the dataset manifest and is authoritative;
    the rate declared in the file is read only to confirm it agrees.
    """
    path = Path(path)
    with open(path) as fh:
        header = [fh.readline() for _ in range(_HEADER_LINES)]

    file_rate = _parse_rate(header[_RATE_LINE], path)
    rate = expected_rate_hz if expected_rate_hz is not None else file_rate
    if abs(file_rate - rate) / rate > rate_tolerance:
        raise ValueError(
            f"{path}: file declares {file_rate} Hz but the manifest says {rate} Hz. "
            "The sample rate enters every timestamp and reported lag, so this is "
            "not a difference to paper over."
        )

    names, columns = _locate_markers(header[_NAME_LINE], marker_prefix, path)
    if expected_markers is not None and len(names) != expected_markers:
        raise ValueError(
            f"{path}: found {len(names)} markers matching {marker_prefix!r} "
            f"({', '.join(names)}), manifest expects {expected_markers}"
        )
    scale = _parse_unit(header[_UNIT_LINE], path)

    df = pd.read_csv(path, skiprows=1, header=5)
    frames = df.iloc[:, 0].to_numpy()
    markers = tuple(
        Marker(name, df.iloc[:, [c, c + 1, c + 2]].to_numpy(dtype=float) * scale)
        for name, c in zip(names, columns, strict=True)
    )

    time_ms = (frames - frames[0]) / rate * 1000.0
    missing = int(np.isnan(np.stack([m.xyz for m in markers])).any(axis=(1, 2)).sum())
    if missing:
        log.info("%s: %d of %d frames have at least one occluded marker",
                 path.name, missing, len(frames))
    return MocapData(frame=frames, time_ms=time_ms, markers=markers, rate_hz=float(rate))


def _parse_rate(line: str, path: Path) -> float:
    token = line.split(",")[0].strip()
    try:
        return float(token)
    except ValueError as exc:
        raise ValueError(
            f"{path}: expected the sample rate on line {_RATE_LINE + 1}, found {token!r}"
        ) from exc


def _locate_markers(line: str, prefix: str, path: Path) -> tuple[list[str], list[int]]:
    """Return marker names and the index of each one's X column.

    Nexus writes each marker's name once, above its X column, leaving the Y and
    Z columns blank -- so the column index carries the association and must be
    used rather than assumed.
    """
    fields = line.rstrip("\n").split(",")
    names, columns = [], []
    for index, field in enumerate(fields):
        name = field.strip()
        if name and prefix in name:
            names.append(name)
            columns.append(index)
    if not names:
        raise ValueError(f"{path}: no marker names containing {prefix!r} on line {_NAME_LINE + 1}")
    gaps = {b - a for a, b in zip(columns, columns[1:], strict=False)}
    if gaps - {3}:
        raise ValueError(
            f"{path}: marker columns are not in contiguous groups of three "
            f"(offsets {sorted(gaps)}); the export layout is not what this reader assumes"
        )
    return names, columns


def _parse_unit(line: str, path: Path) -> float:
    units = {u.strip().lower() for u in line.split(",") if u.strip()}
    if len(units) != 1:
        raise ValueError(f"{path}: inconsistent length units {sorted(units)}")
    unit = units.pop()
    if unit not in _UNIT_TO_MM:
        raise ValueError(
            f"{path}: unrecognised length unit {unit!r}; expected one of "
            f"{sorted(_UNIT_TO_MM)}"
        )
    return _UNIT_TO_MM[unit]
