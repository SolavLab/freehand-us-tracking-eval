"""Dataset manifest loading and cell addressing.

Everything in this package addresses a recording and a pipeline by their
symbolic ids from ``dataset.yaml``. Nothing globs filenames or builds them from
format strings.

That is a deliberate correction. The code this was ported from discovered its
inputs with patterns like ``"{test}_ZED_SDK_trajectory.csv"`` against a
directory tree, and when a file was missing it returned ``None`` and the loop
continued -- so a missing input silently dropped a cell from what was reported
as a complete comparison. Here :meth:`Dataset.load` verifies every file in the
manifest exists and hashes as recorded, and raises before any analysis starts.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

__all__ = ["Cell", "Dataset", "DatasetError"]


class DatasetError(Exception):
    """The dataset on disk does not match its manifest."""


@dataclass(frozen=True, order=True)
class Cell:
    """One pipeline evaluated on one recording -- the unit of the comparison."""

    pipeline: str
    recording: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.pipeline}/{self.recording}"


class Dataset:
    """A loaded, verified dataset directory."""

    def __init__(self, root: Path, manifest: dict):
        self.root = Path(root)
        self.manifest = manifest

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, root: str | Path, *, verify_hashes: bool = True) -> Dataset:
        """Load ``<root>/dataset.yaml`` and check the tree matches it.

        Raises :class:`DatasetError` on the first discrepancy rather than
        letting a partial dataset through.
        """
        root = Path(root)
        path = root / "dataset.yaml"
        if not path.is_file():
            raise DatasetError(f"no dataset.yaml in {root}")
        manifest = yaml.safe_load(path.read_text())

        for key in ("id", "vicon", "recordings", "pipelines", "files"):
            if key not in manifest:
                raise DatasetError(f"{path}: missing top-level key {key!r}")

        problems = []
        for rel, meta in manifest["files"].items():
            f = root / rel
            if not f.is_file():
                problems.append(f"missing: {rel}")
                continue
            if verify_hashes:
                got = _sha256(f)
                if got != meta["sha256"]:
                    problems.append(
                        f"hash mismatch: {rel}\n"
                        f"    manifest {meta['sha256']}\n"
                        f"    on disk  {got}"
                    )
        if problems:
            raise DatasetError(
                f"{root} does not match its manifest:\n  " + "\n  ".join(problems)
            )
        return cls(root, manifest)

    # -------------------------------------------------------------- accessors

    @property
    def id(self) -> str:
        return self.manifest["id"]

    @property
    def pipelines(self) -> list[str]:
        return list(self.manifest["pipelines"])

    def recordings(self, *, published: bool | None = True) -> list[str]:
        """Recording ids, by default only those the paper reports."""
        return [
            r
            for r, m in self.manifest["recordings"].items()
            if published is None or bool(m.get("published", False)) is published
        ]

    def cells(self, *, published: bool | None = True) -> list[Cell]:
        """Every pipeline x recording pair, in a stable order."""
        return [
            Cell(p, r) for p in self.pipelines for r in self.recordings(published=published)
        ]

    def label(self, *, pipeline: str | None = None, recording: str | None = None) -> str:
        """The human-readable name used in tables and figures."""
        if pipeline is not None:
            return self.manifest["pipelines"][pipeline]["label"]
        return self.manifest["recordings"][recording]["label"]

    # ------------------------------------------------------------------ paths

    def tracking_csv(self, cell: Cell) -> Path:
        return self._checked(f"tracking/{cell.pipeline}/{cell.recording}/poses.csv")

    def reference_csv(self, recording: str) -> Path:
        return self._checked(f"reference/{recording}/mocap.csv")

    def rows(self, rel: str) -> int:
        """Data-row count recorded in the manifest, for cheap sanity checks."""
        return self.manifest["files"][rel]["rows"]

    def _checked(self, rel: str) -> Path:
        if rel not in self.manifest["files"]:
            raise DatasetError(f"{rel} is not in the manifest of {self.root}")
        return self.root / rel

    # --------------------------------------------------------------- metadata

    @property
    def vicon_rate_hz(self) -> float:
        """Authoritative Vicon sample rate.

        The loader for the mocap CSV asserts the rate stated on line 4 of the
        export agrees with this. The original hard-coded 240.0 and never read
        the file, behind a comment that said 120.
        """
        return float(self.manifest["vicon"]["nominal_rate_hz"])

    @property
    def marker_prefix(self) -> str:
        return self.manifest["vicon"]["marker_prefix"]

    @property
    def marker_order(self) -> list[str]:
        return list(self.manifest["vicon"]["marker_order"])

    def excluded(self) -> dict[str, dict]:
        """Recordings acquired but not reported, with the reason for each."""
        return dict(self.manifest.get("excluded", {}))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
