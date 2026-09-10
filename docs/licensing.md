# Licensing and attribution

## Licence

**MIT**, see `LICENSE`. Copyright 2026 Zohar Oddes and Dana Solav, Faculty of
Mechanical Engineering, Technion -- Israel Institute of Technology.

MIT is compatible with everything listed below: the upstream projects this work
builds on are Apache-2.0 (NVIDIA Isaac ROS) and permissive (Stereolabs), and
none of their source is redistributed here in any case.

## Third-party code is not redistributed

This was a deliberate design decision, and it is why the repository is small.

The cuVSLAM pipelines depend on NVIDIA Isaac ROS and the Stereolabs ZED ROS 2
wrapper. In the original working tree those were present as clones with their
`.git` directories stripped and their upstream `LICENSE` files untracked, which
would have meant redistributing NVIDIA and Stereolabs code without its licence.

Instead, `environments/isaac_ros/` contains:

- `repos.yaml` — a `vcs` manifest pinning each upstream repository to the exact
  revision used (`isaac_ros_common` at `v3.2-8-7-ga40a017`, Isaac ROS 3.2, the
  Stereolabs wrapper).
- `patches/` — the modifications, as patches against those pinned revisions.

Fetching upstream is therefore the reader's action, under the upstream licences,
and every modification made here is visible as a diff rather than buried in a
copied tree.

The same reasoning applies on the ZED SDK side. Pipeline-I pose extraction was
originally built on a Stereolabs sample and imported that sample's OpenGL viewer
(~1,600 lines of vendored code). The published extractor is headless: the viewer
was used only for on-screen display and is not needed to produce a pose CSV, so
it is gone and no Stereolabs source is redistributed. `pyzed` remains a normal
installed dependency.

## Derivative works requiring attention

Two launch files under `extract/cuvslam/launch/` were written for this work but
began as copies of NVIDIA templates and **carry NVIDIA's copyright header
verbatim**. Their headers must be amended to state that they are derived from
the NVIDIA original and modified here, rather than left implying sole NVIDIA
authorship.

One Dockerfile written for this work inherited NVIDIA's **proprietary** header —
the one asserting that use, reproduction, disclosure or distribution without an
express NVIDIA licence agreement is prohibited. That header must not be
republished on a file authored here. It is removed and replaced with this
project's own notice plus a note on what the file layers onto the NVIDIA base
image.

## Data

The Vicon reference trajectories and the pipeline pose CSVs under `datasets/`
were produced by the authors and are covered by the MIT licence above. The
raw recordings (~20 GiB, see `datasets/probe-tracking-2025-10-23/raw.yaml`)
are too large for this repository and are planned for a separate Zenodo
deposit with its own DOI; CC-BY-4.0 is worth considering for that deposit
specifically, since MIT is written for software. Nothing in the paper's
claims depends on the raw recordings — every table and figure is reproducible
from the CSVs that do ship — but the acquisition stages are not runnable
without them.

## Attribution note

One README in the original tree recorded a debugging session as
"Original source: ChatGPT-assisted debugging". That is fine as a working note
and out of place in a published artifact; the file is reworded to describe what
the configuration does and why.
