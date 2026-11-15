"""The four SDET workflows. Each takes a Ctx and returns a list of Checks."""
from __future__ import annotations

from . import accelerator, cluster, inference, release

WORKFLOWS = {
    "inference": inference.run,
    "accelerator": accelerator.run,
    "cluster": cluster.run,
    "release": release.run,
}
