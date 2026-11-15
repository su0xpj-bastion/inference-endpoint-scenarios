"""The scenarios. Each takes a Ctx and returns a list of Checks."""
from __future__ import annotations

from . import accelerator, agent, cluster, inference, release, wire

WORKFLOWS = {
    "inference": inference.run,
    "accelerator": accelerator.run,
    "cluster": cluster.run,
    "release": release.run,
    "wire": wire.run,
    "agent": agent.run,
}
