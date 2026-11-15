"""The scenarios. Each takes a Ctx and returns a list of Checks."""
from __future__ import annotations

from . import accelerator, agent, cluster, compat, conversation, inference, longctx, release, wire

WORKFLOWS = {
    "inference": inference.run,
    "accelerator": accelerator.run,
    "cluster": cluster.run,
    "release": release.run,
    "wire": wire.run,
    "agent": agent.run,
    "compat": compat.run,
    "conversation": conversation.run,
    "longctx": longctx.run,
}
