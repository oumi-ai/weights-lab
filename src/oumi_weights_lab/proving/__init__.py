"""Proving ground: tasks with hidden answer keys, a claim format, and a scorer.

Build a task with ``build_task``, hand its directory to an agent, and grade the claim the
agent writes with ``grade_claim`` against the key. ``compare`` pairs two arms of results.
Importing this package does not import torch or transformers.
"""

from .claims import CLAIM_FORMAT, load_claim, validate_claim
from .grade import grade_bundle, grade_claim
from .report import beats, compare
from .tasks import (
    ALL_FAMILIES,
    SPLITS,
    build_task,
    load_key,
    read_secret,
    task_id,
    task_seed,
)

__all__ = [
    "ALL_FAMILIES",
    "CLAIM_FORMAT",
    "SPLITS",
    "beats",
    "build_task",
    "compare",
    "grade_bundle",
    "grade_claim",
    "load_claim",
    "load_key",
    "read_secret",
    "task_id",
    "task_seed",
    "validate_claim",
]
