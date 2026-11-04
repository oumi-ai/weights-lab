"""Evaluation rows from a Hugging Face dataset split, pinned to the commit they came from."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

INSTALL = (
    "uv tool install 'oumi-weights-lab[datasets] @ git+https://github.com/oumi-ai/weights-lab'"
)


def hf_rows(
    path: str,
    *,
    split: str,
    to_row: Callable[[dict[str, Any]], dict[str, Any]],
    name: str | None = None,
    revision: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load a split and turn each example into an evaluation row with ``to_row``.

    A Hub dataset's revision (a branch, tag, or None for the default branch) is
    resolved to its commit first, so the rows can be reloaded exactly. Each row
    records where it came from under ``metadata.hf``, never under
    ``metadata.split``, so a Hub "test" split is not mistaken for declared
    held-out data. Rows without an ``id`` get one from their source and index.
    """
    if not callable(to_row):
        raise ValueError("to_row must be callable")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer")
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError(f"hf_rows needs the datasets extra: {INSTALL}") from error
    local = Path(path).exists()
    if local and revision is not None:
        raise ValueError("a local dataset directory has no revision to pin")
    if local:
        commit = None
    else:
        from huggingface_hub import HfApi

        # Resolve once, including when the caller names a tag or branch.
        commit = HfApi().dataset_info(path, revision=revision).sha
    loaded = load_dataset(path, name, split=split, revision=commit)
    rows = []
    for index, example in enumerate(loaded):
        if limit is not None and index >= limit:
            break
        row = to_row(dict(example))
        if not isinstance(row, dict):
            raise ValueError("to_row must return a dict for each example")
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("a row's metadata must be a dict")
        source = {"path": path, "name": name, "split": split, "revision": commit, "index": index}
        rows.append(
            {
                "id": f"{path}/{name or 'default'}/{split}/{index}",
                **row,
                "metadata": {**metadata, "hf": source},
            }
        )
    return rows
