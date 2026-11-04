"""Conservatively decide whether a change needs the model test matrix."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

NON_MODEL_FILES = {b"README.md", b"CONTRIBUTING.md", b"LICENSE", b"plugin.json"}
NON_MODEL_DIRS = (
    b"docs/",
    b"skills/",
    b".claude-plugin/",
    b".codex-plugin/",
    b".agents/plugins/",
)


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True, timeout=30).stdout


def needs_models(base: str) -> bool:
    """Only skip when a trustworthy diff contains exclusively known non-model paths."""
    if not re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", base) or not int(base, 16):
        return True
    try:
        try:
            git("cat-file", "-e", f"{base}^{{commit}}")
        except subprocess.CalledProcessError:
            git("fetch", "--no-tags", "--depth=1", "origin", base)
        paths = git("diff", "--no-renames", "--name-only", "-z", f"{base}..HEAD").split(b"\0")
        paths = [path for path in paths if path]
        return not paths or any(
            path not in NON_MODEL_FILES and not path.startswith(NON_MODEL_DIRS) for path in paths
        )
    except (OSError, subprocess.SubprocessError):
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="")
    args = parser.parse_args()
    print(f"run={str(needs_models(args.base)).lower()}")


if __name__ == "__main__":
    sys.exit(main())
