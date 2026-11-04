"""Checks for broken published links and malformed plugin contracts."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_repo", ROOT / "scripts/check_repo.py")
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


def test_published_repository_contracts():
    assert checks.check_links(ROOT) == []
    assert checks.check_manifests(ROOT) == []


def test_links_include_anchors_references_and_ignore_fenced_examples(tmp_path):
    (tmp_path / "README.md").write_text(
        "[good](docs/page.md#repeat-1)\n[bad](docs/page.md#absent)\n"
        "[missing][ref]\n[ref]: docs/missing.md\n"
        "```md\n[example](missing.md)\n```\n"
        "~~~\n[example](also-missing.md)\n~~~\n"
        "    [indented example](missing.md)\n"
        "[web](https://example.com/missing)\n"
    )
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/page.md").write_text("# Repeat\n## Repeat\n")
    assert checks.check_links(tmp_path) == [
        "README.md: docs/page.md#absent: missing anchor",
        "README.md: docs/missing.md: missing file",
    ]


@pytest.mark.parametrize("field,value", [("version", "999"), ("description", []), ("name", "")])
def test_manifest_contract_rejects_invalid_identity(tmp_path, field, value):
    for relative in (
        "plugin.json",
        ".claude-plugin",
        ".codex-plugin",
        ".agents/plugins",
        "skills",
        "pyproject.toml",
    ):
        source = ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copyfile(source, target)
    manifest_path = tmp_path / "plugin.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest))
    assert any(f"plugin.json: {field}" in error for error in checks.check_manifests(tmp_path))
