"""Checks for broken published links and malformed plugin contracts."""

import importlib.util
import json
import shutil
import subprocess
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


def copy_manifests(tmp_path):
    for relative in (
        "plugin.json",
        ".claude-plugin",
        ".codex-plugin",
        ".agents/plugins",
        "skills",
        "pyproject.toml",
        "src/oumi_weights_lab/__init__.py",
    ):
        source = ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copyfile(source, target)


@pytest.mark.parametrize("field,value", [("version", "999"), ("description", []), ("name", "")])
def test_manifest_contract_rejects_invalid_identity(tmp_path, field, value):
    copy_manifests(tmp_path)
    manifest_path = tmp_path / "plugin.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest))
    assert any(f"plugin.json: {field}" in error for error in checks.check_manifests(tmp_path))


def test_manifest_contract_rejects_diverging_interface_and_catalogs(tmp_path):
    copy_manifests(tmp_path)
    codex_path = tmp_path / ".codex-plugin/plugin.json"
    codex = json.loads(codex_path.read_text())
    codex["interface"]["displayName"] = "Something Else"
    codex_path.write_text(json.dumps(codex))
    catalog_path = tmp_path / ".claude-plugin/marketplace.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["name"] = "another-catalog"
    catalog_path.write_text(json.dumps(catalog))
    errors = checks.check_manifests(tmp_path)
    assert ".codex-plugin/plugin.json: interface differs from portable manifest" in errors
    assert "marketplace catalogs must share one name" in errors


def test_links_skip_documents_git_ignores(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("docs/scratch.md\n")
    (tmp_path / "README.md").write_text("[docs](docs/page.md)\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/page.md").write_text("# Page\n")
    (tmp_path / "docs/scratch.md").write_text("[private](notes/missing.md)\n")
    (tmp_path / "docs/new.md").write_text("[broken](missing.md)\n")
    assert checks.check_links(tmp_path) == ["docs/new.md: missing.md: missing file"]


def test_manifest_contract_rejects_a_half_bumped_version(tmp_path):
    copy_manifests(tmp_path)
    package = tmp_path / "src/oumi_weights_lab/__init__.py"
    package.write_text(package.read_text().replace('__version__ = "', '__version__ = "9'))
    assert "src/oumi_weights_lab/__init__.py: __version__ differs from package" in (
        checks.check_manifests(tmp_path)
    )
