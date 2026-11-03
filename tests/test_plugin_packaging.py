import json
import re
import shutil
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_json(relative):
    return json.loads((ROOT / relative).read_text())


def test_plugin_manifests_share_package_identity_and_skill_location():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    portable = load_json("plugin.json")
    codex = load_json(".codex-plugin/plugin.json")
    claude = load_json(".claude-plugin/plugin.json")
    for manifest in (portable, codex, claude):
        assert manifest["name"] == project["name"]
        assert manifest["version"] == project["version"]
        assert manifest["description"] == portable["description"]
        assert manifest["license"] == project["license"]
    assert portable["$schema"] == "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"
    assert codex["skills"].startswith("./")
    assert (ROOT / codex["skills"]).resolve() == ROOT / "skills"
    settings = portable["extensions"]["com.openai"]
    assert settings["requires_local_executor"] is codex["requires_local_executor"] is True
    assert settings["interface"] == codex["interface"]


def test_marketplaces_resolve_to_the_same_installable_plugin():
    catalogs = [
        load_json(".agents/plugins/marketplace.json"),
        load_json(".claude-plugin/marketplace.json"),
    ]
    assert catalogs[0]["name"] == catalogs[1]["name"]
    for catalog in catalogs:
        plugins = {entry["name"]: entry for entry in catalog["plugins"]}
        entry = plugins[load_json("plugin.json")["name"]]
        source = entry["source"]
        path = source["path"] if isinstance(source, dict) else source
        assert path.startswith("./")
        assert (ROOT / path).resolve() == ROOT
        assert (ROOT / path / "skills/weights-lab/SKILL.md").is_file()


def test_standalone_skill_copy_has_no_repository_file_dependencies(tmp_path):
    skill = tmp_path / "personal-skills" / "weights-lab"
    shutil.copytree(ROOT / "skills/weights-lab", skill)
    text = (skill / "SKILL.md").read_text()
    metadata = yaml.safe_load(text.split("---", 2)[1])
    assert metadata["name"] == skill.name
    assert metadata["description"].strip()
    for document in skill.rglob("*.md"):
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", document.read_text()):
            url = urlsplit(target)
            if url.scheme or not url.path:
                continue
            path = (document.parent / unquote(url.path)).resolve()
            assert path.is_relative_to(skill), (document.name, target)
            assert path.exists(), (document.name, target)
    interface = yaml.safe_load((skill / "agents/openai.yaml").read_text())["interface"]
    assert interface["display_name"].strip()
    assert f"${metadata['name']}" in interface["default_prompt"]
