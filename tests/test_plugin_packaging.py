"""A standalone skill copy must work without this repository.

Manifest identity and marketplace contracts are checked by scripts/check_repo.py
(see tests/test_repo_checks.py).
"""

import re
import shutil
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_standalone_skill_copy_has_no_repository_file_dependencies(tmp_path):
    skill = tmp_path / "personal-skills" / "oumi-weights-lab"
    shutil.copytree(ROOT / "skills/oumi-weights-lab", skill)
    text = (skill / "SKILL.md").read_text()
    metadata = yaml.safe_load(text.split("---", 2)[1])
    assert metadata["name"] == skill.name
    assert metadata["description"].strip()
    for document in skill.rglob("*.md"):
        body = document.read_text()
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", body):
            url = urlsplit(target)
            if url.scheme or not url.path:
                continue
            path = (document.parent / unquote(url.path)).resolve()
            assert path.is_relative_to(skill), (document.name, target)
            assert path.exists(), (document.name, target)
        # Repository docs named as bare paths are missing from a standalone copy.
        assert not re.findall(r"`(?:docs|references)/[^`\s]+\.md`", body), document.name
    interface = yaml.safe_load((skill / "agents/openai.yaml").read_text())["interface"]
    assert interface["display_name"] == "Oumi Weights Lab"
    assert f"${metadata['name']}" in interface["default_prompt"]
