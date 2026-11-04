"""Offline checks for published documentation and plugin manifest contracts."""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit


def prose(text: str) -> str:
    """Remove code blocks so example links and headings do not affect checks."""
    lines = []
    fence = None
    for line in text.splitlines():
        match = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if match:
            marker = match[1]
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            continue
        if fence is None and not line.startswith(("    ", "\t")):
            lines.append(line)
    return "\n".join(lines)


def anchors(text: str) -> set[str]:
    result = set()
    counts: dict[str, int] = {}
    for heading in re.findall(r"^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$", prose(text), re.MULTILINE):
        heading = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", heading)
        heading = re.sub(r"<[^>]*>", "", heading).lower()
        slug = re.sub(r"[^\w\- ]", "", heading).replace(" ", "-")
        suffix = counts.get(slug, 0)
        counts[slug] = suffix + 1
        result.add(f"{slug}-{suffix}" if suffix else slug)
    result.update(re.findall(r"(?:id|name)=[\"\']([^\"\']+)[\"\']", prose(text)))
    return result


def check_links(root: Path) -> list[str]:
    errors = []
    documents = [root / "README.md", root / "CONTRIBUTING.md"]
    documents += sorted((root / "docs").glob("*.md"))
    documents += sorted((root / "skills").rglob("*.md"))
    for document in documents:
        if document.name == "PLAN.md" or not document.exists():
            continue
        text = prose(document.read_text())
        targets = re.findall(r"!?\[[^\]]*\]\(\s*(<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)", text)
        targets += re.findall(r"^ {0,3}\[[^]]+\]:\s*(<[^>]+>|\S+)", text, re.MULTILINE)
        for target in targets:
            target = target.removeprefix("<").removesuffix(">")
            url = urlsplit(target)
            if url.scheme or url.netloc:
                continue
            path = (document.parent / unquote(url.path)).resolve() if url.path else document
            label = f"{document.relative_to(root)}: {target}"
            if not path.exists():
                errors.append(f"{label}: missing file")
            elif url.fragment and path.suffix.lower() == ".md":
                if unquote(url.fragment) not in anchors(path.read_text()):
                    errors.append(f"{label}: missing anchor")
    return errors


def check_manifests(root: Path) -> list[str]:
    """Validate the manifest fields this plugin uses, without harness installs."""
    errors = []
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    manifests = {}
    for relative in ("plugin.json", ".codex-plugin/plugin.json", ".claude-plugin/plugin.json"):
        try:
            manifest = json.loads((root / relative).read_text())
            if not isinstance(manifest, dict):
                raise ValueError("manifest must be an object")
            manifests[relative] = manifest
            for key in ("name", "version", "description", "repository", "license"):
                if not isinstance(manifest.get(key), str) or not manifest[key].strip():
                    errors.append(f"{relative}: {key} must be a nonempty string")
            for key in ("name", "version", "license"):
                if manifest.get(key) != project[key]:
                    errors.append(f"{relative}: {key} differs from package")
        except (OSError, ValueError) as exc:
            errors.append(f"{relative}: {exc}")
    portable = manifests.get("plugin.json", {})
    codex = manifests.get(".codex-plugin/plugin.json", {})
    frontend_name = portable.get("name", "")
    skill = root / "skills" / frontend_name / "SKILL.md"
    skill_name = None
    if skill.is_file():
        frontmatter = skill.read_text().split("---", 2)
        if len(frontmatter) == 3:
            match = re.search(r"^name:\s*(\S+)\s*$", frontmatter[1], re.MULTILINE)
            if match:
                skill_name = match[1]
    for relative, manifest in manifests.items():
        if manifest.get("name") != skill_name:
            errors.append(f"{relative}: name differs from packaged skill frontmatter")
    if portable.get("$schema") != "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json":
        errors.append("plugin.json: expected the versioned portable plugin schema")
    for relative, manifest in manifests.items():
        if manifest.get("description") != portable.get("description"):
            errors.append(f"{relative}: description differs from portable manifest")
    extensions = portable.get("extensions", {})
    settings = extensions.get("com.openai", {}) if isinstance(extensions, dict) else {}
    if not isinstance(settings, dict):
        settings = {}
    for relative, config in (("plugin.json", settings), (".codex-plugin/plugin.json", codex)):
        if config.get("requires_local_executor") is not True:
            errors.append(f"{relative}: requires_local_executor must be true")
        interface = config.get("interface", {})
        if not isinstance(interface, dict):
            interface = {}
        for key in ("displayName", "shortDescription", "longDescription", "category"):
            if not isinstance(interface.get(key), str) or not interface[key].strip():
                errors.append(f"{relative}: interface.{key} must be a nonempty string")
    if codex.get("skills") != "./skills/" or not skill.is_file():
        errors.append(".codex-plugin/plugin.json: skills must resolve to the packaged skill")
    for relative in (".agents/plugins/marketplace.json", ".claude-plugin/marketplace.json"):
        try:
            catalog = json.loads((root / relative).read_text())
            entries = catalog.get("plugins", [])
            if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
                raise ValueError("plugins must be a list of objects")
            if not isinstance(catalog.get("name"), str) or len(entries) != 1:
                raise ValueError("expected a named catalog with one plugin")
            entry = entries[0]
            source = entry.get("source")
            if isinstance(source, dict) and source.get("source") != "local":
                raise ValueError("expected a local plugin source")
            path = source.get("path") if isinstance(source, dict) else source
            if entry.get("name") != frontend_name or path != "./":
                raise ValueError("plugin must resolve to this repository")
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            errors.append(f"{relative}: {exc}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    errors = check_links(args.root) + check_manifests(args.root)
    if errors:
        parser.exit(1, "\n".join(errors) + "\n")
    print("Documentation links and plugin manifest contracts passed.")


if __name__ == "__main__":
    main()
