"""The model-test gate skips only proven documentation/plugin changes."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/ci_changes.py"
spec = importlib.util.spec_from_file_location("ci_changes", SCRIPT)
changes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(changes)
BASE = "a" * 40


@pytest.mark.parametrize(
    "paths,expected",
    [
        (["docs/api.md", "README.md", "CONTRIBUTING.md", "LICENSE"], False),
        (["docs/api.md", "src/oumi_weights_lab/lab.py"], True),
        (["new-area/file.md"], True),
        (
            [
                "plugin.json",
                ".claude-plugin/plugin.json",
                ".codex-plugin/x",
                ".agents/plugins/x",
                "skills/a/SKILL.md",
            ],
            False,
        ),
        (["plugin.json", "pyproject.toml", "uv.lock"], True),
        (
            [
                "tests/test_x.py",
                "scripts/x.py",
                "examples/x.py",
                ".github/workflows/ci.yml",
                ".gitignore",
            ],
            True,
        ),
    ],
)
def test_path_classification(monkeypatch, paths, expected):
    def git(*args):
        if args[0] == "diff":
            assert args == ("diff", "--no-renames", "--name-only", "-z", f"{BASE}..HEAD")
            return b"\0".join(path.encode() for path in paths) + b"\0"
        return b""

    monkeypatch.setattr(changes, "git", git)
    assert changes.needs_models(BASE) is expected


def test_missing_zero_and_invalid_base_never_call_git(monkeypatch):
    def unexpected(*args):
        pytest.fail("Untrusted base must not reach git")

    monkeypatch.setattr(changes, "git", unexpected)
    for base in ("", "0" * 40, "main", "a" * 39, "$(echo unsafe)"):
        assert changes.needs_models(base)


def test_unavailable_base_and_fetch_failure_run_models(monkeypatch):
    calls = []

    def git(*args):
        calls.append(args)
        raise subprocess.CalledProcessError(1, ["git", *args])

    monkeypatch.setattr(changes, "git", git)
    assert changes.needs_models(BASE)
    assert calls == [
        ("cat-file", "-e", f"{BASE}^{{commit}}"),
        ("fetch", "--no-tags", "--depth=1", "origin", BASE),
    ]


def test_fetch_recovers_missing_sha256_base_and_preserves_filename_bytes(monkeypatch):
    base = "b" * 64
    calls = []

    def git(*args):
        calls.append(args)
        if args[0] == "cat-file":
            raise subprocess.CalledProcessError(1, ["git", *args])
        return b"docs/with\nnewline-\xff.md\0" if args[0] == "diff" else b""

    monkeypatch.setattr(changes, "git", git)
    assert not changes.needs_models(base)
    assert calls[1] == ("fetch", "--no-tags", "--depth=1", "origin", base)


def test_git_uses_argv_capture_and_timeout(monkeypatch):
    def run(argv, **kwargs):
        assert argv == ["git", "fetch", "--no-tags", "--depth=1", "origin", BASE]
        assert kwargs == {"check": True, "capture_output": True, "timeout": 30}
        return subprocess.CompletedProcess(argv, 0, stdout=b"")

    monkeypatch.setattr(changes.subprocess, "run", run)
    assert changes.git("fetch", "--no-tags", "--depth=1", "origin", BASE) == b""


def test_git_timeout_and_os_error_run_models(monkeypatch):
    for error in (subprocess.TimeoutExpired("git", 30), OSError("git unavailable")):

        def git(*args, error=error):
            raise error

        monkeypatch.setattr(changes, "git", git)
        assert changes.needs_models(BASE)


def test_deleted_core_and_rename_to_docs_run_models(tmp_path):
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    def commit():
        git("add", ".")
        git(
            "-c", "user.name=CI Test", "-c", "user.email=ci@example.invalid", "commit", "-m", "test"
        )

    def decision(base):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--base", base],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        assert result.stderr == ""
        return result.stdout

    git("init", "--quiet")
    (tmp_path / "src").mkdir()
    (tmp_path / "src/model.py").write_text("model = 1\n")
    commit()
    base = git("rev-parse", "HEAD")
    (tmp_path / "docs").mkdir()
    git("mv", "src/model.py", "docs/model.md")
    commit()
    assert decision(base) == "run=true\n"
    git("mv", "docs/model.md", "src/model.py")
    commit()
    base = git("rev-parse", "HEAD")
    git("rm", "src/model.py")
    commit()
    assert decision(base) == "run=true\n"
    assert decision("") == "run=true\n"
