"""The plugin's bin/oumi-weights-lab runs the CLI from the plugin's own copy."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PATH = "/usr/bin:/bin"


def plugin_copy(tmp_path):
    """Lay out an installed plugin the way Claude Code caches it: the whole repository."""
    plugin = tmp_path / "plugin cache" / "oumi-weights-lab"
    (plugin / "bin").mkdir(parents=True)
    shutil.copy2(ROOT / "bin/oumi-weights-lab", plugin / "bin/oumi-weights-lab")
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(ROOT / name, plugin / name)
    return plugin


def recording_command(directory, name, log):
    """A stand-in command that records its arguments and the uv settings it saw."""
    directory.mkdir(parents=True, exist_ok=True)
    command = directory / name
    command.write_text(
        f'#!/bin/sh\nprintf "%s\\n" "$0" "$@" > "{log}"\n'
        f'env | grep -E "^(UV_|VIRTUAL_ENV)" > "{log}.env" || true\n'
    )
    command.chmod(0o755)
    return command


def run(plugin, path, *args, **env):
    # A timeout turns a wrapper loop into a failure instead of a hung suite.
    return subprocess.run(
        [str(plugin / "bin/oumi-weights-lab"), *args],
        capture_output=True,
        text=True,
        env={"PATH": path, "HOME": os.environ.get("HOME", "/tmp"), **env},
        check=False,
        timeout=10,
    )


def test_wrapper_is_executable_and_portable_shell():
    wrapper = ROOT / "bin/oumi-weights-lab"
    assert os.access(wrapper, os.X_OK)
    assert wrapper.read_text().startswith("#!/bin/sh\n")


def test_wrapper_runs_the_plugin_copy_with_its_lockfile(tmp_path):
    plugin = plugin_copy(tmp_path)
    log = tmp_path / "uv-args"
    tools = tmp_path / "tools"
    recording_command(tools, "uv", log)
    path = f"{plugin / 'bin'}:{tools}:{SYSTEM_PATH}"

    first = run(plugin, path, "send", "--session", "s p", "--code", "print(1)")
    assert first.returncode == 0
    assert "first run downloads PyTorch" in first.stderr
    assert first.stdout == ""  # Setup notices never reach the JSON stdout.
    expected = [
        "run",
        "--frozen",
        "--no-dev",
        "--extra",
        "adapters",
        "--extra",
        "datasets",
        "--extra",
        "multi-gpu",
    ]
    expected += ["--project"]
    expected += [str(plugin.resolve()), "python", "-m", "oumi_weights_lab"]
    expected += ["send", "--session", "s p", "--code", "print(1)"]
    assert log.read_text().splitlines()[1:] == expected

    (plugin / ".venv").mkdir()
    again = run(plugin, path, "peek", "--session", "s")
    assert again.returncode == 0 and again.stderr == ""
    assert log.read_text().splitlines()[1:3] == ["run", "--quiet"]


def test_wrapper_without_uv_defers_to_another_installed_cli(tmp_path):
    plugin = plugin_copy(tmp_path)
    log = tmp_path / "cli-args"
    installed = recording_command(tmp_path / "installed", "oumi-weights-lab", log)
    # The plugin's own bin/ comes first and must be skipped, not re-entered.
    path = f"{plugin / 'bin'}:{installed.parent}:{SYSTEM_PATH}"
    result = run(plugin, path, "stop", "--session", "s")
    assert result.returncode == 0
    assert log.read_text().splitlines() == [str(installed), "stop", "--session", "s"]


@pytest.mark.parametrize("missing", ["uv", "uv.lock"])
def test_wrapper_explains_how_to_install_when_nothing_can_run(tmp_path, missing):
    plugin = plugin_copy(tmp_path)
    path = f"{plugin / 'bin'}:{SYSTEM_PATH}"
    if missing == "uv.lock":
        (plugin / "uv.lock").unlink()
        tools = tmp_path / "tools"
        recording_command(tools, "uv", tmp_path / "unused")
        path = f"{plugin / 'bin'}:{tools}:{SYSTEM_PATH}"
    result = run(plugin, path, "--help")
    assert result.returncode == 127
    assert "uv tool install git+https://github.com/oumi-ai/weights-lab" in result.stderr


def test_wrapper_ignores_uv_settings_that_would_redirect_or_skip_its_environment(tmp_path):
    plugin = plugin_copy(tmp_path)
    log = tmp_path / "uv-args"
    tools = tmp_path / "tools"
    recording_command(tools, "uv", log)
    result = run(
        plugin,
        f"{plugin / 'bin'}:{tools}:{SYSTEM_PATH}",
        "--help",
        UV_PROJECT_ENVIRONMENT=str(tmp_path / "user-env"),
        UV_NO_SYNC="1",
        UV_NO_EDITABLE="1",
        VIRTUAL_ENV=str(tmp_path / "user-env"),
    )
    assert result.returncode == 0
    assert Path(f"{log}.env").read_text() == ""


@pytest.mark.parametrize("second", ["copy", "symlink"])
def test_wrapper_copies_without_uv_never_defer_to_each_other_forever(tmp_path, second):
    plugin = plugin_copy(tmp_path)
    other = tmp_path / "other-bin"
    other.mkdir()
    if second == "copy":
        shutil.copy2(plugin / "bin/oumi-weights-lab", other / "oumi-weights-lab")
    else:
        (other / "oumi-weights-lab").symlink_to(plugin / "bin/oumi-weights-lab")
    result = run(plugin, f"{plugin / 'bin'}:{other}:{SYSTEM_PATH}", "--help")
    assert result.returncode == 127
    assert "uv tool install" in result.stderr
