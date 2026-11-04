"""Opt-in tests against real agent CLIs in disposable containers."""

import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.integration


@pytest.fixture
def harness_image():
    image = os.environ.get("OUMI_WEIGHTS_LAB_HARNESS_IMAGE")
    if not image:
        pytest.skip("set OUMI_WEIGHTS_LAB_HARNESS_IMAGE; see CONTRIBUTING.md")
    if not shutil.which("docker"):
        pytest.skip("Docker CLI is required for isolated harness integration")
    for args in (["info"], ["image", "inspect", image]):
        result = subprocess.run(
            ["docker", *args], capture_output=True, text=True, timeout=15, check=False
        )
        if result.returncode:
            pytest.skip("Docker daemon and a prebuilt local harness image are required")
    return image


@pytest.fixture
def plugin_fixture(tmp_path):
    # Deliberately exclude the host's .git, config, credentials, and experiment data.
    tmp_path.chmod(0o755)  # The image's unprivileged user must traverse the bind mount.
    for name in (".agents/plugins", ".codex-plugin", ".claude-plugin", "skills"):
        shutil.copytree(ROOT / name, tmp_path / name)
    runtime = tmp_path / "src" / "oumi_weights_lab"
    runtime.mkdir(parents=True)
    for source in (ROOT / "src" / "oumi_weights_lab").glob("*.py"):
        shutil.copyfile(source, runtime / source.name)
    shutil.copyfile(ROOT / "plugin.json", tmp_path / "plugin.json")
    shutil.copyfile(Path(__file__).with_name("harness_probe.py"), tmp_path / "probe.py")
    return tmp_path


def run_probe(image, plugin, harness, action):
    name = f"oumi-weights-lab-harness-{uuid.uuid4().hex}"
    args = [
        "docker",
        "run",
        "--rm",
        "--init",
        "--name",
        name,
        "--pull",
        "never",
        "--network",
        "bridge" if action == "invoke" else "none",
        "--mount",
        f"type=bind,source={plugin},target=/fixture,readonly",
        "--workdir",
        "/work",
        "--env",
        "PYTHONPATH=/fixture/src",
    ]
    if action == "invoke":
        # Pass environment variable names, never credential values, in argv.
        key = "OPENAI_API_KEY" if harness == "codex" else "ANTHROPIC_API_KEY"
        args.extend(["--env", key])
        model = f"OUMI_WEIGHTS_LAB_{harness.upper()}_MODEL"
        if os.environ.get(model):
            args.extend(["--env", model])
    args.extend(["--entrypoint", "python3", image, "/fixture/probe.py", harness, action])
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=240, check=False)
        output = result.stdout + result.stderr
        for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            if os.environ.get(key):
                output = output.replace(os.environ[key], "[REDACTED]")
        assert result.returncode == 0, output[-12000:]
        print(output, end="")
    except subprocess.TimeoutExpired:
        pytest.fail(f"{harness} {action} exceeded 240 seconds")
    finally:
        # Killing a timed-out Docker client does not stop its container.
        subprocess.run(
            ["docker", "rm", "--force", name], capture_output=True, timeout=15, check=False
        )


@pytest.mark.parametrize("harness", ["claude", "codex"])
def test_install_and_discover_plugin(harness_image, plugin_fixture, harness):
    run_probe(harness_image, plugin_fixture, harness, "discover")


@pytest.mark.parametrize("harness", ["claude", "codex"])
def test_invoke_installed_plugin(harness_image, plugin_fixture, harness):
    if os.environ.get("OUMI_WEIGHTS_LAB_HARNESS_INVOKE") != "1":
        pytest.skip("set OUMI_WEIGHTS_LAB_HARNESS_INVOKE=1 to authorize real model API calls")
    key = "OPENAI_API_KEY" if harness == "codex" else "ANTHROPIC_API_KEY"
    if not os.environ.get(key):
        pytest.skip(f"{key} is required; host login/keychain credentials are never mounted")
    run_probe(harness_image, plugin_fixture, harness, "invoke")
