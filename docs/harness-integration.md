# Real agent harness integration

`tests/integration/test_harness_plugins.py` exercises the real Claude Code and
Codex CLIs. These tests are separate from the fast manifest/packaging checks in
`tests/test_plugin_packaging.py`. Ordinary pytest runs exclude integration tests.
Explicit integration runs skip these tests unless a local image is selected.

Each test starts a disposable container with a fresh operating-system user's
normal home directory. It installs the checkout's local marketplace and plugin,
then checks the harness's installed inventory and skill discovery. Codex discovery
uses the actual app server's `skills/list` resolver; Claude discovery uses
`plugin details`. The optional invocation tests additionally require the harness
to load the skill, read its bundled worked example, and execute the real
`oumi-weights-lab --help` command. Assertions inspect tool events and successful command
output; a model's claim that it ran the command is insufficient.

Both harnesses passed all four real install, discovery, and API-backed invocation
checks on November 3, 2026, using the pinned versions below and the canonical
`oumi-weights-lab` plugin identity and CLI. The installed Codex plugin's skill
resolves as `oumi-weights-lab:oumi-weights-lab`, with plugin ID
`oumi-weights-lab@oumi-weights-lab-local`; the integration prompt uses that qualified name.
The run retained bounded tool-event evidence of the recipe reads and successful
`oumi-weights-lab --help` executions.

## Prepare the image

Docker must be running. Building the image is an explicit network operation that
downloads the real agent CLIs and a small Python build environment:

```sh
docker build -f tests/integration/Dockerfile -t oumi-weights-lab-harness:local .
OUMI_WEIGHTS_LAB_HARNESS_IMAGE=oumi-weights-lab-harness:local \
  uv run pytest -o addopts= -m integration \
    tests/integration/test_harness_plugins.py -k install_and_discover -rs
```

The Dockerfile pins Codex CLI `0.166.0` and Claude Code `2.1.318`. Override its
`CODEX_VERSION` and `CLAUDE_VERSION` build arguments when testing another release.
No PyTorch installation, checkpoint download, model worker, cloud compute, or GPU
is needed for this test. Python's installed CLI entry point reads the staged
checkout's `src/` through `PYTHONPATH`, so changes to the checkout are tested even
when reusing the image.

The tests never build or pull an image themselves. Missing Docker, a stopped
daemon, or an absent local image produces an explicit prerequisite skip.
The discovery-only tests run with container networking disabled.

## Run actual model invocation

Export `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` securely in the calling shell,
then explicitly enable model requests:

```sh
OUMI_WEIGHTS_LAB_HARNESS_IMAGE=oumi-weights-lab-harness:local \
OUMI_WEIGHTS_LAB_HARNESS_INVOKE=1 \
  uv run pytest -o addopts= -m integration tests/integration/test_harness_plugins.py -rs
```

Each harness with its API key available makes one noninteractive agent invocation.
The harness can make multiple API turns while using tools. Claude has a $0.50
budget; model execution has a 120-second deadline and each container has a
240-second outer deadline. Optional `OUMI_WEIGHTS_LAB_CODEX_MODEL` and
`OUMI_WEIGHTS_LAB_CLAUDE_MODEL` select an accessible model; otherwise each fresh CLI
uses its default. A missing key or invocation opt-in skips that test explicitly.
Authentication errors, unavailable models, skill-loading failures, denied tools,
and missing command evidence fail an enabled test.
Add `-s` to pytest to show concise harness version and phase summaries.

The runner only forwards the selected harness's API-key environment variable.
It mounts a read-only staged copy of plugin metadata, skills, and runtime source;
host home directories, auth files, global settings, Git metadata, and experiment
data are excluded. Neither `HOME` nor `CODEX_HOME` is changed. Plugin installation
and API-key login write only inside the container, which is forcibly removed even
after a timeout. Existing host login/keychain state is never modified or copied.
Codex runs with its inner sandbox disabled inside this disposable container, so
the test does not require Docker to allow nested sandbox namespaces.

The install and invocation commands follow the official
[Codex plugin command reference](https://learn.chatgpt.com/docs/developer-commands?surface=cli),
[OpenAI plugin packaging guide](https://developers.openai.com/plugins/build/plugins),
and [Claude Code plugin guide](https://code.claude.com/docs/en/plugins).
CLI upgrades can change event schemas or permission behavior; a successful
manifest validation alone does not establish install/discovery/invocation parity.
