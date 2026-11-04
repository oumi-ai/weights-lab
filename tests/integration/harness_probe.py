"""Container-side assertions; never run plugin installation on the host."""

import json
import os
import selectors
import subprocess
import sys
import time
from pathlib import Path

PLUGIN = "oumi-weights-lab@oumi-weights-lab"
RECIPE = "references/worked-example.md"


def run(*args, timeout=30, input=None):
    result = subprocess.run(
        args, input=input, capture_output=True, text=True, timeout=timeout, check=False
    )
    if result.returncode:
        raise AssertionError(f"{args[0]} {args[1:3]} failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout


def objects(value):
    """Walk CLI response objects without depending on unrelated inventory fields."""
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from objects(child)


def codex_skill_inventory():
    # Ask the actual app server's skill resolver, without a model request.
    process = subprocess.Popen(
        ["codex", "app-server", "--stdio"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    buffered = b""

    def request(identifier, method, params):
        nonlocal buffered
        message = json.dumps({"id": identifier, "method": method, "params": params}) + "\n"
        process.stdin.write(message.encode())
        process.stdin.flush()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if b"\n" not in buffered:
                if not selector.select(timeout=max(0, deadline - time.monotonic())):
                    break
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    raise AssertionError("Codex app server exited before inventory response")
                buffered += chunk
                continue
            line, buffered = buffered.split(b"\n", 1)
            response = json.loads(line)
            if response.get("id") == identifier:
                assert "error" not in response, response
                return response["result"]
        raise AssertionError(f"Codex app server timed out during {method}")

    try:
        request(1, "initialize", {"clientInfo": {"name": "oumi-weights-lab-test", "version": "1"}})
        process.stdin.write(b'{"method":"initialized","params":{}}\n')
        process.stdin.flush()
        inventory = request(2, "skills/list", {"cwds": ["/work"], "forceReload": True})
        skills = [
            item
            for item in objects(inventory)
            if item.get("name") == "oumi-weights-lab:oumi-weights-lab"
            and item.get("pluginId") == PLUGIN
        ]
        assert len(skills) == 1, f"expected one plugin skill, found {len(skills)}: {inventory}"
        assert skills[0]["enabled"] is True, skills[0]
        path = Path(skills[0]["path"])
        assert path.name == "SKILL.md"
        assert (path.parent / RECIPE).is_file(), "installed plugin lost its bundled recipe"
        return path
    finally:
        selector.close()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def install_and_discover(harness):
    version = run(harness, "--version").strip()
    run(harness, "plugin", "marketplace", "add", "/fixture")
    if harness == "codex":
        run("codex", "plugin", "add", PLUGIN, "--json")
        listing = json.loads(
            run("codex", "plugin", "list", "--marketplace", "oumi-weights-lab", "--json")
        )
        entries = [item for item in listing["installed"] if item.get("name") == "oumi-weights-lab"]
        assert len(entries) == 1 and entries[0]["enabled"] is True, listing
        skill = codex_skill_inventory()
    else:
        run("claude", "plugin", "install", PLUGIN, "--scope", "user")
        listing = json.loads(run("claude", "plugin", "list", "--json"))
        entries = [item for item in objects(listing) if item.get("id") == PLUGIN]
        assert len(entries) == 1 and entries[0]["enabled"] is True, listing
        details = run("claude", "plugin", "details", PLUGIN)
        assert "oumi-weights-lab" in details and "skill" in details.lower(), details
        skill = None
    print(
        json.dumps(
            {
                "harness": version,
                "installed": PLUGIN,
                "discovered_skill": "oumi-weights-lab:oumi-weights-lab",
            }
        )
    )
    return skill


def invoke_codex(skill):
    run("codex", "login", "--with-api-key", input=os.environ["OPENAI_API_KEY"])
    prompt = (
        "$oumi-weights-lab:oumi-weights-lab Use the installed oumi-weights-lab plugin skill. "
        "Read its SKILL.md and bundled references/worked-example.md, then execute exactly "
        "oumi-weights-lab --help once. Do not start a worker, download a model, or change files. "
        "Report the CLI subcommands you observed and stop."
    )
    args = [
        "codex",
        "--no-daemon",
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--json",
        # Docker supplies isolation; nested Linux namespaces are often unavailable.
        "--dangerously-bypass-approvals-and-sandbox",
        "-c",
        'model_reasoning_effort="low"',
        "-c",
        'shell_environment_policy.inherit="all"',
    ]
    if os.environ.get("OUMI_WEIGHTS_LAB_CODEX_MODEL"):
        args.extend(["--model", os.environ["OUMI_WEIGHTS_LAB_CODEX_MODEL"]])
    output = run(*args, prompt, timeout=120)
    events = [json.loads(line) for line in output.splitlines() if line.strip()]
    commands = [
        item
        for event in events
        if event.get("type") == "item.completed"
        for item in [event.get("item", {})]
        if item.get("type") == "command_execution"
    ]
    successful = [item for item in commands if item.get("exit_code") == 0]
    proof = json.dumps(
        [
            {
                "command": item.get("command"),
                "exit_code": item.get("exit_code"),
                "output": item.get("aggregated_output", "")[:300],
            }
            for item in commands
        ]
    )
    assert any(
        str(skill) in item.get("command", "")
        and "Keep one model loaded" in item.get("aggregated_output", "")
        for item in successful
    ), proof
    assert any(
        RECIPE in item.get("command", "") and "Worked example:" in item.get("aggregated_output", "")
        for item in successful
    ), proof
    help_runs = [
        item for item in successful if "oumi-weights-lab --help" in item.get("command", "")
    ]
    assert help_runs, proof
    assert any("usage:" in item.get("aggregated_output", "").lower() for item in help_runs), proof
    print(
        json.dumps(
            {
                "harness": "codex",
                "command_events": [
                    {
                        "command": item["command"],
                        "exit_code": item["exit_code"],
                        "skill_text_observed": "Keep one model loaded" in item["aggregated_output"],
                        "recipe_text_observed": "Worked example:" in item["aggregated_output"],
                        "output_start": item["aggregated_output"][:300],
                    }
                    for item in successful
                ],
            }
        )
    )


def invoke_claude():
    prompt = (
        "Use the Skill tool to invoke oumi-weights-lab:oumi-weights-lab. "
        "Read its bundled references/worked-example.md with Read, then use Bash to execute "
        "exactly oumi-weights-lab --help once. "
        "Do not start a worker, download a model, or change files. "
        "Report the CLI subcommands you observed and stop."
    )
    args = [
        "claude",
        "--print",
        "--verbose",
        "--output-format",
        "stream-json",
        "--no-session-persistence",
        "--strict-mcp-config",
        "--permission-mode",
        "dontAsk",
        "--tools",
        "Skill,Read,Bash",
        "--allowedTools",
        "Skill",
        "Read",
        "Bash(oumi-weights-lab --help)",
        "--max-budget-usd",
        "0.50",
    ]
    if os.environ.get("OUMI_WEIGHTS_LAB_CLAUDE_MODEL"):
        args.extend(["--model", os.environ["OUMI_WEIGHTS_LAB_CLAUDE_MODEL"]])
    output = run(*args, "--", prompt, timeout=120)
    events = [json.loads(line) for line in output.splitlines() if line.strip()]
    init = next(
        event
        for event in events
        if event.get("type") == "system" and event.get("subtype") == "init"
    )
    visible = init.get("slash_commands", []) + init.get("skills", [])
    assert "oumi-weights-lab:oumi-weights-lab" in visible, output
    uses = [item for event in events for item in objects(event) if item.get("type") == "tool_use"]
    assert any(
        item.get("name") == "Skill"
        and item.get("input", {}).get("skill") == "oumi-weights-lab:oumi-weights-lab"
        for item in uses
    ), output
    recipe_ids = {
        item["id"]
        for item in uses
        if item.get("name") == "Read" and RECIPE in item.get("input", {}).get("file_path", "")
    }
    assert recipe_ids, output
    assert any(
        item.get("type") == "tool_result"
        and item.get("tool_use_id") in recipe_ids
        and not item.get("is_error")
        and "Worked example:" in str(item.get("content", ""))
        for event in events
        for item in objects(event)
    ), output
    help_ids = {
        item["id"]
        for item in uses
        if item.get("name") == "Bash"
        and item.get("input", {}).get("command") == "oumi-weights-lab --help"
    }
    assert help_ids, output
    assert any(
        item.get("type") == "tool_result"
        and item.get("tool_use_id") in help_ids
        and not item.get("is_error")
        and "usage:" in str(item.get("content", "")).lower()
        for event in events
        for item in objects(event)
    ), output
    assert any(event.get("type") == "result" and not event.get("is_error") for event in events), (
        output
    )
    print(
        json.dumps(
            {
                "harness": "claude",
                "tool_events": [
                    {"id": item["id"], "name": item["name"], "input": item["input"]}
                    for item in uses
                    if item.get("name") in {"Skill", "Read", "Bash"}
                ],
                "tool_results": [
                    {
                        "tool_use_id": item["tool_use_id"],
                        "is_error": item.get("is_error", False),
                        "output_start": str(item.get("content", ""))[:300],
                    }
                    for event in events
                    for item in objects(event)
                    if item.get("type") == "tool_result"
                    and item.get("tool_use_id") in recipe_ids | help_ids
                ],
            }
        )
    )


def main():
    harness, action = sys.argv[1:]
    # This helper is mounted under /fixture only by the disposable-container runner.
    assert Path("/fixture/probe.py").is_file(), "run through test_harness_plugins.py"
    skill = install_and_discover(harness)
    if action == "invoke":
        if harness == "codex":
            invoke_codex(skill)
        else:
            invoke_claude()
        print(
            json.dumps(
                {
                    "harness": harness,
                    "invoked_skill": True,
                    "recipe_read": RECIPE,
                    "command": "oumi-weights-lab --help",
                    "success": True,
                }
            )
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        message = str(error)
        for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            if os.environ.get(key):
                message = message.replace(os.environ[key], "[REDACTED]")
        print(message[-12000:], file=sys.stderr)
        sys.exit(1)
