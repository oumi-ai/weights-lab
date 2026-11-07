"""Check with Docker that the agent's container reaches its API host and nothing else.

Needs Docker running and an agent image built (see docs/proving-ground.md). Run

    python tests/live/validate_proving_network.py --image weights-lab-proving:control

It makes the network ``proving-ground run`` makes, with the runner's own code: an internal
Docker network and the filtering proxy beside it, from ``--proxy-image`` (default:
``--image``). In a container started with the runner's ``docker run`` command line it
checks that

- a Hugging Face download of GPT-2 small's config fails, and fails within seconds;
- a direct connection that skips the proxy fails, to Hugging Face and to the API host alike;
- a plain HTTP request through the proxy is refused;
- the API host, ``api.anthropic.com``, answers through the proxy with an HTTP status (any
  status: no key is sent).

It prints each check and the proxy's log, and exits non-zero when a check fails. Nothing
is pushed or kept: the container, the proxy and the network are removed afterwards.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from oumi_weights_lab.proving.runner import docker_argv, docker_egress

FAST = 30.0  # seconds within which a blocked request must fail

PROBE = r"""
import json, os, socket, time, urllib.error, urllib.request

os.environ["HF_HUB_OFFLINE"] = "0"  # the image sets it; the network has to do the blocking
results = {}


def check(name, run):
    start = time.monotonic()
    try:
        results[name] = {"reached": True, "answer": run()}
    except Exception as error:
        results[name] = {"reached": False, "error": f"{type(error).__name__}: {error}"[:300]}
    results[name]["seconds"] = round(time.monotonic() - start, 2)


def hugging_face():
    from huggingface_hub import hf_hub_download

    return hf_hub_download("openai-community/gpt2", "config.json", cache_dir="/tmp/hf")


def status(url):
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def direct(host):
    socket.create_connection((host, 443), timeout=10).close()
    return "connected"


check("hugging_face_download", hugging_face)
check("direct_hugging_face", lambda: direct("huggingface.co"))
check("direct_api_host", lambda: direct("api.anthropic.com"))
check("plain_http_through_proxy", lambda: status("http://huggingface.co/"))
check("api_host_through_proxy", lambda: status("https://api.anthropic.com/v1/messages"))
print(json.dumps(results))
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--image", required=True, help="an agent image (either arm)")
    parser.add_argument("--proxy-image", help="image the proxy runs from (default: --image)")
    args = parser.parse_args()
    name = f"wl-proving-check-{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory(prefix="proving-network-") as scratch:
        task_dir, out_dir = Path(scratch) / "task", Path(scratch) / "out"
        task_dir.mkdir()
        out_dir.mkdir()
        with docker_egress(name, args.proxy_image or args.image, ("api.anthropic.com",)) as route:
            argv, _ = docker_argv(
                name=f"{name}-agent",
                image=args.image,
                task_dir=task_dir,
                out_dir=out_dir,
                command=["python", "-c", PROBE],
                prompt="unused",
                egress=route,
            )
            done = subprocess.run(argv, capture_output=True, text=True, timeout=300, check=False)
            proxy_log = subprocess.run(
                ["docker", "logs", f"{name}-proxy"], capture_output=True, text=True, check=False
            )
    print("agent container argv (without the probe):", " ".join(argv[: argv.index("-c")]))
    if done.returncode:
        print(done.stdout, done.stderr, sep="\n")
        return 1
    results = json.loads(done.stdout.strip().splitlines()[-1])
    for check, result in results.items():
        print(f"{check}: {json.dumps(result)}")
    print("proxy log:", (proxy_log.stdout + proxy_log.stderr).strip().replace("\n", " | "))
    blocked = ("hugging_face_download", "direct_hugging_face", "direct_api_host")
    failures = [c for c in blocked if results[c]["reached"] or results[c]["seconds"] > FAST]
    plain = results["plain_http_through_proxy"]
    if plain["reached"] and plain["answer"] != 405:
        failures.append("plain_http_through_proxy")
    api = results["api_host_through_proxy"]
    if not (api["reached"] and isinstance(api["answer"], int)):
        failures.append("api_host_through_proxy")
    print("FAILED: " + ", ".join(failures) if failures else "all network checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
