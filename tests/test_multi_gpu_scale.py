"""Multi-GPU sessions: a longer shutdown grace, cluster disk size, and parallel hashing."""

from types import SimpleNamespace

import pytest

from oumi_weights_lab import cli, persistence, session
from oumi_weights_lab.launchers import SkyPilotLauncher


def test_device_map_sessions_default_to_a_longer_shutdown_grace(tmp_path):
    # Hashing a large model for the deadline bundle takes longer than 30 seconds.
    split = session.create_session(tmp_path / "split", {"model": "m", "device_map": "balanced"})
    single = session.create_session(tmp_path / "single", {"model": "m"})
    chosen = session.create_session(
        tmp_path / "chosen", {"model": "m", "device_map": "balanced", "shutdown_grace": 45}
    )
    grace = [
        session.read_json(path / "config.json")["shutdown_grace"]
        for path in (split, single, chosen)
    ]
    assert grace == [session.MULTI_GPU_SHUTDOWN_GRACE, session.DEFAULT_SHUTDOWN_GRACE, 45.0]
    assert session.MULTI_GPU_SHUTDOWN_GRACE == 300.0


def test_start_leaves_shutdown_grace_unset_unless_given():
    start = ["start", "--session", "s", "--model", "m"]
    assert cli._parser().parse_args(start).shutdown_grace is None
    assert cli._parser().parse_args([*start, "--shutdown-grace", "60"]).shutdown_grace == 60


class Recorder:
    def __init__(self):
        self.task = None

    def __call__(self, command, **kwargs):
        import json

        if command[:2] == ["sky", "launch"]:
            with open(command[-1]) as stream:
                self.task = json.load(stream)
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")


def test_cluster_disk_size_reaches_the_sky_task(tmp_path):
    runner = Recorder()
    launcher = SkyPilotLauncher("weights-test", runner=runner)
    launcher.start(
        "model", "/tmp/session", accelerators="H100:4", disk_size=1024, device_map="balanced"
    )
    assert runner.task["resources"] == {"accelerators": "H100:4", "disk_size": 1024}
    launcher.start("model", "/tmp/session", accelerators="H100:4")
    assert runner.task["resources"] == {"accelerators": "H100:4"}


@pytest.mark.parametrize("size", [0, -5, 1.5, True, "512"])
def test_cluster_disk_size_must_be_a_positive_whole_number(size):
    runner = Recorder()
    with pytest.raises(ValueError, match="disk_size"):
        SkyPilotLauncher("weights-test", runner=runner).start("model", "/tmp/s", disk_size=size)
    assert runner.task is None


def test_disk_size_needs_a_cluster(tmp_path, capsys):
    arguments = ["start", "--session", str(tmp_path / "s"), "--model", "m", "--disk-size", "512"]
    assert cli.main(arguments) != 0
    assert "--disk-size requires --cluster" in capsys.readouterr().err
    assert not (tmp_path / "s").exists()


def test_hashing_reads_every_device_in_turn():
    def tensor(device):
        return SimpleNamespace(device=device)

    targets = {
        "a": tensor("cuda:0"),
        "b": tensor("cuda:0"),
        "c": tensor("cuda:0"),
        "d": tensor("cuda:1"),
        "e": tensor("cuda:1"),
        "f": tensor("cuda:2"),
    }
    assert persistence._interleaved(targets) == ["a", "d", "f", "b", "e", "c"]


def test_hashing_uses_more_threads_without_changing_the_fingerprint(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from oumi_weights_lab import Lab

    model = torch.nn.Sequential(*(torch.nn.Linear(4, 4) for _ in range(40)))
    lab = Lab(model, SimpleNamespace(), tmp_path / "lab", model_id="tiny")
    workers = []
    real = persistence.ThreadPoolExecutor

    def recording(max_workers):
        workers.append(max_workers)
        return real(max_workers=max_workers)

    monkeypatch.setattr(persistence, "ThreadPoolExecutor", recording)
    monkeypatch.setattr(persistence.os, "cpu_count", lambda: 64)
    parallel = persistence._fingerprint(lab)
    monkeypatch.setattr(persistence.os, "cpu_count", lambda: 1)
    assert persistence._fingerprint(lab) == parallel
    assert workers == [32, 1]


def test_tensor_digests_match_the_ones_existing_bundles_recorded():
    # Bundles store a fingerprint built from these digests; changing them would stop
    # every saved bundle from resuming.
    torch = pytest.importorskip("torch")
    chunked = torch.arange(600_000, dtype=torch.float32).reshape(600, 1000) / 7
    small = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    assert persistence._digest(chunked) == (
        "241dbefb50eb321f27742e5d8131fe16d45fff714d87dcd7dd3e4c9ac9371469"
    )
    assert persistence._digest(small) == (
        "99cee2668dea124ff17c6cf7732f1e8ed98c437396ad8805367c0dc861be2869"
    )


def _stopping_worker(path, seconds):
    """A stand-in worker that ignores SIGTERM and finishes after ``seconds``.

    It replaces its status file the way the real worker does, so stop never reads it
    half-written while polling.
    """
    import subprocess
    import sys

    status = path / "status.json"
    code = (
        "import json, os, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"time.sleep({seconds})\n"
        f"with open({str(status) + '.tmp'!r}, 'w') as stream:\n"
        "    json.dump({'state': 'stopped', 'reason': 'idle_timeout', 'edit_bundle': 'saved'}, "
        "stream)\n"
        f"os.replace({str(status) + '.tmp'!r}, {str(status)!r})\n"
    )
    return subprocess.Popen([sys.executable, "-c", code])


@pytest.mark.parametrize(("began", "forced"), [(0, False), (-60, True)])
def test_stop_waits_out_a_deadline_bundle_within_the_shutdown_grace(
    began, forced, tmp_path, capsys, monkeypatch
):
    import json
    import time

    path = session.create_session(tmp_path / "s", {"model": "m", "shutdown_grace": 20})
    worker = _stopping_worker(path, 4 if not forced else 30)
    # stop signals only its own worker; how ps shows a command line varies by platform.
    monkeypatch.setattr(cli, "_matches_worker", lambda pid, _: pid == worker.pid)
    try:
        session.write_json(
            path / "status.json",
            {
                "state": "stopping",
                "pid": worker.pid,
                "reason": "idle_timeout",
                "updated_at": time.time() + began,
            },
        )
        assert cli.main(["stop", "--session", str(path), "--timeout", "0.2"]) == 0
        status = json.loads(capsys.readouterr().out)
        assert status["state"] == "stopped"
        # Within the grace the bundle finishes; past it, stop escalates as before.
        assert bool(status.get("forced")) is forced
        assert ("edit_bundle" in status) is not forced
    finally:
        worker.kill()
        worker.wait(5)


def test_parallel_hashing_on_gpus_matches_hashing_on_cpu(tmp_path, tiny_tokenizer, monkeypatch):
    torch = pytest.importorskip("torch")
    accelerate = pytest.importorskip("accelerate")
    transformers = pytest.importorskip("transformers")
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA GPUs")
    import copy

    from oumi_weights_lab import Lab

    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(vocab_size=6, n_embd=64, n_layer=4, n_head=4, n_positions=32)
    ).eval()
    reference = Lab(copy.deepcopy(model), tiny_tokenizer, tmp_path / "cpu")
    placement = {
        "transformer.wte": 0,
        "transformer.wpe": 0,
        "transformer.drop": 0,
        "transformer.h.0": 0,
        "transformer.h.1": 0,
        "transformer.h.2": 1,
        "transformer.h.3": 1,
        "transformer.ln_f": 1,
        "lm_head": 0,
    }
    split = Lab(
        accelerate.dispatch_model(model, device_map=placement), tiny_tokenizer, tmp_path / "gpus"
    )
    parallel = persistence._fingerprint(split)
    monkeypatch.setattr(persistence.os, "cpu_count", lambda: 1)
    assert persistence._fingerprint(split) == parallel == persistence._fingerprint(reference)
