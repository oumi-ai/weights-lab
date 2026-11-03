import subprocess
from unittest.mock import Mock

import pytest

from weights_lab.launchers import SlurmLauncher
from weights_lab.session import create_session, write_json


@pytest.mark.parametrize(
    "scheduler_output,expected_active",
    [
        ("RUNNING\n", True),
        ("PENDING\n", True),
        ("COMPLETING\n", True),
        ("COMPLETED\n", False),
        ("CANCELLED\n", False),
        ("", False),
        ("CLUSTER: alpha\nCOMPLETED\n", False),
        ("CLUSTER: alpha\n", False),
        ("CLUSTER: alpha\nRUNNING\n", True),
    ],
)
def test_slurm_activity_uses_scheduler_states_and_ignores_cluster_headers(
    scheduler_output, expected_active, tmp_path, monkeypatch
):
    path = create_session(tmp_path / "session", {"model": "unused"})
    write_json(path / "launcher.json", {"launcher": "slurm", "job_id": "42", "cluster": "alpha"})
    runner = Mock(return_value=subprocess.CompletedProcess(["squeue"], 0, scheduler_output, ""))
    local_pid_check = Mock(side_effect=AssertionError("compute-node PID checked locally"))
    monkeypatch.setattr("weights_lab.session.is_process_alive", local_pid_check)

    assert SlurmLauncher(runner=runner).is_active(path) is expected_active
    args = runner.call_args.args[0]
    assert args[0] == "squeue"
    assert args[args.index("--jobs") + 1] == "42"
    assert args[args.index("--clusters") + 1] == "alpha"
    local_pid_check.assert_not_called()
