"""`oumi-weights-lab script FILE`: run a Python file in the lab's own environment."""

from oumi_weights_lab import cli


def test_script_runs_the_file_as_main_with_its_arguments(tmp_path, capsys):
    path = tmp_path / "analysis.py"
    path.write_text(
        "import sys\n"
        "import oumi_weights_lab\n"
        "assert __name__ == '__main__'\n"
        "print('args', sys.argv[1:], callable(oumi_weights_lab.open_lab))\n"
    )
    assert cli.main(["script", str(path), "--layer", "3"]) == 0
    assert "args ['--layer', '3'] True" in capsys.readouterr().out


def test_script_error_prints_its_traceback_and_fails(tmp_path, capsys):
    path = tmp_path / "broken.py"
    path.write_text("def f():\n    raise ValueError('bad input')\nf()\n")
    assert cli.main(["script", str(path)]) == 1
    err = capsys.readouterr().err
    assert "Traceback" in err and "broken.py" in err and "ValueError: bad input" in err


def test_script_exit_code_passes_through(tmp_path):
    path = tmp_path / "exits.py"
    path.write_text("import sys\nsys.exit(3)\n")
    assert cli.main(["script", str(path)]) == 3


def test_script_refuses_a_missing_file(tmp_path, capsys):
    assert cli.main(["script", str(tmp_path / "missing.py")]) == 1
    assert "missing.py" in capsys.readouterr().err


def test_script_can_import_a_module_next_to_it_and_restores_sys_path(tmp_path, capsys):
    import sys

    (tmp_path / "helper_mod_for_script.py").write_text("VALUE = 42\n")
    path = tmp_path / "analysis.py"
    path.write_text("import helper_mod_for_script\nprint('value', helper_mod_for_script.VALUE)\n")
    before = list(sys.path)
    try:
        assert cli.main(["script", str(path)]) == 0
        assert "value 42" in capsys.readouterr().out
        assert sys.path == before
    finally:
        sys.modules.pop("helper_mod_for_script", None)


def test_script_restores_sys_path_after_an_error(tmp_path):
    import sys

    path = tmp_path / "broken.py"
    path.write_text("raise ValueError('x')\n")
    before = list(sys.path)
    assert cli.main(["script", str(path)]) == 1
    assert sys.path == before
