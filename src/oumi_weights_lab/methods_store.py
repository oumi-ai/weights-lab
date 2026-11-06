"""Saved methods: functions an agent wrote, kept with their source, results and setup.

A method is a plain function whose first parameter is ``lab``. ``lab.save_method`` stores
it under ``methods/<name>/<version>/``; a later session loads it by name and runs it. The
version is a hash of the saved source, so a result says exactly which code produced it.
Importing this module does not import torch or transformers.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import symtable
import tempfile
import textwrap
import types
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

METHODS_ENV = "OUMI_WEIGHTS_LAB_METHODS"
NAME_PATTERN = re.compile(r"[a-z0-9_-]{1,64}")
NO_DOCSTRING = "(The method has no docstring. Say what it finds and when to use it.)"
VERSION_PATTERN = re.compile(r"[0-9a-f]{12}")


def methods_dir(directory: str | Path | None = None) -> Path:
    """The methods directory: ``directory``, else the environment variable, else ``./methods``."""
    if directory is not None:
        return Path(directory).expanduser()
    if os.environ.get(METHODS_ENV):
        return Path(os.environ[METHODS_ENV]).expanduser()
    return Path.cwd() / "methods"


def parse_reference(text: str) -> tuple[str, str | None]:
    """Split ``NAME`` or ``NAME@VERSION``."""
    name, _, version = text.partition("@")
    return name, version or None


def _check_name(name: str) -> str:
    if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
        raise ValueError(
            f"method name {name!r} must be 1 to 64 lowercase letters, digits, '_' or '-'; "
            "pass name= to choose one"
        )
    return name


def _version_dir(root: Path, name: str, version: str | None) -> Path:
    """The directory of a saved version (the newest when ``version`` is None)."""
    _check_name(name)
    folder = root / name
    if not folder.is_dir():
        raise ValueError(f"no saved method named {name!r} in {root}")
    if version is not None:
        if not VERSION_PATTERN.fullmatch(version) or not (folder / version).is_dir():
            raise ValueError(f"method {name!r} has no version {version!r}")
        return folder / version
    versions = _versions(folder)
    if not versions:
        raise ValueError(f"method {name!r} has no saved versions")
    return folder / versions[0]["version"]


def _versions(folder: Path) -> list[dict[str, Any]]:
    """The metadata of each version in ``folder``, newest first."""
    found = []
    for child in folder.iterdir():
        meta = child / "meta.json"
        if VERSION_PATTERN.fullmatch(child.name) and meta.is_file():
            found.append(json.loads(meta.read_text(encoding="utf-8")))
    return sorted(found, key=lambda m: (m["created_at"], m["version"]), reverse=True)


def read_method_results(
    root: str | Path | None, name: str, version: str | None = None
) -> list[dict[str, Any]]:
    """The graded results recorded for a saved version; a missing file is empty."""
    path = _version_dir(methods_dir(root), name, version) / "results.jsonl"
    if not path.exists():
        return []
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} line {number} is not JSON: {error}") from error
    return records


def _scored(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The records with a numeric score; a grade whose score was NaN is stored as null."""
    return [
        r
        for r in records
        if isinstance(r.get("score"), dict) and isinstance(r["score"].get("score"), int | float)
    ]


def _clean(value: Any) -> Any:
    """Replace NaN and infinity with None, so a record is valid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(item) for item in value]
    return value


def append_method_result(
    root: str | Path | None,
    name: str,
    version: str,
    result: dict[str, Any],
    setup: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append a graded result to a saved version's ``results.jsonl`` and return the record.

    ``result`` holds the ``task`` id, its ``split`` and the ``score`` dict that
    ``grade_claim`` returned, and may hold the ``setup`` it ran in; ``setup`` overrides
    that. NaN and infinity are written as null. Raises ValueError for a method or version
    that is not saved, or a result missing those fields.
    """
    missing = [k for k in ("task", "split", "score") if k not in result]
    if missing or not isinstance(result["score"], dict) or "score" not in result["score"]:
        raise ValueError(
            "a method result needs a task, a split and the score dict from grade_claim"
            + (f"; missing {', '.join(missing)}" if missing else "")
        )
    directory = _version_dir(methods_dir(root), name, version)
    record = {
        "task": result["task"],
        "split": result["split"],
        "score": _clean(result["score"]),
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    setup = setup if setup is not None else result.get("setup")
    if setup is not None:
        record["setup"] = _clean(setup)
    with (directory / "results.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False, default=str) + "\n")
    return record


def summarize_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Tasks, mean score, solve rate and splits of graded results.

    Each task counts once: its score and solve rate are averaged over its records first.
    """
    per_task: dict[str, list[dict[str, Any]]] = {}
    for record in _scored(records):
        per_task.setdefault(record["task"], []).append(record["score"])
    means = [sum(float(s["score"]) for s in scores) / len(scores) for scores in per_task.values()]
    solved = [
        sum(bool(s.get("solved")) for s in scores) / len(scores) for scores in per_task.values()
    ]
    return {
        "results": len(records),
        "tasks": len(per_task),
        "mean_score": sum(means) / len(means) if means else None,
        "solve_rate": sum(solved) / len(solved) if solved else None,
        "splits": sorted({r["split"] for r in records}),
    }


def list_methods(directory: str | Path | None = None) -> list[dict[str, Any]]:
    """Each saved method with its versions (newest first), their notes and result summaries."""
    root = methods_dir(directory)
    listing = []
    for folder in sorted(root.iterdir()) if root.is_dir() else []:
        if not folder.is_dir() or not NAME_PATTERN.fullmatch(folder.name):
            continue
        versions = _versions(folder)
        if versions:
            listing.append(
                {
                    "name": folder.name,
                    "versions": [
                        {
                            "version": meta["version"],
                            "created_at": meta["created_at"],
                            "note": meta["note"],
                            "function": meta["function"],
                            "signature": meta["signature"],
                            "results": summarize_results(
                                read_method_results(root, folder.name, meta["version"])
                            ),
                        }
                        for meta in versions
                    ],
                }
            )
    return listing


# Saving


def _import_line(name: str, module: types.ModuleType) -> str:
    target = module.__name__
    if name == target:
        return f"import {target}"
    parent, _, leaf = target.rpartition(".")
    if parent and leaf == name:
        return f"from {parent} import {name}"
    return f"import {target} as {name}"


def _global_names(table: symtable.SymbolTable, defined: str | None = None) -> set[str]:
    """Names a scope and the scopes nested in it read or write as module globals.

    Names a class body binds, and the variables of lambdas and comprehensions, are local to
    their own scope and are not listed. Attribute names (``x.model``) are not names at all.
    Pass the function's name as ``defined`` for the module scope that holds its ``def``
    statement, where decorators, defaults and annotations run: there a name counts when it
    is read and not bound, and the function's own body is skipped.
    """
    names = set()
    for symbol in table.get_symbols():
        if defined is not None:
            if symbol.is_referenced() and not symbol.is_assigned():
                names.add(symbol.get_name())
        elif symbol.is_global() and (symbol.is_referenced() or symbol.is_assigned()):
            names.add(symbol.get_name())
    for child in table.get_children():
        if defined is None or child.get_name() != defined or child.get_type() != "function":
            names |= _global_names(child)
    return names


def _saved_source(function: Any) -> tuple[str, str]:
    """The self-contained source to save and the function's own name; refuses what cannot be."""
    if not inspect.isfunction(function) or function.__name__ == "<lambda>":
        raise ValueError("a method must be a plain function defined with def, not a lambda")
    first = next(iter(inspect.signature(function).parameters.values()), None)
    if (
        first is None
        or first.name != "lab"
        or first.kind not in (first.POSITIONAL_ONLY, first.POSITIONAL_OR_KEYWORD)
    ):
        raise ValueError(
            f"{function.__name__}: the first parameter must be named 'lab', "
            "as in def method(lab, ...)"
        )
    try:
        body = textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError) as error:
        raise ValueError(
            f"the source of {function.__name__} is not available ({error}); "
            "define the method in a file or a snippet and pass the function"
        ) from error
    try:
        tree = ast.parse(body)
    except SyntaxError as error:
        raise ValueError(f"the source of {function.__name__} does not parse: {error}") from error
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError(f"the source of {function.__name__} is not one plain def")
    variables = inspect.getclosurevars(function)
    known = function.__globals__
    module = symtable.symtable(body, f"<{function.__name__}>", "exec")
    used = _global_names(module, function.__name__)
    for child in module.get_children():
        if child.get_name() == function.__name__ and child.get_type() == "function":
            used |= _global_names(child)
    modules = {n for n in used if isinstance(known.get(n), types.ModuleType)}
    others = {n for n in used - modules if n in known or not hasattr(builtins, n)}
    if variables.nonlocals or others:
        parts = []
        if variables.nonlocals:
            parts.append(f"the closure variable(s) {', '.join(sorted(variables.nonlocals))}")
        if others:
            parts.append(f"the global name(s) {', '.join(sorted(others))}")
        raise ValueError(
            f"{function.__name__} uses {' and '.join(parts)}. A saved method must be "
            "self-contained: pass values as arguments, or import inside the function"
        )
    imports = sorted(_import_line(n, known[n]) for n in modules)
    source = "".join(f"{line}\n" for line in imports)
    source += ("\n\n" if imports else "") + body.rstrip("\n") + "\n"
    return source, function.__name__


def _check_defines_one_function(source: str, function_name: str) -> None:
    """Run the saved source in a fresh namespace; it must define exactly that function."""
    namespace: dict[str, Any] = {"__name__": "oumi_weights_lab_method_check"}
    try:
        exec(compile(source, "<saved method>", "exec"), namespace)
    except Exception as error:
        raise ValueError(
            f"the saved source does not run on its own ({type(error).__name__}: {error}); "
            "a method must be self-contained: pass values as arguments, or import inside "
            "the function"
        ) from error
    defined = [
        key
        for key, value in namespace.items()
        if inspect.isfunction(value) and value.__module__ == namespace["__name__"]
    ]
    if defined != [function_name]:
        raise ValueError(f"the saved source must define only {function_name}, found {defined}")


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".meta-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, default=str) + "\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _version_of(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]


class Method:
    """A saved method loaded into a lab. Calling it runs ``function(lab, *args, **kwargs)``.

    A dict result comes back as a copy with ``"saved_method": {"name", "version"}`` added,
    and is refused when it already has that key. Any other result comes back unchanged.
    """

    def __init__(self, lab: Any, meta: dict[str, Any], source: str, function: Any) -> None:
        self.lab = lab
        self.name = meta["name"]
        self.version = meta["version"]
        self.source = source
        self.meta = meta
        self.function = function

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result = self.function(self.lab, *args, **kwargs)
        if isinstance(result, dict):
            if "saved_method" in result:
                raise ValueError(
                    f"{self.name}@{self.version} returned a dict with a 'saved_method' key, "
                    "which holds the name and version; rename that key"
                )
            result = {**result, "saved_method": {"name": self.name, "version": self.version}}
        return result

    def __repr__(self) -> str:
        return f"Method({self.name}@{self.version})"


class SavedMethodsMixin:
    """``Lab`` calls that save, list, load and run methods by name."""

    def save_method(
        self, function: Any, *, name: str | None = None, note: str = ""
    ) -> dict[str, Any]:
        """Save a function as a method with its source and this session's setup.

        The function must be a plain ``def`` whose first parameter is ``lab``. The saved
        source must run on its own: a module it uses becomes an import line, and any other
        global, a closure variable, a decorator or default that needs one, or a first
        parameter not named ``lab`` is refused, naming the offenders. Pass values as
        arguments or import inside the function. ``name`` defaults to the function's name
        and holds lowercase letters, digits, ``_`` and ``-``. The version is the first 12
        hex digits of the SHA-256 of the saved source; saving the same source again
        returns the existing entry. Saving it again in a different setup adds that setup to the
        entry's ``setups`` list. Files go to ``OUMI_WEIGHTS_LAB_METHODS`` or
        ``methods/`` in the worker's directory. Returns the metadata.
        """
        name = _check_name(name if name is not None else getattr(function, "__name__", ""))
        source, function_name = _saved_source(function)
        _check_defines_one_function(source, function_name)
        version = _version_of(source)
        folder = methods_dir() / name
        target = folder / version
        setup = self._setup()
        if (target / "meta.json").is_file():
            existing = json.loads((target / "meta.json").read_text(encoding="utf-8"))
            setups = existing.setdefault("setups", [existing["setup"]])
            if setup not in setups:
                setups.append(setup)
                _write_json_atomic(target / "meta.json", existing)
            return existing
        from . import __version__

        meta = {
            "name": name,
            "version": version,
            "function": function_name,
            "signature": str(inspect.signature(function)),
            "docstring": inspect.getdoc(function),
            "note": note,
            "created_at": datetime.now(UTC).isoformat(timespec="microseconds"),
            "lab_version": __version__,
            "setup": setup,
            "setups": [setup],
        }
        folder.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".saving-", dir=folder))
        try:
            (staging / "method.py").write_bytes(source.encode("utf-8"))
            (staging / "meta.json").write_text(
                json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8"
            )
            (staging / "results.jsonl").touch()
            try:
                staging.rename(target)
            except OSError:
                if not target.is_dir():
                    raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return json.loads((target / "meta.json").read_text(encoding="utf-8"))

    def record_method_result(
        self, name: str, version: str, result: dict[str, Any]
    ) -> dict[str, Any]:
        """Record a graded result for a saved version, with this session's setup.

        ``result`` holds the ``task`` id, its ``split`` and the ``score`` dict from
        ``grade_claim``. NaN is stored as null. Raises ValueError for a method or version
        that is not saved. Returns the stored record.
        """
        return append_method_result(None, name, version, result, self._setup())

    def methods(self) -> list[dict[str, Any]]:
        """Each saved method with its versions (newest first), notes and result summaries."""
        return list_methods()

    def load_method(self, name: str, version: str | None = None) -> Method:
        """Load a saved method: the newest version unless ``version`` is named.

        Executes the saved Python in a fresh namespace with this process's permissions, as
        snippets do, so load only methods you trust. Refuses an unknown name or version
        and a source that no longer matches its version.
        """
        directory = _version_dir(methods_dir(), name, version)
        path = directory / "method.py"
        raw = path.read_bytes()
        source = raw.decode("utf-8")
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        if _version_of(source) != meta["version"]:
            raise ValueError(
                f"{path} has changed since it was saved as version {meta['version']}; "
                "save the edited function again to get a new version"
            )
        namespace: dict[str, Any] = {"__name__": f"oumi_weights_lab_method_{meta['name']}"}
        exec(compile(source, str(path), "exec"), namespace)
        return Method(self, meta, source, namespace[meta["function"]])

    def run_method(
        self, name: str, /, *args: Any, version: str | None = None, **kwargs: Any
    ) -> Any:
        """Load a saved method and call it with ``args`` and ``kwargs`` in one step."""
        return self.load_method(name, version)(*args, **kwargs)


# Promotion


def _fence(source: str) -> str:
    longest = max((len(run) for run in re.findall(r"`+", source)), default=0)
    return "`" * max(3, longest + 1)


def _title(name: str) -> str:
    words = re.sub(r"[_-]+", " ", name).strip()
    return words[:1].upper() + words[1:]


def _setup_list(meta: dict[str, Any], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every setup of a version: where it was saved, then any a result was graded in."""
    setups = list(meta.get("setups") or [meta["setup"]])
    for record in records:
        setup = record.get("setup")
        if setup is not None and setup not in setups:
            setups.append(setup)
    return setups


def _task_rows(records: list[dict[str, Any]], setups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        setup = record.get("setup")
        number = setups.index(setup) + 1 if setup is not None else 0
        row = rows.setdefault(
            (record["task"], number), {"split": record["split"], "scores": [], "setup": number}
        )
        row["scores"].append(record["score"])
    return [
        {
            "task": task,
            "split": row["split"],
            "setup": row["setup"] or "-",
            "runs": len(row["scores"]),
            "score": sum(float(s["score"]) for s in row["scores"]) / len(row["scores"]),
            "solved": sum(bool(s.get("solved")) for s in row["scores"]) / len(row["scores"]),
        }
        for (task, _), row in sorted(rows.items())
    ]


def _matching_setups(
    mine: list[dict[str, Any]], other: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Keep the results of each side that have a same-task result of the other in a matching setup.

    Results graded by ``proving-ground grade`` carry no setup: the task id fixes the model,
    so they match on the task. When both results carry a setup, the setups must be equal.
    Returns both kept lists and what was left out: the counts and the setup fields that
    differed.
    """
    from .evaluation import _setup_differences

    def compatible(a: dict[str, Any], b: dict[str, Any]) -> bool:
        left, right = a.get("setup"), b.get("setup")
        return left is None or right is None or not _setup_differences(left, right)

    fields: set[str] = set()

    def keep(side: list[dict[str, Any]], against: list[dict[str, Any]]) -> list[dict[str, Any]]:
        kept = []
        for record in side:
            counterparts = [r for r in against if r["task"] == record["task"]]
            if any(compatible(record, r) for r in counterparts):
                kept.append(record)
            else:
                for r in counterparts:
                    fields.update(_setup_differences(record.get("setup"), r.get("setup")))
        return kept

    kept_mine, kept_other = keep(mine, other), keep(other, mine)
    # A task kept on one side only because the other side's matching result was itself
    # unmatched cannot happen: compatibility is symmetric.
    left_out = {
        "mine": sum(r["task"] in {o["task"] for o in other} for r in mine) - len(kept_mine),
        "other": sum(r["task"] in {m["task"] for m in mine} for r in other) - len(kept_other),
        "fields": sorted(fields),
    }
    return kept_mine, kept_other, left_out


def promotion_draft(
    name: str,
    version: str | None = None,
    *,
    against: str | None = None,
    directory: str | Path | None = None,
    force: bool = False,
) -> str:
    """The text of a recipe draft for a saved method, for a maintainer to edit.

    Holds a title, the docstring, the source, every setup the method was saved or graded
    in, and a table of its proving-ground results per task. With ``against`` (``NAME`` or
    ``NAME@VERSION``) it adds the paired comparison on the tasks both were graded on, using
    only results from matching setups and saying what it left out. Refuses a method without
    scored results, and with ``against`` a method that does not beat the other by the
    paired interval unless ``force``. Two methods with no task and setup in common are
    always refused. A grade whose score was NaN has no score and is left out.
    """
    root = methods_dir(directory)
    path = _version_dir(root, name, version)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    everything = read_method_results(root, name, meta["version"])
    records = _scored(everything)
    if not records:
        raise ValueError(
            f"{name}@{meta['version']} has no scored proving-ground results; grade it with "
            "proving-ground grade --method first"
        )
    comparison = ""
    if against is not None:
        try:
            from .proving import beats, compare
        except ImportError as error:
            raise ValueError(
                "--against needs the proving package, which is not installed"
            ) from error
        other_name, other_version = parse_reference(against)
        other = _version_dir(root, other_name, other_version).name
        other_records = _scored(read_method_results(root, other_name, other))
        if not other_records:
            raise ValueError(f"{other_name}@{other} has no scored proving-ground results")
        mine, theirs, left_out = _matching_setups(records, other_records)
        if not mine:
            raise ValueError(
                f"{name}@{meta['version']} and {other_name}@{other} share no task graded in "
                "a matching setup, so there is nothing to pair"
            )
        report = compare(theirs, mine)
        wins = beats(report)
        if not wins and not force:
            low, high = report["delta_ci"]
            raise ValueError(
                f"{name}@{meta['version']} does not beat {other_name}@{other}: the paired "
                f"difference is {report['delta']:+.3f} with a 95% interval of "
                f"[{low:+.3f}, {high:+.3f}]; pass --force to write the draft anyway"
            )
        low, high = report["delta_ci"]
        excluded = ""
        if left_out["mine"] or left_out["other"]:
            excluded = (
                f"\nLeft out because the setups differ ({', '.join(left_out['fields'])}): "
                f"{left_out['mine']} results of this method and {left_out['other']} of "
                f"{other_name}@{other}.\n"
            )
        comparison = (
            f"\n## Compared with {other_name}@{other}\n\n"
            f"On the {report['tasks']} tasks both were graded on, the mean score is "
            f"{report['right_mean']:.3f} for this method and {report['left_mean']:.3f} for "
            f"{other_name}@{other}. The paired difference is {report['delta']:+.3f}, with a "
            f"95% interval of [{low:+.3f}, {high:+.3f}] and a permutation p of "
            f"{report['permutation_p']:.4f}. Solve rate: {report['right_solve_rate']:.2f} against "
            f"{report['left_solve_rate']:.2f}. This method "
            f"{'beats' if wins else 'does not beat'} the other by the paired interval.\n"
            f"{excluded}"
        )
    source = (path / "method.py").read_text(encoding="utf-8")
    fence = _fence(source)
    setups = _setup_list(meta, records)
    table = (
        "| Task | Split | Runs | Mean score | Solve rate | Setup |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
    )
    table += "".join(
        f"| {r['task']} | {r['split']} | {r['runs']} | {r['score']:.3f} | {r['solved']:.2f} "
        f"| {r['setup']} |\n"
        for r in _task_rows(records, setups)
    )
    summary = summarize_results(records)
    unscored = (
        f" {len(everything) - len(records)} results had no score and are left out."
        if len(everything) > len(records)
        else ""
    )
    setup_text = "".join(
        f"### Setup {number}\n\n```json\n"
        f"{json.dumps(setup, indent=2, sort_keys=True, default=str)}\n```\n\n"
        for number, setup in enumerate(setups, 1)
    )
    return (
        f"<!-- Draft for maintainer review, written from saved method {name}@{meta['version']} "
        f"(lab {meta['lab_version']}). Edit it before it goes into the skill. -->\n"
        f"# {_title(name)}\n\n"
        f"{meta['docstring'] or NO_DOCSTRING}\n\n"
        f"## Source\n\nVersion {meta['version']}. Call it with `lab.run_method('{name}', ...)`.\n\n"
        f"{fence}python\n{source}{fence}\n\n"
        f"## Setups it ran on\n\nThe method was saved or graded in each of these. A result's "
        f"Setup column names the one it ran in; a dash means the grade recorded none, and the "
        f"task fixes the model.\n\n{setup_text}"
        f"## Results on the proving ground\n\n"
        f"{summary['tasks']} tasks, mean score {summary['mean_score']:.3f}, solve rate "
        f"{summary['solve_rate']:.2f}.{unscored}\n\n{table}{comparison}"
    )


def write_draft(text: str, output: str | Path, *, overwrite: bool = False) -> Path:
    """Write a recipe draft.

    Refuses a path under a ``skills`` directory and, without ``overwrite``, an existing file.
    """
    path = Path(output).expanduser().resolve()
    if "skills" in path.parts:
        raise ValueError(
            f"{path} is inside a skills directory; write the draft elsewhere and add it to "
            "the skill in a reviewed change"
        )
    if path.exists() and not overwrite:
        raise ValueError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
