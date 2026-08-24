#!/usr/bin/env python3
"""Fail closed before a TileFoundry performance PR is pushed or updated."""

from __future__ import annotations

import argparse
import ast
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path


class GateError(ValueError):
    """The round has not established a reviewable TileFoundry optimization."""


CORE_ANALYSIS_FLAGS = ("--compute-cost", "--memory", "--roofline", "--performance")
ALLOWED_CHANGE_ROOTS = (
    "src/tileops/kernels/",
    "src/tileops/ops/",
    "tests/kernels/",
    "tests/ops/",
)
PR_SECTIONS = (
    "Summary",
    "TileFoundry Description",
    "Performance",
    "Result And Limitations",
)
FINDING_CLASSES = {
    "semantic-blocker",
    "lowering/codegen-blocker",
    "runtime-blocker",
    "performance-blocker",
    "ergonomics",
}


def _real_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and not value.strip().startswith("<")
    )


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise GateError(f"expected a JSON object: {path}")
    return value


def _required_file(root: Path, relative: object) -> Path:
    if not _real_text(relative):
        raise GateError("required artifact path is empty")
    path = (root / str(relative)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise GateError(f"artifact escapes the round directory: {relative}") from error
    if not path.is_file() or path.stat().st_size == 0:
        raise GateError(f"required artifact is missing or empty: {relative}")
    return path


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        return _call_name(node.func)
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _terminal_name(node: ast.AST) -> str:
    return _call_name(node).rsplit(".", 1)[-1]


def _names_and_strings(node: ast.AST) -> set[str]:
    values = {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}
    values.update(
        item.attr for item in ast.walk(node) if isinstance(item, ast.Attribute)
    )
    values.update(
        item.value
        for item in ast.walk(node)
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    )
    return values


def _module_classes(tree: ast.Module) -> list[ast.ClassDef]:
    return [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            _terminal_name(decorator) == "module" for decorator in node.decorator_list
        )
    ]


def _has_target(module: ast.ClassDef) -> bool:
    for decorator in module.decorator_list:
        if not isinstance(decorator, ast.Call) or _terminal_name(decorator) != "module":
            continue
        for keyword in decorator.keywords:
            if keyword.arg == "target" and not (
                isinstance(keyword.value, ast.Constant) and keyword.value.value is None
            ):
                return True
    return False


def _reshard_storage(call: ast.Call) -> set[str]:
    values: list[ast.AST] = []
    values.extend(
        keyword.value for keyword in call.keywords if keyword.arg == "storage"
    )
    if len(call.args) >= 3:
        values.append(call.args[2])
    names: set[str] = set()
    for value in values:
        names.update(_names_and_strings(value))
    return names


def _validate_placed_hir_text(source: str, label: str) -> str:
    try:
        tree = ast.parse(source, filename=label)
    except SyntaxError as error:
        raise GateError(f"cannot parse HIR {label}: {error}") from error
    modules = _module_classes(tree)
    if len(modules) != 1:
        raise GateError(f"{label} must contain exactly one @module class")
    module = modules[0]
    if not _has_target(module):
        raise GateError(f"{label} @module must declare its compilation target")

    calls = [node for node in ast.walk(module) if isinstance(node, ast.Call)]
    call_names = {_terminal_name(call) for call in calls}
    if "Mesh" not in call_names:
        raise GateError(f"{label} must declare an explicit Mesh")
    reshards = [call for call in calls if _terminal_name(call) == "reshard"]
    if len(reshards) < 2:
        raise GateError(
            f"{label} must explicitly move into and out of placed storage with reshard"
        )
    has_sharded_layout = "ShardLayout" in call_names or any(
        isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult)
        for node in ast.walk(module)
    )
    if not has_sharded_layout:
        raise GateError(f"{label} must bind tensor dimensions to the Mesh")
    storage = set().union(*(_reshard_storage(call) for call in reshards))
    if "gmem" not in storage:
        raise GateError(f"{label} must include explicit global-memory placement")
    if not storage.intersection({"smem", "rmem", "tmem"}):
        raise GateError(f"{label} must include an explicit local storage tier")
    return ast.dump(module, include_attributes=False)


def _validate_placed_hir(path: Path) -> str:
    return _validate_placed_hir_text(path.read_text(encoding="utf-8"), str(path))


def _command(record: object, section: str, command: str) -> list[str]:
    if not isinstance(record, dict):
        raise GateError(f"{section} must be an object")
    value = record.get("command")
    if not _real_text(value) or "..." in str(value):
        raise GateError(f"{section} must record the exact command")
    try:
        argv = shlex.split(str(value))
    except ValueError as error:
        raise GateError(f"{section} command cannot be parsed: {error}") from error
    pairs = [
        (Path(argv[index]).name, argv[index + 1]) for index in range(len(argv) - 1)
    ]
    if ("tilefoundry", command) not in pairs:
        raise GateError(f"{section} must run tilefoundry {command}")
    return argv


def _json_report(root: Path, record: dict, section: str) -> dict:
    report = _load_json(_required_file(root, record.get("report")))
    if not report:
        raise GateError(f"{section} report is empty")
    return report


def _supported_analysis_flags() -> tuple[str, ...]:
    process = subprocess.run(
        ["tilefoundry", "analyze", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise GateError("cannot query the installed tilefoundry analyze surface")
    help_text = process.stdout + process.stderr
    flags = tuple(flag for flag in CORE_ANALYSIS_FLAGS if flag in help_text)
    if not flags:
        raise GateError(
            "installed tilefoundry exposes no recognized core analysis surface"
        )
    return flags


def _validate_runtime(path: Path, production_symbol: str) -> None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as error:
        raise GateError(f"cannot parse runtime twin {path}: {error}") from error
    calls = {
        _terminal_name(node) for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    vocabulary = _names_and_strings(tree)
    decorators = {
        _terminal_name(node) for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    decorators.update(
        _terminal_name(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    )
    if "runtime_module" not in decorators or "runtime_func" not in vocabulary:
        raise GateError("runtime twin must use @runtime_module and @runtime_func")
    if "NotImplementedError" in vocabulary:
        raise GateError("runtime twin still contains a placeholder implementation")
    if production_symbol not in calls:
        raise GateError("runtime twin must call the declared production kernel symbol")


def _git(repo: Path, *args: str, text: bool = True) -> subprocess.CompletedProcess:
    process = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=text,
    )
    if process.returncode != 0:
        detail = (
            process.stderr.strip()
            if text
            else process.stderr.decode(errors="replace").strip()
        )
        raise GateError(detail or f"git {' '.join(args)} failed")
    return process


def _changed_paths(repo: Path, base: str, head: str) -> list[str]:
    process = _git(repo, "diff", "--name-only", "-z", base, head, text=False)
    return [
        part.decode(errors="replace") for part in process.stdout.split(b"\0") if part
    ]


def _git_source(repo: Path, revision: str, path: str) -> str | None:
    process = subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{path}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode == 0:
        return process.stdout
    if (
        "does not exist" in process.stderr
        or "exists on disk, but not in" in process.stderr
    ):
        return None
    raise GateError(process.stderr.strip() or f"cannot read {revision}:{path}")


def _public_surface(source: str | None, revision: str, path: str) -> dict[str, str]:
    if source is None:
        return {}
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise GateError(f"cannot parse {revision}:{path}: {error}") from error
    surface: dict[str, str] = {}
    for node in tree.body:
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and not node.name.startswith("_"):
            surface[node.name] = ast.dump(node.args, include_attributes=False)
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            for member in node.body:
                if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if member.name.startswith("_") and member.name not in {
                    "__init__",
                    "__call__",
                }:
                    continue
                surface[f"{node.name}.{member.name}"] = ast.dump(
                    member.args, include_attributes=False
                )
    return surface


def _kernel_function(
    source: str | None, symbol: str, revision: str, path: str
) -> ast.AST | None:
    if source is None:
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise GateError(f"cannot parse {revision}:{path}: {error}") from error
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == symbol
        and any(
            _terminal_name(decorator) in {"prim_func", "macro"}
            for decorator in node.decorator_list
        )
    ]
    if len(matches) > 1:
        raise GateError(
            f"{revision}:{path} contains multiple decorated kernels named {symbol}"
        )
    return matches[0] if matches else None


class _EraseConstants(ast.NodeTransformer):
    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        marker = f"<{type(node.value).__name__}>"
        return ast.copy_location(ast.Constant(value=marker), node)


def _structural_dump(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    normalized = _EraseConstants().visit(ast.fix_missing_locations(node))
    return ast.dump(normalized, include_attributes=False)


def _validate_kernel_diff(repo: Path, base: str, head: str, declaration: str) -> None:
    changed = _changed_paths(repo, base, head)
    disallowed = [path for path in changed if not path.startswith(ALLOWED_CHANGE_ROOTS)]
    if disallowed:
        raise GateError(
            "production diff changes forbidden paths: " + ", ".join(disallowed)
        )
    for path in changed:
        if not path.startswith("src/tileops/ops/") or not path.endswith(".py"):
            continue
        before = _public_surface(_git_source(repo, base, path), base, path)
        after = _public_surface(_git_source(repo, head, path), head, path)
        if before != after:
            raise GateError(
                f"production dispatch changes the public Op surface: {path}"
            )

    try:
        path, symbol = declaration.rsplit(":", 1)
    except ValueError as error:
        raise GateError("production_kernel must be relative-path:symbol") from error
    if not path.startswith("src/tileops/kernels/") or not path.endswith(".py"):
        raise GateError("production kernel must live under src/tileops/kernels/")
    if path not in changed:
        raise GateError("the declared production kernel file is unchanged")
    before = _kernel_function(_git_source(repo, base, path), symbol, base, path)
    after = _kernel_function(_git_source(repo, head, path), symbol, head, path)
    if after is None:
        raise GateError(f"declared @T.prim_func/@T.macro is absent: {declaration}")
    if _structural_dump(before) == _structural_dump(after):
        raise GateError(
            "production kernel changed only constants/configuration; a structural "
            "@T.prim_func/@T.macro implementation or schedule change is required"
        )

    messages = _git(repo, "log", "--format=%B", f"{base}..{head}").stdout
    if re.search(
        r"^(Co-authored-by|Signed-off-by):",
        messages,
        flags=re.MULTILINE | re.IGNORECASE,
    ):
        raise GateError("commit messages contain an extra authorship trailer")


def _validate_pr(round_dir: Path, final_module: str) -> None:
    title = (
        _required_file(round_dir, "pr-title.txt").read_text(encoding="utf-8").strip()
    )
    if not re.fullmatch(r"\[Perf\]\[foundry\]\[[A-Za-z0-9_-]+\] .+", title):
        raise GateError(
            "pr-title.txt does not follow [Perf][foundry][Scope] description"
        )
    body = _required_file(round_dir, "pr-body.md").read_text(encoding="utf-8")
    headings = tuple(re.findall(r"^## (.+)$", body, flags=re.MULTILINE))
    if headings != PR_SECTIONS:
        raise GateError(
            "pr-body.md must contain exactly the four ordered public sections"
        )
    section = body.split("## TileFoundry Description", 1)[1].split("## Performance", 1)[
        0
    ]
    match = re.search(r"```python\s*\n(.*?)\n```", section, flags=re.DOTALL)
    if not match:
        raise GateError("TileFoundry Description must contain one Python HIR block")
    public_module = _validate_placed_hir_text(match.group(1), "TileFoundry Description")
    if public_module != final_module:
        raise GateError("TileFoundry Description is not the exact final placed HIR")
    forbidden = re.search(
        r"(?:/home/|/mnt/|/tmp/|/workspace/|/Users/|file://|[A-Za-z]:\\)", body
    )
    if forbidden:
        raise GateError("pr-body.md exposes a local path")


def _validate_findings(round_dir: Path) -> None:
    data = _load_json(_required_file(round_dir, "findings.json"))
    findings = data.get("findings")
    if not isinstance(findings, list):
        raise GateError("findings.json must contain a findings list")
    for finding in findings:
        if not isinstance(finding, dict) or not _real_text(finding.get("id")):
            raise GateError("every finding must have an id")
        if finding.get("classification") not in FINDING_CLASSES:
            raise GateError("every finding must use a supported classification")
        for field in (
            "command",
            "expected",
            "actual",
            "workaround_cost",
            "public_surface",
        ):
            if not _real_text(finding.get(field)):
                raise GateError(f"every finding must record {field}")
        if not isinstance(finding.get("affected_workloads"), list):
            raise GateError("every finding must record affected_workloads")
        _required_file(round_dir, finding.get("reproducer"))


def validate_round(
    round_dir: Path,
    tileops_repo: Path,
    head: str = "HEAD",
    analysis_flags: tuple[str, ...] | None = None,
) -> dict:
    round_dir = round_dir.resolve()
    tileops_repo = tileops_repo.resolve()
    provenance = _load_json(_required_file(round_dir, "provenance.json"))
    base = provenance.get("tileops_base")
    if not isinstance(base, str) or re.fullmatch(r"[0-9a-f]{40}", base) is None:
        raise GateError("provenance tileops_base must be the full admitted commit")
    if _git(tileops_repo, "rev-parse", f"{base}^{{commit}}").stdout.strip() != base:
        raise GateError(
            "provenance tileops_base is not available in the TileOPs repository"
        )
    if _git(tileops_repo, "status", "--porcelain").stdout.strip():
        raise GateError(
            "TileOPs worktree must be clean and committed before the PR gate"
        )

    classification = provenance.get("classification")
    if classification not in {
        "measured SOTA",
        "improvement without SOTA",
        "no improvement",
    }:
        raise GateError(
            "classification must be measured SOTA, improvement without SOTA, or no improvement"
        )
    if classification == "no improvement":
        raise GateError("no-improvement rounds cannot open or retain a performance PR")

    final_hir_value = provenance.get("final_hir")
    final_hir = _required_file(round_dir, final_hir_value)
    final_module = _validate_placed_hir(final_hir)
    production_kernel = provenance.get("production_kernel")
    if not _real_text(production_kernel):
        raise GateError(
            "provenance must declare production_kernel as relative-path:symbol"
        )
    production_symbol = str(production_kernel).rsplit(":", 1)[-1]
    _validate_runtime(
        _required_file(round_dir, provenance.get("runtime_twin")), production_symbol
    )

    check_record = provenance.get("tilefoundry_check")
    _command(check_record, "tilefoundry_check", "check")
    check_report = _json_report(round_dir, check_record, "tilefoundry_check")
    if check_report.get("passed") is not True:
        raise GateError("tilefoundry check report must say passed=true")

    flags = (
        analysis_flags if analysis_flags is not None else _supported_analysis_flags()
    )
    iterations = provenance.get("iterations")
    if not isinstance(iterations, list) or len(iterations) < 2:
        raise GateError(
            "at least two materially different placed-HIR iterations are required"
        )
    verdicts: list[str] = []
    placements: set[str] = set()
    hir_structures: set[str] = set()
    kept_hir: str | None = None
    for index, iteration in enumerate(iterations, start=1):
        section = f"iterations[{index}]"
        if not isinstance(iteration, dict):
            raise GateError(f"{section} must be an object")
        for field in ("name", "hypothesis", "placement"):
            if not _real_text(iteration.get(field)):
                raise GateError(f"{section} must record {field}")
        placement = str(iteration["placement"]).strip()
        placements.add(placement)
        hir_value = iteration.get("hir")
        hir_path = _required_file(round_dir, hir_value)
        hir_structures.add(_validate_placed_hir(hir_path))
        verdict = iteration.get("verdict")
        if verdict not in {"kept", "rejected"}:
            raise GateError(f"{section} verdict must be kept or rejected")
        verdicts.append(str(verdict))
        if verdict == "kept":
            kept_hir = str(hir_value)
        measured_ms = iteration.get("measured_ms")
        if (
            not isinstance(measured_ms, (int, float))
            or isinstance(measured_ms, bool)
            or measured_ms <= 0
        ):
            raise GateError(f"{section} must record a positive measured_ms")

        analysis = iteration.get("analysis")
        analysis_argv = _command(analysis, f"{section}.analysis", "analyze")
        missing = [flag for flag in flags if flag not in analysis_argv]
        if missing or "--json" not in analysis_argv:
            raise GateError(
                f"{section}.analysis is missing current core surfaces: {', '.join(missing)}"
            )
        _json_report(round_dir, analysis, f"{section}.analysis")

        schedule = iteration.get("schedule")
        schedule_argv = _command(schedule, f"{section}.schedule", "schedule")
        if "--topology" not in schedule_argv or "--json" not in schedule_argv:
            raise GateError(
                f"{section}.schedule must record topology and JSON evidence"
            )
        if verdict == "kept" and "--first-plan" in schedule_argv:
            raise GateError(
                "the final kept schedule must complete its search, not use --first-plan"
            )
        _json_report(round_dir, schedule, f"{section}.schedule")

    if verdicts.count("kept") != 1 or "rejected" not in verdicts:
        raise GateError(
            "iterations must contain exactly one kept placement and at least one rejected placement"
        )
    if kept_hir != final_hir_value:
        raise GateError("the kept iteration must point to final_hir")
    if len(placements) < 2 or len(hir_structures) < 2:
        raise GateError(
            "iterations must compare materially different HIR and placement strategies"
        )

    decisions = provenance.get("decisions")
    required_decisions = (
        "analysis_fact",
        "schedule_fact",
        "hir_change",
        "kernel_change",
    )
    if not isinstance(decisions, list) or not decisions:
        raise GateError("at least one evidence-to-kernel decision is required")
    for decision in decisions:
        if not isinstance(decision, dict) or any(
            not _real_text(decision.get(field)) for field in required_decisions
        ):
            raise GateError(
                "every decision must link analysis, schedule, HIR, and kernel changes"
            )

    for section in ("correctness", "benchmark", "profile"):
        record = provenance.get(section)
        if not isinstance(record, dict):
            raise GateError(f"provenance is missing {section}")
        if not _real_text(record.get("command")) or "..." in str(record.get("command")):
            raise GateError(f"{section} must record the exact command")
        _required_file(round_dir, record.get("report"))

    _required_file(round_dir, "report.md")
    _validate_findings(round_dir)
    _validate_kernel_diff(tileops_repo, base, head, str(production_kernel))
    _validate_pr(round_dir, final_module)
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-dir", type=Path, default=Path.cwd())
    parser.add_argument("--tileops-repo", type=Path, default=Path("/workspace/tileops"))
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args()
    try:
        validate_round(args.round_dir, args.tileops_repo, args.head)
    except GateError as error:
        print(f"TileFoundry PR gate: FAIL: {error}", file=sys.stderr)
        return 1
    print("TileFoundry PR gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
