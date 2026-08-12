"""Validate the TileFoundry provenance required for a performance round."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


class RoundError(ValueError):
    """A round is missing required provenance."""


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RoundError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise RoundError(f"expected a JSON object: {path}")
    return value


def _call_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            names.add(target.attr)
    return names


def _names_and_strings(tree: ast.AST) -> set[str]:
    values = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    values.update(node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute))
    values.update(
        node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )
    return values


def _require_path(round_dir: Path, relative: str) -> Path:
    path = (round_dir / relative).resolve()
    try:
        path.relative_to(round_dir.resolve())
    except ValueError as error:
        raise RoundError(f"path escapes round directory: {relative}") from error
    if not path.is_file():
        raise RoundError(f"required file is missing: {relative}")
    return path


def _real_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and not value.strip().startswith("<")


def _validate_hir(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    calls = _call_names(tree)
    vocabulary = _names_and_strings(tree)
    if not ({"Mesh", "ShardLayout"} & calls):
        raise RoundError("authored HIR must declare a Mesh or ShardLayout")
    if "reshard" not in calls:
        raise RoundError("authored HIR must make shard/storage movement explicit with reshard")
    if "gmem" not in vocabulary:
        raise RoundError("authored HIR must include global-memory storage")
    if not ({"rmem", "smem", "tmem"} & vocabulary):
        raise RoundError("authored HIR must include an explicit local storage tier")


def _validate_report(round_dir: Path, section: str, record: dict, findings: set[str]) -> None:
    command = record.get("command")
    if not _real_text(command) or "..." in str(command):
        raise RoundError(f"{section} must record the exact command")
    status = record.get("status")
    if status == "blocked":
        finding_id = record.get("finding_id")
        if finding_id not in findings:
            raise RoundError(f"{section} is blocked without a matching finding")
        return
    if status != "passed":
        raise RoundError(f"{section} status must be passed or blocked")
    report = _load_json(_require_path(round_dir, str(record.get("report", ""))))
    if not report:
        raise RoundError(f"{section} report is empty")
    if section == "tilefoundry_check" and report.get("passed") is not True:
        raise RoundError("TileFoundry check report does not say passed=true")


def validate_round(round_dir: Path) -> dict:
    round_dir = round_dir.resolve()
    provenance = _load_json(_require_path(round_dir, "provenance.json"))
    tilefoundry = provenance.get("tilefoundry")
    if not isinstance(tilefoundry, dict):
        raise RoundError("provenance must record the admitted TileFoundry wheel")
    if re.fullmatch(r"[0-9a-f]{40}", str(tilefoundry.get("commit", ""))) is None:
        raise RoundError("TileFoundry commit must be a full Git commit")
    if re.fullmatch(r"[0-9a-f]{64}", str(tilefoundry.get("wheel_sha256", ""))) is None:
        raise RoundError("TileFoundry wheel_sha256 must be a SHA-256 digest")
    if not _real_text(tilefoundry.get("version")):
        raise RoundError("TileFoundry installed version is missing")
    findings_data = _load_json(_require_path(round_dir, "findings.json"))
    findings = findings_data.get("findings")
    if not isinstance(findings, list):
        raise RoundError("findings.json must contain a findings list")
    finding_ids: set[str] = set()
    for finding in findings:
        if not isinstance(finding, dict) or not finding.get("id"):
            raise RoundError("every finding must be an object with an id")
        if finding.get("classification") not in {
            "semantic-blocker",
            "lowering/codegen-blocker",
            "runtime-blocker",
            "performance-blocker",
            "ergonomics",
        }:
            raise RoundError("every finding must have a supported classification")
        for field in ("command", "expected", "actual", "workaround_cost", "public_surface"):
            if not _real_text(finding.get(field)):
                raise RoundError(f"every finding must record {field}")
        if not isinstance(finding.get("affected_workloads"), list):
            raise RoundError("every finding must record affected_workloads")
        finding_ids.add(str(finding["id"]))
        _require_path(round_dir, str(finding.get("reproducer", "")))

    hir = provenance.get("hir")
    runtime = provenance.get("runtime_twin")
    if not isinstance(hir, dict) or not isinstance(runtime, dict):
        raise RoundError("provenance must declare hir and runtime_twin")
    _validate_hir(_require_path(round_dir, str(hir.get("source", ""))))
    runtime_path = _require_path(round_dir, str(runtime.get("source", "")))
    runtime_source = runtime_path.read_text(encoding="utf-8")
    runtime_tree = ast.parse(runtime_source, filename=str(runtime_path))
    calls = _call_names(runtime_tree)
    vocabulary = _names_and_strings(runtime_tree)
    if "runtime_module" not in calls or "runtime_func" not in vocabulary:
        raise RoundError("runtime twin must use runtime_module and runtime_func")
    if "NotImplementedError" in vocabulary:
        raise RoundError("runtime twin still contains its placeholder implementation")
    production_symbol = runtime.get("production_symbol")
    if not isinstance(production_symbol, str) or production_symbol not in calls:
        raise RoundError("runtime twin must call the declared production symbol")

    for section in ("tilefoundry_check", "analysis", "schedule"):
        record = provenance.get(section)
        if not isinstance(record, dict):
            raise RoundError(f"provenance is missing {section}")
        _validate_report(round_dir, section, record, finding_ids)
    analysis_command = str(provenance["analysis"].get("command", ""))
    for option in ("--compute-cost", "--memory", "--roofline", "--timeline"):
        if option not in analysis_command:
            raise RoundError(f"analysis command is missing {option}")

    decisions = provenance.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise RoundError("at least one TileFoundry-to-kernel decision is required")
    for decision in decisions:
        required = ("tilefoundry_evidence", "hir_choice", "kernel_site")
        if not isinstance(decision, dict) or any(not _real_text(decision.get(key)) for key in required):
            raise RoundError("every decision must link evidence, an HIR choice, and a kernel site")

    experiments = provenance.get("primitive_experiments")
    if not isinstance(experiments, list) or not experiments:
        raise RoundError("at least one profiler-driven lower-level primitive experiment is required")
    for experiment in experiments:
        required = ("bottleneck", "primitive", "hypothesis")
        if not isinstance(experiment, dict) or any(
            not _real_text(experiment.get(key)) for key in required
        ):
            raise RoundError("every primitive experiment needs bottleneck, primitive, and hypothesis")
        if experiment.get("verdict") not in {"kept", "rejected", "inconclusive"}:
            raise RoundError("primitive experiment verdict must be kept, rejected, or inconclusive")
        if not all(
            isinstance(experiment.get(key), (int, float)) and experiment[key] > 0
            for key in ("control_ms", "candidate_ms")
        ):
            raise RoundError("primitive experiment requires positive control_ms and candidate_ms")
    return provenance
