"""Render and validate a TileFoundry-originated TileOPs performance PR."""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CLASSIFICATIONS = {
    "measured SOTA",
    "improvement without SOTA",
    "no improvement",
}
PR_TYPES = {
    "Bench",
    "BugFix",
    "Chore",
    "CI",
    "Design",
    "Doc",
    "Enhancement",
    "Feat",
    "Fix",
    "Maintain",
    "Perf",
    "Refactor",
    "Style",
    "Test",
}
REQUIRED_ROLES = {"candidate", "incumbent", "external"}
PR_FIELDS = {
    "classification",
    "comparators",
    "dimensions",
    "environment",
    "limitations",
    "method",
    "operator",
    "scope",
    "subject",
    "summary",
    "type",
    "workloads",
}
RENDERER_OWNED_FIELDS = {
    "artifacts",
    "correctness",
    "reproduce",
    "tilefoundry_description",
}
FORBIDDEN_PUBLIC_TEXT = (
    "## Artifacts",
    "## Correctness",
    "## Reproduce",
    "Entrypoint:",
    "Source:",
    "/home/",
    "/mnt/",
    "/tmp/",
    "/workspace/",
    "/Users/",
    "file://",
)


class PRContractError(ValueError):
    """The structured PR data does not satisfy the workflow contract."""


@dataclass(frozen=True)
class RenderedPR:
    title: str
    body: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PRContractError(message)


def _text(value: Any, field: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{field} must be non-empty text")
    return value.strip()


def _list(value: Any, field: str) -> list[Any]:
    _require(isinstance(value, list) and bool(value), f"{field} must be a non-empty list")
    return value


def _escape_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _format_number(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def load_data(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PRContractError(f"cannot load {path}: {error}") from error
    _require(isinstance(value, dict), "PR data root must be an object")
    return value


def _module_description(data_path: Path) -> str:
    source_path = data_path.parent / "authored_hir.py"
    _require(source_path.is_file(), f"TileFoundry HIR source does not exist: {source_path}")
    source = source_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise PRContractError(
            f"TileFoundry HIR source must be valid Python syntax: {error}"
        ) from error
    module_classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            (isinstance(decorator, ast.Name) and decorator.id == "module")
            or (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "module"
            )
            for decorator in node.decorator_list
        )
    ]
    _require(
        len(module_classes) == 1,
        "authored_hir.py must contain exactly one top-level @module class",
    )
    module_class = module_classes[0]
    start_line = min(
        (decorator.lineno for decorator in module_class.decorator_list),
        default=module_class.lineno,
    )
    _require(
        module_class.end_lineno is not None,
        f"cannot extract HIR class {module_class.name}",
    )
    extracted = "\n".join(
        source.splitlines()[start_line - 1 : module_class.end_lineno]
    ).strip()
    _require(
        not any(
            isinstance(node, (ast.Import, ast.ImportFrom))
            for node in ast.walk(module_class)
        ),
        "extracted TileFoundry module must not contain imports",
    )
    return extracted


def _validate_public_body(body: str) -> None:
    leaked = [item for item in FORBIDDEN_PUBLIC_TEXT if item in body]
    _require(
        not leaked,
        "public PR body contains private evidence or local paths: " + ", ".join(leaked),
    )
    _require(
        re.search(r"(?<![A-Za-z0-9_])[A-Za-z0-9_./-]+\.py\b", body) is None,
        "public PR body contains a Python source filename",
    )
    _require(
        re.search(r"(?<![A-Za-z0-9_])[A-Za-z]:\\", body) is None,
        "public PR body contains a Windows local path",
    )


def _validate(data: dict[str, Any], data_path: Path) -> str:
    forbidden = sorted(RENDERER_OWNED_FIELDS.intersection(data))
    _require(
        not forbidden,
        "renderer-owned or private fields do not belong in PR data: "
        + ", ".join(forbidden),
    )
    unknown = sorted(set(data) - PR_FIELDS)
    _require(not unknown, "unknown PR data fields: " + ", ".join(unknown))
    missing = sorted(PR_FIELDS - set(data))
    _require(not missing, "missing PR data fields: " + ", ".join(missing))
    _text(data.get("subject"), "subject")
    change_type = _text(data.get("type"), "type")
    _require(change_type in PR_TYPES, f"type must be one of {sorted(PR_TYPES)}")
    scope = _text(data.get("scope"), "scope")
    _require(
        re.fullmatch(r"[A-Za-z0-9_-]+", scope) is not None,
        "scope must contain only letters, digits, underscores, or hyphens",
    )
    _require(
        scope.lower() != "foundry",
        "scope must not reuse the reserved foundry origin tag",
    )
    _text(data.get("operator"), "operator")
    classification = _text(data.get("classification"), "classification")
    _require(
        classification in CLASSIFICATIONS,
        f"classification must be one of {sorted(CLASSIFICATIONS)}",
    )

    for index, item in enumerate(_list(data.get("summary"), "summary")):
        _text(item, f"summary[{index}]")

    tilefoundry_description = _module_description(data_path)

    environment = data.get("environment")
    _require(isinstance(environment, dict) and bool(environment), "environment must be an object")
    for key, value in environment.items():
        _text(key, "environment key")
        _text(value, f"environment.{key}")
    _text(data.get("method"), "method")

    dimensions = _list(data.get("dimensions"), "dimensions")
    _require(len(dimensions) == len(set(dimensions)), "dimensions must be unique")
    for index, item in enumerate(dimensions):
        _text(item, f"dimensions[{index}]")

    comparators = _list(data.get("comparators"), "comparators")
    comparator_ids: list[str] = []
    roles: list[str] = []
    for index, comparator in enumerate(comparators):
        _require(isinstance(comparator, dict), f"comparators[{index}] must be an object")
        comparator_ids.append(_text(comparator.get("id"), f"comparators[{index}].id"))
        _text(comparator.get("label"), f"comparators[{index}].label")
        roles.append(_text(comparator.get("role"), f"comparators[{index}].role"))
    _require(len(comparator_ids) == len(set(comparator_ids)), "comparator ids must be unique")
    _require(roles.count("candidate") == 1, "exactly one candidate comparator is required")
    _require(roles.count("incumbent") == 1, "exactly one incumbent comparator is required")
    _require(REQUIRED_ROLES.issubset(roles), "at least one external comparator is required")
    _require(set(roles) <= REQUIRED_ROLES, f"comparator roles must be in {sorted(REQUIRED_ROLES)}")

    workloads = _list(data.get("workloads"), "workloads")
    labels: list[str] = []
    for index, workload in enumerate(workloads):
        _require(isinstance(workload, dict), f"workloads[{index}] must be an object")
        labels.append(_text(workload.get("label"), f"workloads[{index}].label"))
        _text(workload.get("dtype"), f"workloads[{index}].dtype")
        workload_dimensions = workload.get("dimensions")
        _require(isinstance(workload_dimensions, dict), f"workloads[{index}].dimensions must be an object")
        _require(
            set(workload_dimensions) == set(dimensions),
            f"workloads[{index}].dimensions must exactly match dimensions",
        )
        results = workload.get("results")
        _require(isinstance(results, dict), f"workloads[{index}].results must be an object")
        _require(
            set(results) == set(comparator_ids),
            f"workloads[{index}].results must contain every comparator exactly once",
        )
        for comparator_id, result in results.items():
            _require(isinstance(result, dict), f"result {index}/{comparator_id} must be an object")
            latency = result.get("median_ms")
            _require(
                isinstance(latency, (int, float)) and not isinstance(latency, bool) and latency > 0,
                f"result {index}/{comparator_id}.median_ms must be positive",
            )
            if "noise_pct" in result:
                noise = result["noise_pct"]
                _require(
                    isinstance(noise, (int, float)) and not isinstance(noise, bool) and noise >= 0,
                    f"result {index}/{comparator_id}.noise_pct must be non-negative",
                )
    _require(len(labels) == len(set(labels)), "workload labels must be unique")

    for index, item in enumerate(_list(data.get("limitations"), "limitations")):
        _text(item, f"limitations[{index}]")

    return tilefoundry_description


def _performance_table(data: dict[str, Any]) -> str:
    dimensions: list[str] = data["dimensions"]
    comparators: list[dict[str, str]] = data["comparators"]
    workloads: list[dict[str, Any]] = data["workloads"]
    candidate = next(item for item in comparators if item["role"] == "candidate")
    ratios = [item for item in comparators if item["role"] != "candidate"]

    headers = ["Workload", *dimensions, "Dtype"]
    headers.extend(item["label"] + " (ms)" for item in comparators)
    headers.extend(item["label"] + " / candidate" for item in ratios)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", *["---:" for _ in dimensions], "---", *["---:" for _ in comparators], *["---:" for _ in ratios]]) + " |",
    ]

    for workload in workloads:
        results = workload["results"]
        candidate_latency = float(results[candidate["id"]]["median_ms"])
        row = [workload["label"]]
        row.extend(_format_number(workload["dimensions"][item]) for item in dimensions)
        row.append(workload["dtype"])
        for comparator in comparators:
            result = results[comparator["id"]]
            cell = f"{float(result['median_ms']):.6g}"
            if "noise_pct" in result:
                cell += f" (+/-{float(result['noise_pct']):.3g}%)"
            row.append(cell)
        for comparator in ratios:
            ratio = float(results[comparator["id"]]["median_ms"]) / candidate_latency
            row.append(f"{ratio:.4f}x")
        lines.append("| " + " | ".join(_escape_cell(item) for item in row) + " |")

    geomeans = {
        comparator["id"]: _geomean(
            [float(workload["results"][comparator["id"]]["median_ms"]) for workload in workloads]
        )
        for comparator in comparators
    }
    row = ["geometric mean", *["" for _ in dimensions], ""]
    row.extend(f"{geomeans[item['id']]:.6g}" for item in comparators)
    row.extend(
        f"{geomeans[item['id']] / geomeans[candidate['id']]:.4f}x" for item in ratios
    )
    lines.append("| " + " | ".join(_escape_cell(item) for item in row) + " |")
    return "\n".join(lines)


def render_pr(data: dict[str, Any], data_path: Path) -> RenderedPR:
    tilefoundry_description = _validate(data, data_path)
    title = (
        f"[{data['type'].strip()}][foundry][{data['scope'].strip()}] "
        f"{data['subject'].strip()}"
    )

    environment = "\n".join(
        f"| {_escape_cell(key)} | {_escape_cell(value)} |"
        for key, value in data["environment"].items()
    )
    summary = "\n".join(f"- {item}" for item in data["summary"])
    limitations = "\n".join(f"- {item}" for item in data["limitations"])

    body = f"""## Summary

{summary}

## TileFoundry Description

```python
{tilefoundry_description}
```

## Performance

Operator: `{data['operator']}`

| Environment | Value |
| --- | --- |
{environment}

Method: {data['method']}

{_performance_table(data)}

## Result And Limitations

**{data['classification']}**

{limitations}
"""
    _validate_public_body(body)
    return RenderedPR(title=title, body=body)
