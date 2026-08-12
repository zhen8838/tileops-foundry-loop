"""Validate that a round materially changed an executed TileLang kernel body."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path


class KernelDiffError(ValueError):
    """The declared kernel body does not establish a performance-PR change."""


_ALLOWED_CHANGE_ROOTS = ("src/tileops/kernels/", "tests/kernels/")


def _git_source(repo: Path, revision: str, path: str) -> str | None:
    process = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode == 0:
        return process.stdout
    if "does not exist" in process.stderr or "exists on disk, but not in" in process.stderr:
        return None
    raise KernelDiffError(process.stderr.strip() or f"cannot read {revision}:{path}")


def _changed_paths(repo: Path, base: str, head: str) -> list[str]:
    process = subprocess.run(
        ["git", "diff", "--name-only", "-z", base, head],
        cwd=repo,
        check=False,
        capture_output=True,
    )
    if process.returncode != 0:
        detail = process.stderr.decode(errors="replace").strip()
        raise KernelDiffError(detail or f"cannot compare {base} to {head}")
    return [value.decode(errors="replace") for value in process.stdout.split(b"\0") if value]


def _check_change_scope(repo: Path, base: str, head: str) -> None:
    changed = _changed_paths(repo, base, head)
    disallowed = [path for path in changed if not path.startswith(_ALLOWED_CHANGE_ROOTS)]
    if disallowed:
        rendered = ", ".join(disallowed)
        raise KernelDiffError(
            "performance branch changes files outside kernel implementation and kernel "
            f"correctness tests: {rendered}"
        )


def _decorator_name(decorator: ast.expr) -> str:
    if isinstance(decorator, ast.Call):
        return _decorator_name(decorator.func)
    if isinstance(decorator, ast.Attribute):
        prefix = _decorator_name(decorator.value)
        return f"{prefix}.{decorator.attr}" if prefix else decorator.attr
    if isinstance(decorator, ast.Name):
        return decorator.id
    return ""


def _kernel_body(source: str | None, symbol: str, revision: str, path: str) -> str | None:
    if source is None:
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise KernelDiffError(f"cannot parse {revision}:{path}: {error}") from error
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == symbol
        and any(
            _decorator_name(decorator).endswith((".prim_func", ".macro"))
            or _decorator_name(decorator) in {"prim_func", "macro"}
            for decorator in node.decorator_list
        )
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise KernelDiffError(f"{revision}:{path} contains multiple decorated kernels named {symbol}")
    return ast.dump(matches[0], include_attributes=False)


def check_kernel_diff(repo: Path, base: str, head: str, declaration: str) -> None:
    _check_change_scope(repo, base, head)
    try:
        path, symbol = declaration.rsplit(":", 1)
    except ValueError as error:
        raise KernelDiffError("kernel must be declared as relative-path:symbol") from error
    if not path.startswith("src/tileops/kernels/") or not path.endswith(".py"):
        raise KernelDiffError("kernel path must be a Python file under src/tileops/kernels/")
    if not symbol.isidentifier():
        raise KernelDiffError("kernel symbol must be a Python identifier")

    base_body = _kernel_body(_git_source(repo, base, path), symbol, base, path)
    head_body = _kernel_body(_git_source(repo, head, path), symbol, head, path)
    if head_body is None:
        raise KernelDiffError(f"declared @T.prim_func/@T.macro {symbol} is absent at {head}:{path}")
    if base_body == head_body:
        raise KernelDiffError(f"declared kernel body {path}:{symbol} is unchanged from {base} to {head}")
