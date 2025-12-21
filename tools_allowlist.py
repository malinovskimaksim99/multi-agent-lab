from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any, Callable

import repo_tools as rt
from db import get_recent_errors


def _truncate(text: str, limit: int = 8000) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _run_cmd(cmd: list[str], timeout_s: int = 20) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _run_pytest() -> dict:
    cmd = ["python", "-m", "pytest", "-q"]
    try:
        code, out, err = _run_cmd(cmd, timeout_s=120)
    except subprocess.TimeoutExpired:
        return {
            "returncode": "timeout",
            "stdout": "",
            "stderr": "Timeout (120s)",
        }
    return {
        "returncode": code,
        "stdout": _truncate(out.strip()),
        "stderr": _truncate(err.strip()),
    }


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_schema: dict
    arg_specs: dict[str, dict[str, Any]]
    handler: Callable[[dict[str, Any]], Any]


def _validate_args(spec: ToolSpec, args: Any) -> dict[str, Any]:
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ValueError("args must be an object")

    for key in args.keys():
        if key not in spec.arg_specs:
            raise ValueError(f"unknown arg: {key}")

    validated: dict[str, Any] = {}
    for key, info in spec.arg_specs.items():
        kind = info.get("type")
        value = args.get(key)
        if value is None and "default" in info:
            value = info["default"]

        if kind == "int?":
            if value is None:
                validated[key] = None
            else:
                if not isinstance(value, int):
                    raise ValueError(f"{key} must be int")
                if value <= 0:
                    raise ValueError(f"{key} must be > 0")
                validated[key] = value
        elif kind == "string":
            if not isinstance(value, str):
                raise ValueError(f"{key} must be string")
            value = value.strip()
            if key == "query" and len(value) < 2:
                raise ValueError("query too short")
            if not value:
                raise ValueError(f"{key} is required")
            validated[key] = value
        else:
            raise ValueError("invalid args schema")

    return validated


def _tool_git_status(_args: dict[str, Any]) -> dict:
    return rt.git_status()


def _tool_git_diff(args: dict[str, Any]) -> dict:
    limit = args.get("limit", 8000)
    if limit is None:
        limit = 8000
    return rt.git_diff(limit=limit)


def _tool_repo_search(args: dict[str, Any]) -> dict:
    return rt.repo_search(query=args["query"], max_matches=args.get("max_matches", 50))


def _tool_recent_errors(args: dict[str, Any]) -> list[dict]:
    return get_recent_errors(limit=args.get("limit", 10))


def _tool_pytest(_args: dict[str, Any]) -> dict:
    return _run_pytest()


_TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="git_status",
        description="Show git status (porcelain v1 + branch)",
        args_schema={},
        arg_specs={},
        handler=_tool_git_status,
    ),
    ToolSpec(
        name="git_diff",
        description="Show git diff (optionally truncated)",
        args_schema={"limit": "int?"},
        arg_specs={
            "limit": {"type": "int?", "default": 8000},
        },
        handler=_tool_git_diff,
    ),
    ToolSpec(
        name="repo_search",
        description="Search in repo (grep)",
        args_schema={"query": "string", "max_matches": "int?"},
        arg_specs={
            "query": {"type": "string", "default": ""},
            "max_matches": {"type": "int?", "default": 50},
        },
        handler=_tool_repo_search,
    ),
    ToolSpec(
        name="recent_errors",
        description="Recent run errors from DB",
        args_schema={"limit": "int?"},
        arg_specs={
            "limit": {"type": "int?", "default": 10},
        },
        handler=_tool_recent_errors,
    ),
    ToolSpec(
        name="pytest",
        description="Run pytest -q",
        args_schema={},
        arg_specs={},
        handler=_tool_pytest,
    ),
]

_TOOL_BY_NAME: dict[str, ToolSpec] = {tool.name: tool for tool in _TOOLS}


def list_tools() -> list[dict]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "args_schema": tool.args_schema,
        }
        for tool in _TOOLS
    ]


def run_tool(name: str, args: dict) -> dict:
    tool = _TOOL_BY_NAME.get(name)
    if tool is None:
        raise ValueError(f"unknown tool: {name}")

    validated_args = _validate_args(tool, args)

    try:
        payload = tool.handler(validated_args)
        return {"ok": True, "data": payload}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
