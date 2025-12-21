from __future__ import annotations

import subprocess

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


def list_tools() -> list[dict]:
    return [
        {
            "name": "git_status",
            "description": "Show git status (porcelain v1 + branch)",
            "args_schema": {},
        },
        {
            "name": "git_diff",
            "description": "Show git diff (optionally truncated)",
            "args_schema": {"limit": "int?"},
        },
        {
            "name": "repo_search",
            "description": "Search in repo (grep)",
            "args_schema": {"query": "string", "max_matches": "int?"},
        },
        {
            "name": "recent_errors",
            "description": "Recent run errors from DB",
            "args_schema": {"limit": "int?"},
        },
        {
            "name": "pytest",
            "description": "Run pytest -q",
            "args_schema": {},
        },
    ]


def _run_pytest() -> dict:
    cmd = ["python", "-m", "pytest", "-q"]
    try:
        code, out, err = _run_cmd(cmd, timeout_s=300)
    except subprocess.TimeoutExpired:
        return {
            "return_code": "timeout",
            "stdout": "",
            "stderr": "Timeout (300s)",
        }
    return {
        "return_code": code,
        "stdout": _truncate(out.strip()),
        "stderr": _truncate(err.strip()),
    }


def run_tool(name: str, args: dict) -> dict:
    if not isinstance(args, dict):
        raise ValueError("args must be an object")

    try:
        if name == "git_status":
            data = rt.git_status()
            return {"ok": True, "data": data}
        if name == "git_diff":
            limit = args.get("limit", 8000)
            if not isinstance(limit, int):
                raise ValueError("limit must be int")
            data = rt.git_diff(limit=limit)
            return {"ok": True, "data": data}
        if name == "repo_search":
            query = args.get("query")
            if not isinstance(query, str):
                raise ValueError("query is required")
            max_matches = args.get("max_matches", 50)
            if not isinstance(max_matches, int):
                raise ValueError("max_matches must be int")
            data = rt.repo_search(query, max_matches=max_matches)
            return {"ok": True, "data": data}
        if name == "recent_errors":
            limit = args.get("limit", 10)
            if not isinstance(limit, int):
                raise ValueError("limit must be int")
            data = get_recent_errors(limit=limit)
            return {"ok": True, "data": data}
        if name == "pytest":
            data = _run_pytest()
            return {"ok": True, "data": data}
        raise ValueError("unknown tool")
    except ValueError:
        raise
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
