from __future__ import annotations

import subprocess
from typing import Tuple, Dict


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


def _cmd_error(cmd_str: str, code: int, out: str, err: str) -> str:
    out = _truncate(out.strip())
    err = _truncate(err.strip())
    if err:
        return f"{cmd_str} failed (code={code}): {err}"
    if out:
        return f"{cmd_str} failed (code={code}): {out}"
    return f"{cmd_str} failed (code={code})"


def git_status() -> dict:
    cmd = ["git", "status", "--porcelain=v1", "-b"]
    code, out, err = _run_cmd(cmd, timeout_s=20)
    if code != 0:
        raise RuntimeError(_cmd_error("git status", code, out, err))
    return {
        "stdout": _truncate(out.strip()),
        "stderr": _truncate(err.strip()),
    }


def git_diff(limit: int = 8000) -> dict:
    cmd = ["git", "diff"]
    code, out, err = _run_cmd(cmd, timeout_s=20)
    if code != 0:
        raise RuntimeError(_cmd_error("git diff", code, out, err))

    if len(out) > limit:
        stat_code, stat_out, stat_err = _run_cmd(["git", "diff", "--stat"], timeout_s=20)
        if stat_code != 0:
            raise RuntimeError(_cmd_error("git diff --stat", stat_code, stat_out, stat_err))
        return {
            "truncated": True,
            "stat": _truncate(stat_out.strip()),
            "stderr": _truncate(stat_err.strip()),
        }

    return {
        "truncated": False,
        "diff": _truncate(out.strip(), limit=limit),
        "stderr": _truncate(err.strip()),
    }


def repo_search(query: str, max_matches: int = 50) -> dict:
    q = (query or "").strip()
    if len(q) < 2:
        raise ValueError("query too short")

    cmd = [
        "grep",
        "-R",
        "--line-number",
        "--binary-files=without-match",
        "--exclude-dir=.git",
        "--exclude-dir=.venv",
        "-m",
        str(max_matches),
        "--",
        q,
        ".",
    ]
    code, out, err = _run_cmd(cmd, timeout_s=20)
    if code == 1:
        return {
            "found": False,
            "matches": "",
        }
    if code != 0:
        raise RuntimeError(_cmd_error("grep", code, out, err))
    return {
        "found": True,
        "matches": _truncate(out.strip()),
        "stderr": _truncate(err.strip()),
    }
