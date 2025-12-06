import json
from pathlib import Path
from typing import Any, Dict

MEMORY_PATH = Path("memory/memory.json")

DEFAULT_MEMORY = {
    "rules": [],
    "examples": [],
    "flags": {}
}

def load_memory() -> Dict[str, Any]:
    if MEMORY_PATH.exists():
        data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
        # backward compatible merge
        merged = {**DEFAULT_MEMORY, **data}
        merged["flags"] = {**DEFAULT_MEMORY["flags"], **(data.get("flags") or {})}
        merged["rules"] = data.get("rules") or []
        merged["examples"] = data.get("examples") or []
        return merged
    return DEFAULT_MEMORY.copy()

def save_memory(mem: Dict[str, Any]) -> None:
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_PATH.write_text(
        json.dumps(mem, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def add_rule(mem: Dict[str, Any], rule: str) -> None:
    if rule not in mem["rules"]:
        mem["rules"].append(rule)

def set_flag(mem: Dict[str, Any], key: str, value: Any = True) -> None:
    mem.setdefault("flags", {})
    mem["flags"][key] = value
