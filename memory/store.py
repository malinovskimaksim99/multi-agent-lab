import json
from pathlib import Path
from typing import Any, Dict

MEMORY_PATH = Path("memory/memory.json")

def load_memory() -> Dict[str, Any]:
    if MEMORY_PATH.exists():
        return json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    return {"rules": [], "examples": []}

def save_memory(mem: Dict[str, Any]) -> None:
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_PATH.write_text(
        json.dumps(mem, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def add_rule(mem: Dict[str, Any], rule: str) -> None:
    if rule not in mem["rules"]:
        mem["rules"].append(rule)
