import json
from copy import deepcopy
from collections import Counter

from agents.supervisor import Supervisor
from memory.store import load_memory, save_memory, set_flag


def load_tasks(path="tests/sample_tasks.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("tasks", [])


def run_suite(tasks, memory):
    """Run all tasks with given memory and return tag stats."""
    sup = Supervisor()
    stats = Counter()
    results = []

    for task in tasks:
        res = sup.run(task, memory)
        tags = res.get("critique_tags", []) or []
        stats.update(tags)
        results.append(res)

    return stats, results


def simulate_learning(tasks, memory):
    """
    One simple learning pass:
    if Critic tags 'structure' -> set force_structure flag.
    """
    sup = Supervisor()
    for task in tasks:
        res = sup.run(task, memory)
        tags = res.get("critique_tags", []) or []
        if "structure" in tags:
            set_flag(memory, "force_structure", True)
    return memory


def pct_change(before, after):
    if before == 0 and after == 0:
        return 0.0
    if before == 0 and after > 0:
        return -100.0
    return ((before - after) / before) * 100.0


def main(write_memory=False):
    tasks = load_tasks()
    if not tasks:
        print("No tasks found in tests/sample_tasks.json")
        return

    # Load current memory from disk
    mem_disk = load_memory()

    # ---- Baseline memory (no flags) ----
    mem_base = deepcopy(mem_disk)
    mem_base["flags"] = {}  # clear learned flags

    # ---- Run BEFORE ----
    before_stats, _ = run_suite(tasks, mem_base)

    # ---- Simulate learning in-memory ----
    mem_learned = deepcopy(mem_base)
    mem_learned = simulate_learning(tasks, mem_learned)

    # ---- Run AFTER ----
    after_stats, _ = run_suite(tasks, mem_learned)

    # ---- Report ----
    all_tags = sorted(set(before_stats.keys()) | set(after_stats.keys()))
    print("=== Progress Report ===")
    print(f"Tasks: {len(tasks)}\n")

    if not all_tags:
        print("No critique tags detected in either run.")
    else:
        print("Tag stats (before -> after):")
        for t in all_tags:
            b = before_stats.get(t, 0)
            a = after_stats.get(t, 0)
            change = pct_change(b, a)
            sign = "+" if change >= 0 else ""
            print(f"  {t}: {b} -> {a}  ({sign}{change:.1f}% improvement)")

    print("\nLearned flags (simulation):")
    print(mem_learned.get("flags", {}))

    # Optionally persist learned memory to disk
    if write_memory:
        save_memory(mem_learned)
        print("\nSaved learned memory to memory/memory.json")


if __name__ == "__main__":
    # Change to True if you want to overwrite memory.json
    main(write_memory=False)
