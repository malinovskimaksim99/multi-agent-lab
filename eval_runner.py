import argparse
import json
from collections import Counter
from pathlib import Path

import agents  # noqa: F401
from agents.supervisor import Supervisor
from memory.store import load_memory

EVAL_DIR = Path("eval")
EVAL_DIR.mkdir(exist_ok=True)


DEFAULT_TASKS = [
    # docs
    "Write a short README section for installation",
    "Create a minimal project README outline",

    # explain
    "Explain planning vs critique",
    "Explain the roles of Planner, Analyst, Critic",
    "Compare single-solver vs team-solver modes",

    # planning
    "Create a short plan to add a new agent safely",
    "Outline next steps to improve our multi-agent lab",

    # general
    "Summarize how our multi-agent pipeline works",
    "List key design principles for agent routing",
]


def run_mode(mode: str, tasks, memory):
    if mode == "auto":
        sup = Supervisor(auto_solver=True, auto_team=False)
    elif mode == "team":
        sup = Supervisor(auto_solver=False, auto_team=True, team_size=2)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    results = []
    tag_counter = Counter()
    tagged_tasks = []

    for task in tasks:
        r = sup.run(task, memory)
        tags = r.get("critique_tags") or []
        tag_counter.update(tags)

        if tags:
            tagged_tasks.append({"task": task, "tags": tags})

        results.append({
            "task": task,
            "solver_agent": r.get("solver_agent"),
            "team_agents": r.get("team_agents"),
            "team_profile": r.get("team_profile"),
            "critique_tags": tags,
        })

    return results, tag_counter, tagged_tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["auto", "team", "both"], default="both")
    parser.add_argument("--tasks-file", type=str, default="")
    args = parser.parse_args()

    memory = load_memory()

    tasks = DEFAULT_TASKS
    if args.tasks_file:
        p = Path(args.tasks_file)
        if p.exists():
            tasks = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

    summary = {"modes": {}, "tasks_count": len(tasks)}

    modes = ["auto", "team"] if args.mode == "both" else [args.mode]

    for m in modes:
        results, tag_counter, tagged_tasks = run_mode(m, tasks, memory)
        summary["modes"][m] = {
            "tag_stats": dict(tag_counter),
            "tagged_tasks": tagged_tasks[:20],
            "results": results,
        }

    out_path = EVAL_DIR / "latest_eval.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Evaluation ===")
    print(f"Tasks: {len(tasks)}")
    print(f"Saved: {out_path}")

    for m in modes:
        print(f"\n--- Mode: {m} ---")
        tags = summary["modes"][m]["tag_stats"]
        if not tags:
            print("No critique tags found.")
        else:
            for k, v in sorted(tags.items(), key=lambda x: (-x[1], x[0])):
                print(f"{k}: {v}")

        tt = summary["modes"][m]["tagged_tasks"]
        if tt:
            print("\nTagged examples:")
            for item in tt:
                print(f"- {item['task']} | tags: {item['tags']}")


if __name__ == "__main__":
    main()
