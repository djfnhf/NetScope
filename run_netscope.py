#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence

from netscope.tasks import TASK_INDEX, TASKS, Task


ROOT = Path(__file__).resolve().parent
COMMANDS = {"list", "search", "show", "run", "run-group", "doctor"}


def _strip_delimiter(args: List[str]) -> List[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def _normalize_argv(argv: Sequence[str]) -> List[str]:
    """Allow direct execution: `python run_netscope.py <task_id> -- ...`."""
    if not argv:
        return list(argv)
    first = argv[0]
    if first in COMMANDS:
        return list(argv)
    if first in TASK_INDEX:
        return ["run", first, *argv[1:]]
    return list(argv)


def _run_one(task: Task, script_args: List[str]) -> int:
    script_path = task.script_path
    if not script_path.exists():
        print(f"[ERROR] Script not found for {task.task_id}: {script_path}")
        return 2

    cmd = [sys.executable, str(script_path), *_strip_delimiter(script_args)]
    print(f"\n[RUN] {task.task_id}")
    print(f"[DIM] {task.paper_dimension}")
    print("[CMD] " + " ".join(cmd))

    completed = subprocess.run(cmd, cwd=str(ROOT))
    print(f"[EXIT] {completed.returncode}")
    return int(completed.returncode)


def _filter_tasks(
    *,
    prefix: str | None = None,
    keyword: str | None = None,
    dimension: str | None = None,
) -> List[Task]:
    tasks: Iterable[Task] = TASKS

    if prefix:
        tasks = [t for t in tasks if t.task_id.startswith(prefix)]

    if keyword:
        key = keyword.lower().strip()
        tasks = [
            t
            for t in tasks
            if key in t.task_id.lower()
            or key in t.summary.lower()
            or key in t.paper_dimension.lower()
        ]

    if dimension:
        dim = dimension.lower().strip()
        tasks = [t for t in tasks if dim in t.paper_dimension.lower()]

    return sorted(tasks, key=lambda x: (x.paper_dimension, x.task_id))


def _print_tasks(tasks: Sequence[Task], show_script: bool = False) -> None:
    if not tasks:
        print("[INFO] No tasks matched the current filters.")
        return

    by_dim = defaultdict(list)
    for task in tasks:
        by_dim[task.paper_dimension].append(task)

    for dim in sorted(by_dim.keys()):
        print(f"\n## {dim}")
        for task in sorted(by_dim[dim], key=lambda x: x.task_id):
            print(f"- {task.task_id}: {task.summary}")
            if show_script:
                print(f"  script: {task.script_relpath}")


def _select_by_prefix(prefix: str) -> List[Task]:
    selected = [t for t in TASKS if t.task_id.startswith(prefix)]
    return sorted(selected, key=lambda x: x.task_id)


def _check_scripts(tasks: Sequence[Task]) -> int:
    missing = [t for t in tasks if not t.script_path.exists()]
    if not missing:
        print("[OK] All task scripts exist.")
        return 0

    print(f"[ERROR] {len(missing)} task scripts are missing:")
    for task in missing:
        print(f"- {task.task_id}: {task.script_relpath}")
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    normalized_argv = _normalize_argv(sys.argv[1:] if argv is None else argv)

    parser = argparse.ArgumentParser(
        description="NetScope task launcher for classification, generation, and preprocessing workflows."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    list_cmd = sub.add_parser("list", help="List tasks grouped by paper dimensions.")
    list_cmd.add_argument("--prefix", help="Filter by task id prefix, e.g. cls. or gen.")
    list_cmd.add_argument("--keyword", help="Filter by keyword in id/summary/dimension.")
    list_cmd.add_argument("--dimension", help="Filter by paper dimension substring.")
    list_cmd.add_argument("--show-script", action="store_true", help="Also print script relative paths.")

    show = sub.add_parser("show", help="Show details of one task.")
    show.add_argument("task_id", help="Task identifier, e.g. cls.robustness.perturbation_builder")

    search = sub.add_parser("search", help="Search tasks by free-text query.")
    search.add_argument("query", help="Keyword matched against id, summary, and dimension.")
    search.add_argument("--show-script", action="store_true", help="Also print script relative paths.")

    run = sub.add_parser("run", help="Run one task and pass through extra args to the target script.")
    run.add_argument("task_id", help="Task identifier.")
    run.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed to the target script.")

    run_group = sub.add_parser("run-group", help="Run tasks whose id starts with a prefix.")
    run_group.add_argument("prefix", help="Prefix such as cls., gen., cls.cost., gen.preprocess.")
    run_group.add_argument("--continue-on-error", action="store_true", help="Continue even if one task fails.")
    run_group.add_argument("script_args", nargs=argparse.REMAINDER, help="Same extra args for each task.")

    doctor = sub.add_parser("doctor", help="Validate task registrations and script file existence.")
    doctor.add_argument("--prefix", help="Optionally validate only one task prefix.")

    args = parser.parse_args(normalized_argv)

    if args.command == "list":
        tasks = _filter_tasks(prefix=args.prefix, keyword=args.keyword, dimension=args.dimension)
        _print_tasks(tasks, show_script=args.show_script)
        return 0

    if args.command == "search":
        tasks = _filter_tasks(keyword=args.query)
        _print_tasks(tasks, show_script=args.show_script)
        return 0

    if args.command == "show":
        task = TASK_INDEX.get(args.task_id)
        if task is None:
            print(f"[ERROR] Unknown task_id: {args.task_id}")
            return 2
        print(f"task_id: {task.task_id}")
        print(f"paper_dimension: {task.paper_dimension}")
        print(f"script: {task.script_relpath}")
        print(f"summary: {task.summary}")
        return 0

    if args.command == "run":
        task = TASK_INDEX.get(args.task_id)
        if task is None:
            print(f"[ERROR] Unknown task_id: {args.task_id}")
            return 2
        return _run_one(task, args.script_args)

    if args.command == "run-group":
        tasks = _select_by_prefix(args.prefix)
        if not tasks:
            print(f"[ERROR] No tasks found for prefix: {args.prefix}")
            return 2

        script_args = args.script_args
        for task in tasks:
            code = _run_one(task, script_args)
            if code != 0 and not args.continue_on_error:
                return code
        return 0

    if args.command == "doctor":
        tasks = _select_by_prefix(args.prefix) if args.prefix else list(TASKS)
        if args.prefix and not tasks:
            print(f"[ERROR] No tasks found for prefix: {args.prefix}")
            return 2
        return _check_scripts(tasks)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
