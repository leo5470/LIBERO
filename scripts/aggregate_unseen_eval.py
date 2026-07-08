"""Aggregate policy-eval results into per-object-category success-rate tables.

Consumes the per-task-incremental results JSONs written by the patched eval clients
(openpi examples/libero/main.py and cosmos-policy run_libero_eval.py; shared schema:
``{model, suite, num_trials_per_task, tasks: [{task_id, task_name, language, episodes,
successes, episode_results: [{episode_idx, success, ...}]}], ...}``) and joins them
against the libero_object_unseen manifest (task -> target_category). Stock
libero_object task names are parsed for their category directly (seen control rows).

Outputs (to --out-dir):
  category_success_rates.csv  one row per object category, one SR column per model
  failure_index.csv           one row per failed episode, for failure-case drill-down
  summary.json                full join (per-model overall/per-category/per-task)

Example:
    python scripts/aggregate_unseen_eval.py --results /tmp2/leocheng/eval_results/*.json
"""

import argparse
import csv
import glob
import json
import os
import re

STOCK_NAME_RE = re.compile(r"pick_up_the_(.+?)_and_place_it_in_the_basket")


def category_of_task(task_name, unseen_categories):
    if task_name in unseen_categories:
        return unseen_categories[task_name]
    m = STOCK_NAME_RE.fullmatch(task_name)
    return m.group(1) if m else task_name


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True,
                    help="results JSON files (globs ok)")
    ap.add_argument("--manifest", default=os.path.join(
        repo, "libero/libero/bddl_files/libero_object_unseen/manifest.json"))
    ap.add_argument("--out-dir", default=None,
                    help="default: directory of the first results file")
    args = ap.parse_args()

    paths = sorted({p for pattern in args.results for p in glob.glob(pattern)})
    assert paths, f"no results files match {args.results}"
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(paths[0]))
    os.makedirs(out_dir, exist_ok=True)

    with open(args.manifest) as f:
        manifest = json.load(f)
    unseen_categories = {t["name"]: t["target_category"] for t in manifest["tasks"]}

    # (suite, category) -> model -> {episodes, successes, tasks:set}
    cells = {}
    models = []
    failures = []
    per_task = {}  # (suite, task_name) -> model -> {episodes, successes}
    overall = {}   # model -> suite -> {episodes, successes}

    for path in paths:
        with open(path) as f:
            run = json.load(f)
        model, suite = run["model"], run["suite"]
        if model not in models:
            models.append(model)
        for task in run["tasks"]:
            category = category_of_task(task["task_name"], unseen_categories)
            cell = cells.setdefault((suite, category), {}).setdefault(
                model, {"episodes": 0, "successes": 0, "tasks": set()})
            cell["episodes"] += task["episodes"]
            cell["successes"] += task["successes"]
            cell["tasks"].add(task["task_name"])
            pt = per_task.setdefault((suite, task["task_name"]), {})
            pt[model] = {"episodes": task["episodes"], "successes": task["successes"]}
            ov = overall.setdefault(model, {}).setdefault(
                suite, {"episodes": 0, "successes": 0})
            ov["episodes"] += task["episodes"]
            ov["successes"] += task["successes"]
            for ep in task.get("episode_results", []):
                if not ep.get("success"):
                    failures.append({
                        "model": model,
                        "suite": suite,
                        "category": category,
                        "task_name": task["task_name"],
                        "episode_idx": ep.get("episode_idx"),
                        "steps": ep.get("steps", ""),
                        "video_hint": f"*{suite}_task{task['task_id']}_ep{ep.get('episode_idx')}_*failure.mp4",
                    })

    def rate(cell):
        return cell["successes"] / cell["episodes"] if cell and cell["episodes"] else None

    def mean_rate(model_cells):
        rates = [rate(c) for c in model_cells.values() if c["episodes"]]
        return sum(rates) / len(rates) if rates else 1.0

    # unseen rows first, worst mean SR first; seen-control rows after
    rows = sorted(
        cells.items(),
        key=lambda kv: (kv[0][0] != "libero_object_unseen", mean_rate(kv[1]), kv[0][1]),
    )

    csv_path = os.path.join(out_dir, "category_success_rates.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["suite", "category", "n_tasks", "n_episodes"] + models)
        for (suite, category), model_cells in rows:
            n_tasks = max(len(c["tasks"]) for c in model_cells.values())
            n_episodes = max(c["episodes"] for c in model_cells.values())
            writer.writerow(
                [suite, category, n_tasks, n_episodes]
                + [
                    f"{rate(model_cells[m]):.3f}" if m in model_cells else ""
                    for m in models
                ]
            )

    fail_path = os.path.join(out_dir, "failure_index.csv")
    failures.sort(key=lambda r: (r["model"], r["suite"], r["category"], r["task_name"],
                                 r["episode_idx"] if r["episode_idx"] is not None else -1))
    with open(fail_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "suite", "category", "task_name", "episode_idx", "steps", "video_hint"])
        writer.writeheader()
        writer.writerows(failures)

    summary = {
        "results_files": paths,
        "models": models,
        "overall": {
            m: {s: {**v, "success_rate": rate(v)} for s, v in suites.items()}
            for m, suites in overall.items()
        },
        "categories": [
            {
                "suite": suite,
                "category": category,
                "models": {
                    m: {"episodes": c["episodes"], "successes": c["successes"],
                        "success_rate": rate(c), "tasks": sorted(c["tasks"])}
                    for m, c in model_cells.items()
                },
            }
            for (suite, category), model_cells in rows
        ],
        "tasks": [
            {"suite": suite, "task_name": name,
             "category": category_of_task(name, unseen_categories),
             "tier": next((t["tier"] for t in manifest["tasks"] if t["name"] == name), None),
             "models": stats}
            for (suite, name), stats in sorted(per_task.items())
        ],
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[csv]     {csv_path}  ({len(rows)} categories x {len(models)} models)")
    print(f"[csv]     {fail_path}  ({len(failures)} failed episodes)")
    print(f"[summary] {summary_path}")
    for m, suites in overall.items():
        for s, v in suites.items():
            print(f"  {m} / {s}: {v['successes']}/{v['episodes']} = {rate(v):.3f}")


if __name__ == "__main__":
    main()
