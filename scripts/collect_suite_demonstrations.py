"""Suite-wide scripted demo collection for manifest-generated LIBERO suites.

Runs ``scripts/collect_scripted_demonstrations.py`` over every feasible task of a suite
(default: ``libero_object_unseen_stockbg``), in parallel, with per-task logs and
``result.json`` resumability -- the collect-stage structure proven in
``run_scale_pipeline.py``, re-keyed to ``manifest["tasks"]`` + ``feasible_task_ids.txt``.

The primary success metric is **per-category coverage**: the report rolls demos up by
``target_category`` and lists ``categories_with_zero_demos`` explicitly. An optional
``--escalate`` pass re-attacks exactly those categories with bigger budgets (more
sampler candidates, a relaxed friction cone, doubled attempt caps) and, for tasks whose
environment could not even build/reset, a freshly seeded init-state pool generated via
``scripts/create_suite_init_states.py`` (kept in ``<out-root>/train_init`` -- train-time
initial conditions stay disjoint from the eval ``.pruned_init`` pools).

Collection is low-dim only (no GPU); rendering stays a separate, later stage.

Examples:
    # 110-category pilot: one task per category
    python scripts/collect_suite_demonstrations.py \
        --out-root /tmp2/.../stockbg_demos --per-category 1 --workers 16

    # full run + escalation
    python scripts/collect_suite_demonstrations.py \
        --out-root /tmp2/.../stockbg_demos --workers 16 --escalate
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PY = sys.executable  # run children with the same (libero) interpreter
_DEFAULT_MANIFEST = os.path.join(
    _REPO, "libero", "libero", "bddl_files", "libero_object_unseen_stockbg",
    "manifest.json")


def sh(cmd, log_path=None, env=None, timeout=None):
    """Run a subprocess, tee combined output to log_path; return (rc, tail).

    ``timeout`` (seconds) guards against a single task hanging the whole run: a rollout
    can rarely stall inside MuJoCo's contact solver on degenerate geometry (observed on
    kettle/steak — a rollout frozen for 17 h). On timeout the child (and its process
    group) is killed and rc=-1 is returned so the task is recorded as failed and the
    orchestrator moves on."""
    e = dict(os.environ)
    if env:
        e.update(env)
    with open(log_path, "w") if log_path else open(os.devnull, "w") as lf:
        try:
            p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=e,
                               cwd=_REPO, timeout=timeout, start_new_session=True)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            lf.write(f"\n[orchestrator] killed: exceeded {timeout}s timeout\n")
            lf.flush()
            rc = -1
    tail = ""
    if log_path and os.path.isfile(log_path):
        with open(log_path) as f:
            tail = "".join(f.readlines()[-8:])
    return rc, tail


def load_tasks(manifest_path, feasible_ids_path):
    manifest = json.load(open(manifest_path))
    bddl_dir = os.path.dirname(os.path.abspath(manifest_path))
    all_tasks = manifest["tasks"]
    tasks = [dict(t) for t in all_tasks if not t.get("excluded")]
    if feasible_ids_path and os.path.isfile(feasible_ids_path):
        toks = [tok for tok in open(feasible_ids_path).read().replace(",", " ").split()
                if tok]
        if toks and all(tok.isdigit() for tok in toks):
            # the stockbg format: comma-separated indices into manifest["tasks"]
            feasible = {all_tasks[int(tok)]["name"] for tok in toks}
        else:
            feasible = set(toks)
        tasks = [t for t in tasks if t["name"] in feasible]
    for t in tasks:
        t["bddl"] = os.path.join(bddl_dir, t["name"] + ".bddl")
    missing = [t["name"] for t in tasks if not os.path.isfile(t["bddl"])]
    if missing:
        raise SystemExit(f"{len(missing)} manifest tasks have no .bddl on disk, e.g. "
                         + missing[0])
    return tasks


def select_tasks(tasks, select, limit, per_category):
    if select:
        subs = [s for s in select.split(",") if s.strip()]
        tasks = [t for t in tasks if any(s in t["name"] for s in subs)]
    if per_category:
        by_cat, kept = defaultdict(int), []
        for t in tasks:
            if by_cat[t["target_category"]] < per_category:
                by_cat[t["target_category"]] += 1
                kept.append(t)
        tasks = kept
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


def read_result(out_root, name):
    rj = os.path.join(out_root, "collect", name, "result.json")
    if not os.path.isfile(rj):
        return None
    try:
        return json.load(open(rj))
    except (json.JSONDecodeError, OSError):
        return None


def run_pass(tasks, args, escalated=False, init_dir=None):
    """One parallel collection pass; skips tasks whose result.json already exists
    (base pass) or already has demos (escalation pass)."""
    tag = "escalate" if escalated else "collect"

    def collect_one(task):
        name = task["name"]
        out_dir = os.path.join(args.out_root, "collect", name)
        rj = os.path.join(out_dir, "result.json")
        prior = read_result(args.out_root, name)
        if not escalated and prior is not None:
            return name, "skip"
        if escalated:
            if prior is not None and prior.get("n_demos", 0) > 0:
                return name, "skip"
            if prior is not None:  # keep the pre-escalation verdict for the report
                os.replace(rj, os.path.join(out_dir, "result.pre_escalate.json"))
        os.makedirs(out_dir, exist_ok=True)
        cmd = [_PY, "scripts/collect_scripted_demonstrations.py",
               "--bddl-file", task["bddl"], "--out-dir", out_dir,
               "--result-json", rj,
               "--num-success", str(args.num_success),
               "--calib-attempts", str(args.calib_attempts),
               "--seed", str(args.seed)]
        if escalated:
            cmd += ["--max-attempts", str(2 * args.max_attempts),
                    "--top-k-candidates", "8", "--mu", "0.8",
                    "--build-tries", "40"]
            pool = init_dir and os.path.join(init_dir, name + ".pruned_init")
            if pool and os.path.isfile(pool):
                cmd += ["--init-states", pool]
        else:
            cmd += ["--max-attempts", str(args.max_attempts)]
        if args.collector_args:
            cmd += args.collector_args.split()
        rc, _ = sh(cmd, log_path=os.path.join(args.out_root, "logs",
                                              f"{name}.{tag}.log"),
                   timeout=args.task_timeout)
        res = read_result(args.out_root, name)
        if res is None:
            return name, ("timeout" if rc == -1 else f"fail(rc={rc})")
        return name, f"demos={res.get('n_demos', 0)}"

    t0 = time.time()
    print(f"[stage] {tag} ({len(tasks)} tasks, {args.workers} workers)")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(collect_one, t): t["name"] for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            name, status = fut.result()
            print(f"  [{tag} {i}/{len(futs)}] {name}: {status}", flush=True)
    print(f"[stage] {tag} done in {(time.time() - t0) / 60:.1f} min")


def regenerate_init_pools(tasks, args):
    """Fresh-seed init pools for tasks whose env could not build/reset (plan step 1).

    Writes a filtered temp manifest and runs the existing create_suite_init_states.py
    into <out-root>/train_init, so train-time initial conditions never reuse the eval
    ``.pruned_init`` files."""
    init_dir = os.path.join(args.out_root, "train_init")
    todo = [t for t in tasks
            if not os.path.isfile(os.path.join(init_dir, t["name"] + ".pruned_init"))]
    if not todo:
        return init_dir
    os.makedirs(init_dir, exist_ok=True)
    tmp_manifest = os.path.join(init_dir, "_escalate_manifest.json")
    json.dump({"suite": "train_init", "tasks": [{"name": t["name"]} for t in todo]},
              open(tmp_manifest, "w"))
    # create_suite_init_states resolves bddls next to the manifest -> symlink them in
    bddl_dir = os.path.dirname(os.path.abspath(args.manifest))
    for t in todo:
        link = os.path.join(init_dir, t["name"] + ".bddl")
        if not os.path.exists(link):
            os.symlink(os.path.join(bddl_dir, t["name"] + ".bddl"), link)
    print(f"[init] regenerating {len(todo)} init pools (seed {args.init_seed}) "
          f"-> {init_dir}")
    sh([_PY, "scripts/create_suite_init_states.py", "--manifest", tmp_manifest,
        "--out-dir", init_dir, "--num-states", "25", "--seed", str(args.init_seed),
        "--overwrite"],
       log_path=os.path.join(args.out_root, "logs", "train_init.log"))
    return init_dir


def write_report(tasks, args):
    per_cat = defaultdict(lambda: {"n_tasks": 0, "n_attempted": 0, "n_with_demos": 0,
                                   "n_demos": 0, "n_failed_runs": 0,
                                   "best_collect_yield": 0.0})
    rows = []
    for t in tasks:
        cat = t["target_category"]
        c = per_cat[cat]
        c["n_tasks"] += 1
        res = read_result(args.out_root, t["name"])
        row = {"name": t["name"], "category": cat, "n_demos": 0, "status": "missing"}
        if res is None:
            log = os.path.join(args.out_root, "logs", f"{t['name']}.collect.log")
            if os.path.isfile(log):
                c["n_failed_runs"] += 1
                row["status"] = "crashed"
        else:
            c["n_attempted"] += 1
            n = int(res.get("n_demos", 0))
            row.update(n_demos=n, status="ok" if n else "zero_demos",
                       collect_yield=res.get("collect_yield"),
                       chosen=res.get("chosen_entry", {}) or {},
                       escalated=os.path.isfile(os.path.join(
                           args.out_root, "collect", t["name"],
                           "result.pre_escalate.json")))
            c["n_demos"] += n
            c["n_with_demos"] += int(n > 0)
            c["best_collect_yield"] = max(c["best_collect_yield"],
                                          res.get("collect_yield") or 0.0)
        rows.append(row)

    zero_cats = sorted(c for c, v in per_cat.items() if v["n_demos"] == 0)
    report = {
        "suite": os.path.basename(os.path.dirname(os.path.abspath(args.manifest))),
        "n_tasks": len(tasks),
        "n_tasks_with_demos": sum(r["n_demos"] > 0 for r in rows),
        "total_demos": sum(r["n_demos"] for r in rows),
        "n_categories": len(per_cat),
        "n_categories_with_demos": len(per_cat) - len(zero_cats),
        "categories_with_zero_demos": zero_cats,
        "per_category": {c: per_cat[c] for c in sorted(per_cat)},
        "tasks": rows,
    }
    rp = os.path.join(args.out_root, "suite_collect_report.json")
    json.dump(report, open(rp, "w"), indent=2)

    md = [f"# Suite collection report: {report['suite']}", "",
          f"- tasks: {report['n_tasks_with_demos']}/{report['n_tasks']} with demos; "
          f"{report['total_demos']} demos total",
          f"- categories: {report['n_categories_with_demos']}/{report['n_categories']}"
          f" with demos",
          f"- categories with ZERO demos: "
          f"{', '.join(zero_cats) if zero_cats else 'none'}", "",
          "| category | tasks | with demos | demos | best yield % |",
          "| :--- | ---: | ---: | ---: | ---: |"]
    for c in sorted(per_cat):
        v = per_cat[c]
        md.append(f"| {c} | {v['n_tasks']} | {v['n_with_demos']} | {v['n_demos']} | "
                  f"{v['best_collect_yield']:.0f} |")
    with open(os.path.join(args.out_root, "suite_collect_report.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[report] {report['n_tasks_with_demos']}/{report['n_tasks']} tasks, "
          f"{report['n_categories_with_demos']}/{report['n_categories']} categories, "
          f"{report['total_demos']} demos -> {rp}")
    if zero_cats:
        print(f"[report] categories with ZERO demos ({len(zero_cats)}): "
              + ", ".join(zero_cats))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=_DEFAULT_MANIFEST)
    ap.add_argument("--feasible-ids", default=None,
                    help="default: feasible_task_ids.txt next to the manifest")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--workers", type=int, default=16,
                    help="parallel collector processes (each ~1 GB RAM)")
    ap.add_argument("--num-success", type=int, default=50)
    ap.add_argument("--max-attempts", type=int, default=100)
    ap.add_argument("--calib-attempts", type=int, default=5)
    ap.add_argument("--task-timeout", type=int, default=2400,
                    help="per-task wall-clock cap (s); a task exceeding it is killed and "
                         "recorded failed so one MuJoCo stall can't hang the whole run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-seed", type=int, default=20260727,
                    help="seed for regenerated train-time init pools (escalation)")
    ap.add_argument("--select", default=None,
                    help="comma-separated substrings on task names (e.g. one object)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--per-category", type=int, default=0,
                    help="N tasks per category (pilot mode); 0 = all")
    ap.add_argument("--escalate", action="store_true",
                    help="second pass over zero-demo categories: bigger budgets, "
                         "relaxed cone, fresh init pools for build failures")
    ap.add_argument("--skip-collect", action="store_true",
                    help="only (re)write the report from existing results")
    ap.add_argument("--collector-args", default=None,
                    help="extra args passed through to every collector run, e.g. "
                         "'--no-align-yaw'")
    args = ap.parse_args()

    feasible = args.feasible_ids or os.path.join(
        os.path.dirname(os.path.abspath(args.manifest)), "feasible_task_ids.txt")
    tasks = select_tasks(load_tasks(args.manifest, feasible),
                         args.select, args.limit, args.per_category)
    if not tasks:
        raise SystemExit("no tasks selected")
    n_cats = len({t["target_category"] for t in tasks})
    os.makedirs(os.path.join(args.out_root, "logs"), exist_ok=True)
    print(f"[run] {len(tasks)} tasks / {n_cats} categories | out-root={args.out_root}")

    if not args.skip_collect:
        run_pass(tasks, args, escalated=False)

    report = write_report(tasks, args)

    if args.escalate and report["categories_with_zero_demos"]:
        zero = set(report["categories_with_zero_demos"])
        esc_tasks = [t for t in tasks if t["target_category"] in zero]
        # fresh init pools for the subset whose env never produced a result at all
        crashed = [t for t in esc_tasks if read_result(args.out_root, t["name"]) is None]
        init_dir = regenerate_init_pools(crashed, args) if crashed else None
        run_pass(esc_tasks, args, escalated=True, init_dir=init_dir)
        write_report(tasks, args)


if __name__ == "__main__":
    main()
