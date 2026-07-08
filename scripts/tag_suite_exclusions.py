"""Fold physics-smoke + init-state failures into a suite manifest as exclusion tags.

The ungated full pool keeps every generated task in the BDDL set (regenerating after
exclusions would flip layout variants of later tasks and stale their .pruned_init).
Instead, tasks that fail any policy-independent feasibility check are tagged
``excluded: true`` + ``exclusion_reason`` in manifest.json, and eval runs only the
feasible task ids.

Inputs (all produced by the Phase-2 pipeline):
  - smoke_report(_shardIofN).json  from scripts/smoke_check_suite.py
  - init_failures_shardIofN.json   from scripts/create_suite_init_states.py
  - missing .pruned_init files     (init shard crash / never ran)

Outputs:
  - manifest.json updated in place (backup written alongside as manifest.json.bak)
  - exclusion_report.json next to the manifest (reasons, per-check detail)
  - feasible_task_ids.txt (comma-joinable id list for the eval clients)

Usage:
    python scripts/tag_suite_exclusions.py \
        --manifest libero/libero/bddl_files/libero_object_unseen_full/manifest.json \
        --smoke-reports 'libero/libero/bddl_files/libero_object_unseen_full/smoke_report*.json' \
        --init-failures 'libero/libero/init_files/libero_object_unseen_full/init_failures_*.json'
"""

import argparse
import glob
import json
import os
import shutil


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--smoke-reports", nargs="+", required=True,
                    help="glob(s) for smoke_report JSONs (shards merge natively)")
    ap.add_argument("--init-failures", nargs="+", default=[],
                    help="glob(s) for init_failures_*.json from create_suite_init_states.py")
    ap.add_argument("--init-dir", default=None,
                    help="default: libero/libero/init_files/<suite>")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    init_dir = args.init_dir or os.path.join(
        repo, "libero/libero/init_files", manifest["suite"])

    smoke = {}  # task name -> result record
    for pattern in args.smoke_reports:
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                for rec in json.load(f)["results"]:
                    smoke[rec["name"]] = rec

    init_fail = {}  # task name -> error string
    for pattern in args.init_failures:
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                for rec in json.load(f):
                    init_fail[rec["name"]] = rec["error"]

    excluded, feasible_ids = [], []
    for task_id, task in enumerate(manifest["tasks"]):
        name = task["name"]
        reasons = []
        if name in init_fail:
            reasons.append(f"init_states_failed: {init_fail[name]}")
        elif not os.path.isfile(os.path.join(init_dir, name + ".pruned_init")):
            reasons.append("init_states_missing (shard never produced the file)")
        rec = smoke.get(name)
        if rec is None:
            reasons.append("smoke_not_run")
        elif not rec["ok"]:
            reasons.append(f"smoke_failed: {rec['error']}")
        if reasons:
            task["excluded"] = True
            task["exclusion_reason"] = "; ".join(reasons)
            excluded.append({"task_id": task_id, "name": name,
                             "target_key": task["target_key"],
                             "target_category": task["target_category"],
                             "reasons": reasons,
                             "smoke_checks": (rec or {}).get("checks")})
        else:
            task.pop("excluded", None)
            task.pop("exclusion_reason", None)
            feasible_ids.append(task_id)

    out_dir = os.path.dirname(os.path.abspath(args.manifest))
    shutil.copy2(args.manifest, args.manifest + ".bak")
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(out_dir, "exclusion_report.json"), "w") as f:
        json.dump({"suite": manifest["suite"], "n_tasks": len(manifest["tasks"]),
                   "n_excluded": len(excluded), "n_feasible": len(feasible_ids),
                   "excluded": excluded}, f, indent=2)
    with open(os.path.join(out_dir, "feasible_task_ids.txt"), "w") as f:
        f.write(",".join(str(i) for i in feasible_ids) + "\n")

    print(f"[tagged] {len(feasible_ids)} feasible / {len(excluded)} excluded "
          f"of {len(manifest['tasks'])} tasks")
    print(f"[out] exclusion_report.json + feasible_task_ids.txt in {out_dir}")


if __name__ == "__main__":
    main()
