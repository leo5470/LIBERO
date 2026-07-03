"""Orchestrate the RoboCasa->LIBERO scale run over a scale_pool_manifest.json.

Stages (each resumable; --stages to run a subset):
  convert  -> tools/convert_robocasa_xml.py --manifest  (skip-on-failure + auto-fit-scale)
  bddl     -> scripts/create_robocasa_pickplace_tasks.py (also the register/load gate)
  collect  -> scripts/collect_scripted_demonstrations.py, PARALLEL across CPU workers,
              gated on reached_target (>=num_success within max_attempts)
  render   -> scripts/create_dataset.py --use-camera-obs --compress, SERIALIZED (shared GPU),
              output redirected to --out-root (NOT the default /home datasets path)
  report   -> scale_pool_report.{json,md} + tools/make_tier_split.py -> tier_split.json

Collect writes low-dim demos to <out-root>/collect/<key>/, rendered image HDF5s to
<out-root>/rendered/, logs to <out-root>/logs/. Re-running skips keys that already have a
result.json (collect) / rendered HDF5 (render).

Pilot: --pilot-seen S --pilot-tierb T restricts to the first S Seen-candidate + T
Tier-B-candidate categories (and all their keys) to measure real gate-yield / disk / time
before committing to the full run.

Usage (pilot):
    python scripts/run_scale_pipeline.py --manifest scale_pool_manifest.json \
        --out-root /tmp2/leocheng/ricl_scratch/scale_pilot \
        --pilot-seen 4 --pilot-tierb 4 --workers 8 --render
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PY = sys.executable  # run children with the same (libero) interpreter
_DEFAULT_BDDL = os.path.join(_REPO, "libero", "libero", "bddl_files", "robocasa_pickplace_scale")


def sh(cmd, log_path=None, env=None):
    """Run a subprocess, tee combined output to log_path; return (rc, tail)."""
    e = dict(os.environ)
    if env:
        e.update(env)
    with open(log_path, "w") if log_path else open(os.devnull, "w") as lf:
        p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=e, cwd=_REPO)
    tail = ""
    if log_path:
        with open(log_path) as f:
            tail = "".join(f.readlines()[-8:])
    return p.returncode, tail


def select_keys(manifest, pilot_seen, pilot_tierb, categories=None):
    if categories:
        keep = set(categories)
        return [k for k in manifest["keys"] if k["category"] in keep]
    if not pilot_seen and not pilot_tierb:
        return list(manifest["keys"])
    seen_c = set(manifest["seen_candidate_categories"][:pilot_seen or 0])
    tierb_c = set(manifest["tierb_candidate_categories"][:pilot_tierb or 0])
    keep = seen_c | tierb_c
    return [k for k in manifest["keys"] if k["category"] in keep]


def bddl_for_key(bddl_dir, key):
    """Match the generated bddl whose language embeds the raw key."""
    needle = f"the_{key}_and_place"
    for fn in os.listdir(bddl_dir):
        if fn.endswith(".bddl") and needle in fn:
            return os.path.join(bddl_dir, fn)
    return None


def rendered_path(out_root, bddl_file):
    rel = bddl_file.split("bddl_files/")[-1].replace(".bddl", "_demo.hdf5")
    return os.path.join(out_root, "rendered", rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--bddl-dir", default=_DEFAULT_BDDL)
    ap.add_argument("--stages", default="convert,bddl,collect,render,report")
    ap.add_argument("--pilot-seen", type=int, default=0)
    ap.add_argument("--pilot-tierb", type=int, default=0)
    ap.add_argument("--categories", nargs="*", default=None,
                    help="explicit category subset (overrides --pilot-*); keeps all their keys")
    ap.add_argument("--num-success", type=int, default=50)
    ap.add_argument("--max-attempts", type=int, default=100)
    ap.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 2) - 1))
    ap.add_argument("--render", action="store_true", help="run the render stage")
    ap.add_argument("--gpu", type=int, default=None,
                    help="pin the (EGL) render to this single GPU to avoid collisions on a "
                         "shared box (sets MUJOCO_EGL_DEVICE_ID + CUDA_VISIBLE_DEVICES)")
    args = ap.parse_args()
    stages = set(s.strip() for s in args.stages.split(","))

    manifest = json.load(open(args.manifest))
    keys = select_keys(manifest, args.pilot_seen, args.pilot_tierb, args.categories)
    key_recs = {k["key"]: k for k in keys}
    os.makedirs(args.out_root, exist_ok=True)
    os.makedirs(os.path.join(args.out_root, "logs"), exist_ok=True)
    work_manifest = os.path.join(args.out_root, "working_manifest.json")
    json.dump({**{k: manifest[k] for k in manifest if k != "keys"}, "keys": keys},
              open(work_manifest, "w"), indent=2)
    convert_report = os.path.join(args.out_root, "convert_report.json")
    print(f"[run] {len(keys)} keys | out-root={args.out_root} | workers={args.workers}")

    # --- convert ---
    if "convert" in stages:
        print("[stage] convert")
        sh([_PY, "tools/convert_robocasa_xml.py", "--manifest", work_manifest,
            "--report", convert_report],
           log_path=os.path.join(args.out_root, "logs", "convert.log"))
    converted = [k for k in keys
                 if os.path.isfile(os.path.join(_REPO, "libero/libero/assets/robocasa_objects",
                                                k["key"], "model.xml"))]
    print(f"[run] converted keys on disk: {len(converted)}/{len(keys)}")

    # --- register-check (fresh process; objects register at import) + bddl ---
    if "bddl" in stages:
        print("[stage] bddl")
        chk = subprocess.run(
            [_PY, "-c",
             "import sys, json; from libero.libero.envs.objects import get_object_dict; "
             "d=get_object_dict(); print(json.dumps([k for k in sys.argv[1:] if k in d]))",
             *[k["key"] for k in converted]],
            capture_output=True, text=True, cwd=_REPO)
        registered = json.loads(chk.stdout.strip().splitlines()[-1]) if chk.returncode == 0 else []
        print(f"[run] registered: {len(registered)}/{len(converted)}")
        if registered:
            sh([_PY, "scripts/create_robocasa_pickplace_tasks.py",
                "--manipulanda", *registered, "--receptacle", "rc_tray",
                "--out-dir", args.bddl_dir],
               log_path=os.path.join(args.out_root, "logs", "bddl.log"))

    # map key -> bddl
    key_bddl = {k["key"]: bddl_for_key(args.bddl_dir, k["key"])
                for k in converted if os.path.isdir(args.bddl_dir)}
    key_bddl = {k: v for k, v in key_bddl.items() if v}

    # --- collect (parallel, CPU, gated) ---
    if "collect" in stages:
        print(f"[stage] collect ({len(key_bddl)} keys, {args.workers} workers)")

        def collect_one(key):
            out_dir = os.path.join(args.out_root, "collect", key)
            rj = os.path.join(out_dir, "result.json")
            if os.path.isfile(rj):
                return key, "skip"
            os.makedirs(out_dir, exist_ok=True)
            rc, _ = sh([_PY, "scripts/collect_scripted_demonstrations.py",
                        "--bddl-file", key_bddl[key], "--out-dir", out_dir,
                        "--result-json", rj, "--num-success", str(args.num_success),
                        "--max-attempts", str(args.max_attempts)],
                       log_path=os.path.join(args.out_root, "logs", f"{key}.collect.log"))
            return key, ("ok" if os.path.isfile(rj) else f"fail(rc={rc})")

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(collect_one, k): k for k in key_bddl}
            for i, fut in enumerate(as_completed(futs), 1):
                key, status = fut.result()
                print(f"  [collect {i}/{len(futs)}] {key}: {status}")
        print(f"[run] collect done in {(time.time()-t0)/60:.1f} min")

    # --- render survivors (serialized, GPU) ---
    survivors = []
    for key in key_bddl:
        rj = os.path.join(args.out_root, "collect", key, "result.json")
        if os.path.isfile(rj) and json.load(open(rj)).get("reached_target"):
            survivors.append(key)
    if "render" in stages and args.render:
        render_env = {"MUJOCO_GL": "egl"}
        if args.gpu is not None:
            render_env["MUJOCO_EGL_DEVICE_ID"] = str(args.gpu)
            render_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"[stage] render ({len(survivors)} survivors, serialized"
              f"{', GPU ' + str(args.gpu) if args.gpu is not None else ''})")
        for i, key in enumerate(survivors, 1):
            out_hdf5 = rendered_path(args.out_root, key_bddl[key])
            if os.path.isfile(out_hdf5):
                print(f"  [render {i}/{len(survivors)}] {key}: skip")
                continue
            rc, tail = sh([_PY, "scripts/create_dataset.py",
                           "--demo-file", os.path.join(args.out_root, "collect", key, "demo.hdf5"),
                           "--use-camera-obs", "--compress",
                           "--output-root", os.path.join(args.out_root, "rendered")],
                          log_path=os.path.join(args.out_root, "logs", f"{key}.render.log"),
                          env=render_env)
            print(f"  [render {i}/{len(survivors)}] {key}: {'ok' if os.path.isfile(out_hdf5) else f'fail(rc={rc})'}")

    # --- report + tier split ---
    if "report" in stages:
        print("[stage] report")
        conv = json.load(open(convert_report)) if os.path.isfile(convert_report) else {"keys": []}
        rows, n_surv, n_drop, n_nocol, n_rend = [], 0, 0, 0, 0
        for k in keys:
            key = k["key"]
            rj = os.path.join(args.out_root, "collect", key, "result.json")
            row = {"key": key, "category": k["category"], "source": k["source"],
                   "role": k["role_candidate"], "is_overlap": k.get("is_overlap"),
                   "bbox_half": k.get("bbox_half")}
            if os.path.isfile(rj):
                r = json.load(open(rj))
                row.update(chosen_grasp_frac=r.get("chosen_grasp_frac"),
                           gate_yield=r.get("gate_yield"), n_demos=r.get("n_demos"),
                           reached_target=r.get("reached_target"))
                if r.get("reached_target"):
                    n_surv += 1
                    if key in key_bddl and os.path.isfile(rendered_path(args.out_root, key_bddl[key])):
                        n_rend += 1
                else:
                    n_drop += 1
            else:
                row["reached_target"] = None
                n_nocol += 1
            rows.append(row)

        report = {"n_keys": len(keys), "n_converted": len(converted),
                  "n_survivors": n_surv, "n_dropped": n_drop, "n_not_collected": n_nocol,
                  "n_rendered": n_rend, "convert": conv.get("keys", []), "keys": rows}
        json.dump(report, open(os.path.join(args.out_root, "scale_pool_report.json"), "w"), indent=2)

        # tier split over survivors
        sh([_PY, "tools/make_tier_split.py", "--manifest", work_manifest,
            "--collect-dir", os.path.join(args.out_root, "collect"),
            "--out", os.path.join(args.out_root, "tier_split.json")],
           log_path=os.path.join(args.out_root, "logs", "tier_split.log"))
        ts = json.load(open(os.path.join(args.out_root, "tier_split.json"))) \
            if os.path.isfile(os.path.join(args.out_root, "tier_split.json")) else {"counts": {}}

        with open(os.path.join(args.out_root, "scale_pool_report.md"), "w") as f:
            f.write(f"# Scale pool report\n\n")
            f.write(f"- keys: {len(keys)} | converted: {len(converted)} | "
                    f"survivors: {n_surv} | dropped(gated): {n_drop} | "
                    f"not_collected: {n_nocol} | rendered: {n_rend}\n")
            f.write(f"- tiers (survivors): {json.dumps(ts.get('counts', {}))}\n\n")
            f.write("## Dropped / not-collected (per-object ceiling)\n\n")
            for row in rows:
                if row.get("reached_target") in (False, None):
                    f.write(f"- {row['key']} ({row['role']}, {row['source']}): "
                            f"reached_target={row.get('reached_target')} "
                            f"gate_yield={row.get('gate_yield')} n_demos={row.get('n_demos')}\n")
        print(f"[run] report -> {args.out_root}/scale_pool_report.md  "
              f"survivors={n_surv} dropped={n_drop} not_collected={n_nocol} rendered={n_rend}")


if __name__ == "__main__":
    main()
