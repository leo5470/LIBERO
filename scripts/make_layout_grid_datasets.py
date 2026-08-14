"""Per-layout scripted datasets: one object, many table layouts, one demo each.

WHY THIS EXISTS. Measured over the 612-task ``stockbg_layoutext`` run, 50 scripted demos
collected in a SINGLE layout span 8 principal components and a normalised pairwise spread
of 0.149; the same 50 demos spread over 6 layouts span 0.595 -- the same figure as 50
*human* LIBERO-Object demos of one task (which span 22 PCs). The scripted policy is
deterministic given state, so repeats inside one layout differ only by the +-2.5 cm reset
jitter and add almost nothing. Diversity lives on the layout axis, so this script spends
the budget there: **many layouts, one demo each, each layout a standalone dataset.**

WHY A NEW GENERATOR. ``scripts/create_stockbg_train_variants.py`` swaps the target into one
of the 5 stock ``other_object_region_*`` slots, so it caps at 6 target positions. Going past
that needs NEW regions, which that script deliberately refuses to invent (RandomizationError
is this suite's #1 blocker). This one invents them, and pays for it with an empirical screen:
every generated layout is only kept if the scripted policy actually completes it.

GEOMETRY (measured, not guessed -- see ``--verify``):
  * cells come from the union bounding box of the 6 stock object regions,
    x in [-0.225, 0.175], y in [-0.265, 0.085]. Staying inside the span the stock scene
    already proves placeable and reachable is the entire safety argument;
  * 5 cm pitch -> 8 x 7 = 56 cells; region boxes stay 5 x 5 cm like stock so per-reset
    jitter behaves identically;
  * target + 5 distractors are packed with pairwise centre separation >= 11.2 cm, which is
    the stock scene's OWN minimum (target_object_region <-> other_object_region_2), so we
    never pack tighter than a layout LIBERO already ships. All 56 cells admit a valid
    packing, and every cell is >= 20 cm from the basket.

SCALING. Batch size is a parameter, never an assumption:
  * layouts are seeded on ``crc32(target_key)``, so an item gets IDENTICAL layouts whether
    generated alone or inside a 500-item batch -- adding items never perturbs existing ones;
  * the work unit is one ``(item, layout)`` directory with its own ``result.json``, and
    anything already present is skipped, so expansion is incremental and interrupts are free;
  * workers pull from one flat ``(item x layout)`` queue, so a slow item cannot stall the pool.

OUTPUT. ``layouts.json`` is the authority on which layouts are usable -- failed candidates
keep their directory (that is what makes resume cheap) but are marked ``failed`` and are not
indexed. Consumers should iterate ``layouts.json``, not ``glob(layout_*)``.

    <out>/
      index.json                       # items, geometry, per-item survival counts
      <target_key>/
        grasp.json                     # the reused object-local grasp
        layouts.json                   # ordered survivors + every candidate's fate
        layout_00/{task.bddl, demo.hdf5, result.json}
        ...

Examples:
    # one named item (the primary workflow)
    python scripts/make_layout_grid_datasets.py --items mushroom__aigen_3 --out /tmp2/.../grid

    # one item per category, from a collect report
    python scripts/make_layout_grid_datasets.py --per-category 1 \
        --categories egg boxed_drink brussel_sprout honey_bottle mushroom --out /tmp2/.../grid

    # hundreds later -- only the new items are collected
    python scripts/make_layout_grid_datasets.py --items-file keys.txt --out /tmp2/.../grid
"""

import argparse
import collections
import json
import math
import os
import re
import subprocess
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PY = sys.executable  # children run under the same (libero) interpreter

# --------------------------------------------------------------------------- #
# Geometry. These are the knobs; every one is also a CLI flag.
# --------------------------------------------------------------------------- #
GRID_BOX = (-0.225, 0.175, -0.265, 0.085)   # x0, x1, y0, y1 -- union bbox of stock regions
GRID_PITCH = 0.05        # cell spacing
REGION_SIZE = 0.05       # emitted (:ranges ...) box edge, matches stock
MIN_SEP = 0.112          # min centre-to-centre between any two objects (stock's own min)
BASKET_XY = (0.0, 0.26)
MIN_BASKET_SEP = 0.16    # keep objects out of the basket's footprint

SUITE_DIR_DEFAULT = os.path.join(
    _REPO, "libero/libero/bddl_files/libero_object_unseen_stockbg")
GRASP_REPORT_DEFAULT = "/tmp2/leocheng/ricl_scratch/stockbg_final/suite_collect_report.json"
RANK_REPORT_DEFAULT = "/tmp2/leocheng/ricl_scratch/stockbg_layoutext/suite_collect_report.json"

_REGION_RE = re.compile(
    r"(\((?:target_object_region|other_object_region_\d)\s*"
    r"\(:target floor\)\s*\(:ranges \(\s*\()([^()]+)(\))", re.S)
_REGION_NAME_RE = re.compile(
    r"\((target_object_region|other_object_region_\d)\s*"
    r"\(:target floor\)\s*\(:ranges \(\s*\(([^()]+)\)", re.S)


# --------------------------------------------------------------------------- #
# BDDL rewriting
# --------------------------------------------------------------------------- #
def region_ranges(raw):
    """{region_name: 'x0 y0 x1 y1'} for the movable object regions."""
    return {m.group(1): m.group(2).strip() for m in _REGION_NAME_RE.finditer(raw)}


def set_region_centres(raw, centres, size=REGION_SIZE):
    """Rewrite each named region's ranges to a `size` box centred on its given (x, y)."""
    names = list(region_ranges(raw))
    missing = [n for n in centres if n not in names]
    assert not missing, f"regions absent from the BDDL: {missing}"
    half = size / 2.0

    def repl(m):
        name = m.group(0).split("(", 1)[1].split()[0]
        if name not in centres:
            return m.group(0)
        cx, cy = centres[name]
        coords = f"{cx - half} {cy - half} {cx + half} {cy + half}"
        return m.group(1) + coords + m.group(3)

    out, n = _REGION_RE.subn(repl, raw)
    assert n == len(names), f"rewrote {n} regions, expected {len(names)}"
    return out


def swap_distractors(raw, old_cats, new_cats):
    """Rewrite the 5 distractor identities (objects block + init lines), in order.

    Same contract as ``create_stockbg_train_variants.swap_distractors``.
    """
    for old, new in zip(old_cats, new_cats):
        if old == new:
            continue
        pat_obj = re.compile(rf"^(\s*){re.escape(old)}_1 - {re.escape(old)}$", re.M)
        raw, n1 = pat_obj.subn(rf"\g<1>{new}_1 - {new}", raw, count=1)
        pat_init = re.compile(
            rf"^(\s*)\(On {re.escape(old)}_1 (floor_other_object_region_\d)\)$", re.M)
        raw, n2 = pat_init.subn(rf"\g<1>(On {new}_1 \g<2>)", raw, count=1)
        assert n1 == 1 and n2 == 1, (old, new, n1, n2)
    return raw


# --------------------------------------------------------------------------- #
# Layout planning
# --------------------------------------------------------------------------- #
def grid_cells(box=GRID_BOX, pitch=GRID_PITCH):
    """Cell centres, row-major and deterministic."""
    x0, x1, y0, y1 = box
    xs = np.arange(x0 + pitch / 2, x1 - pitch / 2 + 1e-9, pitch)
    ys = np.arange(y0 + pitch / 2, y1 - pitch / 2 + 1e-9, pitch)
    return [(round(float(x), 4), round(float(y), 4)) for x in xs for y in ys]


def _far_enough(cell, chosen, min_sep):
    return all(math.hypot(cell[0] - c[0], cell[1] - c[1]) >= min_sep - 1e-9 for c in chosen)


def place_distractors(target, cells, rng, min_sep=MIN_SEP, n=5, tries=4000):
    """Seat `n` distractors on grid cells, pairwise >= min_sep from each other and target."""
    pool = [c for c in cells if c != target
            and math.hypot(c[0] - BASKET_XY[0], c[1] - BASKET_XY[1]) >= MIN_BASKET_SEP]
    for _ in range(tries):
        order = rng.permutation(len(pool))
        chosen = [target]
        for i in order:
            c = pool[int(i)]
            if _far_enough(c, chosen, min_sep):
                chosen.append(c)
                if len(chosen) == n + 1:
                    return chosen[1:]
    return None


def plan_layouts(key, category, parent_distractors, pool, args, rng=None):
    """Deterministic per-item layout plan. Seeded on the ITEM, never on batch position."""
    rng = rng or np.random.default_rng([args.seed, zlib.crc32(key.encode())])
    cells = grid_cells(tuple(args.grid_box), args.grid_pitch)
    order = list(range(len(cells)))
    if not args.no_shuffle_cells:
        order = [int(i) for i in rng.permutation(len(cells))]
    n_cand = args.n_candidates or len(cells)
    cand_pool = [c for c in pool if c != category]

    layouts = []
    for idx in order[:n_cand]:
        target = cells[idx]
        if math.hypot(target[0] - BASKET_XY[0],
                      target[1] - BASKET_XY[1]) < MIN_BASKET_SEP:
            continue
        dcells = place_distractors(target, cells, rng, args.min_sep)
        if dcells is None:
            continue
        dcats = [str(c) for c in rng.choice(cand_pool, size=5, replace=False)]
        layouts.append({
            "layout": len(layouts),
            "name": f"layout_{len(layouts):02d}",
            "cell": list(target),
            "cell_index": idx,
            "distractor_cells": [list(c) for c in dcells],
            "distractors": dcats,
            "parent_distractors": list(parent_distractors),
        })
    return layouts


def render_layout(parent_raw, layout, size):
    centres = {"target_object_region": tuple(layout["cell"])}
    for i, c in enumerate(layout["distractor_cells"]):
        centres[f"other_object_region_{i}"] = tuple(c)
    raw = set_region_centres(parent_raw, centres, size)
    return swap_distractors(raw, layout["parent_distractors"], layout["distractors"])


# --------------------------------------------------------------------------- #
# Item selection
# --------------------------------------------------------------------------- #
def load_report_tasks(path):
    return {t["name"]: t for t in json.load(open(path))["tasks"]}


def select_items(args, manifest):
    """Resolve the requested items to [{key, category, parent_task, ...}]."""
    by_key = {t["target_key"]: t for t in manifest["tasks"]}
    rank = load_report_tasks(args.rank_report) if os.path.isfile(args.rank_report) else {}

    # per-item robustness = the WORST yield among the rows that mention it
    worst = collections.defaultdict(lambda: None)
    for name, row in rank.items():
        base = name.rsplit("_var", 1)[0]
        t = by_key.get(row.get("target_key")) or manifest_lookup(manifest, base)
        if t is None:
            continue
        y = row.get("collect_yield")
        if y is None:
            continue
        k = t["target_key"]
        worst[k] = y if worst[k] is None else min(worst[k], y)

    keys = []
    if args.items:
        keys = list(args.items)
    elif args.items_file:
        keys = [ln.strip() for ln in open(args.items_file) if ln.strip()]
    else:
        cand = [k for k in by_key
                if not args.categories or by_key[k]["target_category"] in args.categories]
        cand = [k for k in cand
                if worst.get(k) is not None and worst[k] >= args.min_parent_yield]
        if args.per_category:
            per = collections.defaultdict(list)
            for k in sorted(cand):
                per[by_key[k]["target_category"]].append(k)
            for cat in sorted(per):
                per[cat].sort(key=lambda k: (-worst[k], k))
                keys += per[cat][:args.per_category]
        elif args.all:
            keys = sorted(cand)
        else:
            raise SystemExit("give --items / --items-file / --per-category N / --all")

    unknown = [k for k in keys if k not in by_key]
    if unknown:
        raise SystemExit(f"unknown target keys (not in the suite manifest): {unknown[:5]}")
    return [{"key": k, "category": by_key[k]["target_category"],
             "parent_task": by_key[k]["name"],
             "distractors": by_key[k]["distractors"],
             "robustness": worst.get(k)} for k in keys]


def manifest_lookup(manifest, name):
    for t in manifest["tasks"]:
        if t["name"] == name:
            return t
    return None


def resolve_grasp(item, grasp_tasks):
    """The recorded object-local grasp for this item, or None (-> normal calibration)."""
    row = grasp_tasks.get(item["parent_task"])
    if row is None:
        return None
    entry = row.get("chosen") or row.get("chosen_entry")
    if not entry or entry.get("local_off") is None or entry.get("kind") == "grasp_frac":
        return None
    return entry


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #
def sh(cmd, log_path, timeout):
    with open(log_path, "w") as lf:
        try:
            p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=_REPO,
                               timeout=timeout, start_new_session=True)
            return p.returncode
        except subprocess.TimeoutExpired:
            lf.write(f"\n[orchestrator] killed: exceeded {timeout}s\n")
            return -1


def collect_all(jobs, args):
    """Run every pending (item, layout) job through the collector, in parallel."""
    todo = [j for j in jobs if not os.path.isfile(os.path.join(j["dir"], "result.json"))]
    print(f"[collect] {len(todo)} pending of {len(jobs)} layouts "
          f"({len(jobs) - len(todo)} already done)")
    if not todo:
        return
    done = 0
    t0 = time.time()

    def one(job):
        os.makedirs(job["dir"], exist_ok=True)
        cmd = [_PY, "scripts/collect_scripted_demonstrations.py",
               "--bddl-file", job["bddl"], "--out-dir", job["dir"],
               "--result-json", os.path.join(job["dir"], "result.json"),
               "--num-success", str(args.demos_per_layout),
               "--max-attempts", str(args.max_attempts),
               "--build-tries", str(args.build_tries),
               "--seed", str(args.seed)]
        if job["grasp"]:
            cmd += ["--grasp-json", job["grasp"]]
        rc = sh(cmd, job["log"], args.task_timeout)
        return job, rc

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one, j) for j in todo]
        for f in as_completed(futs):
            job, rc = f.result()
            done += 1
            n = 0
            rj = os.path.join(job["dir"], "result.json")
            if os.path.isfile(rj):
                try:
                    n = json.load(open(rj)).get("n_demos", 0)
                except Exception:
                    pass
            rate = done / max(time.time() - t0, 1e-9) * 60
            print(f"  [{done}/{len(todo)}] {job['item']}/{job['name']}: demos={n}"
                  + (f"  rc={rc}" if rc else "") + f"   ({rate:.0f}/min)", flush=True)


# --------------------------------------------------------------------------- #
# Packaging + verification
# --------------------------------------------------------------------------- #
def discover_items(out):
    """Every item already present in the folder, newest run included.

    The index must describe the WHOLE folder, not just the items this run touched --
    otherwise adding one item to a 300-item dataset would shrink index.json to that item.
    """
    found = []
    for name in sorted(os.listdir(out)):
        lj = os.path.join(out, name, "layouts.json")
        if os.path.isfile(lj):
            blob = json.load(open(lj))
            found.append({"key": blob.get("target_key", name),
                          "category": blob.get("category"),
                          "parent_task": blob.get("parent_task"),
                          "grasp_path": (os.path.join(out, name, "grasp.json")
                                         if os.path.isfile(
                                             os.path.join(out, name, "grasp.json")) else None)})
    return found


def package(items, args):
    index = {"out": os.path.abspath(args.out),
             "geometry": {"grid_box": list(args.grid_box), "grid_pitch": args.grid_pitch,
                          "region_size": args.region_size, "min_sep": args.min_sep,
                          "n_cells": len(grid_cells(tuple(args.grid_box), args.grid_pitch))},
             "demos_per_layout": args.demos_per_layout,
             "n_layouts_requested": args.n_layouts,
             "seed": args.seed, "items": []}
    for item in items:
        idir = os.path.join(args.out, item["key"])
        lj = os.path.join(idir, "layouts.json")
        blob = json.load(open(lj))
        ok = 0
        for lay in blob["layouts"]:
            rj = os.path.join(idir, lay["name"], "result.json")
            n = 0
            if os.path.isfile(rj):
                try:
                    n = json.load(open(rj)).get("n_demos", 0)
                except Exception:
                    n = 0
            lay["n_demos"] = n
            if n > 0 and ok < args.n_layouts:
                lay["status"] = "ok"
                lay["layout_index"] = ok
                ok += 1
            elif n > 0:
                lay["status"] = "extra"
                lay["layout_index"] = None
            else:
                lay["status"] = "failed"
                lay["layout_index"] = None
        blob["n_ok"] = ok
        blob["n_failed"] = sum(l["status"] == "failed" for l in blob["layouts"])
        blob["n_extra"] = sum(l["status"] == "extra" for l in blob["layouts"])
        json.dump(blob, open(lj, "w"), indent=2)
        index["items"].append({
            "key": item["key"], "category": item["category"],
            "parent_task": item["parent_task"], "grasp_reused": bool(item.get("grasp_path")),
            "n_candidates": len(blob["layouts"]), "n_ok": ok,
            "n_failed": blob["n_failed"], "n_extra": blob["n_extra"],
            "n_demos": sum(l["n_demos"] for l in blob["layouts"] if l["status"] == "ok"),
        })
    json.dump(index, open(os.path.join(args.out, "index.json"), "w"), indent=2)

    print(f"\n{'item':28s} {'cat':16s} {'ok':>4s} {'fail':>5s} {'extra':>6s} {'demos':>6s}")
    short = []
    for r in index["items"]:
        print(f"{r['key']:28s} {r['category']:16s} {r['n_ok']:4d} {r['n_failed']:5d} "
              f"{r['n_extra']:6d} {r['n_demos']:6d}")
        if r["n_ok"] < args.n_layouts:
            short.append((r["key"], r["n_ok"]))
    print(f"\n[index] {len(index['items'])} items -> {os.path.join(args.out, 'index.json')}")
    if short:
        print(f"[warn] {len(short)} item(s) below the requested {args.n_layouts} layouts: "
              + ", ".join(f"{k}={n}" for k, n in short), file=sys.stderr)
    return index


def verify(items, args):
    """Geometry constraints over every generated layout. Exits non-zero on violation."""
    errs = []
    cells = set(grid_cells(tuple(args.grid_box), args.grid_pitch))
    for item in items:
        idir = os.path.join(args.out, item["key"])
        blob = json.load(open(os.path.join(idir, "layouts.json")))
        seen = set()
        for lay in blob["layouts"]:
            tag = f"{item['key']}/{lay['name']}"
            pts = [tuple(lay["cell"])] + [tuple(c) for c in lay["distractor_cells"]]
            for p in pts:
                if p not in cells:
                    errs.append(f"{tag}: {p} is not a grid cell")
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    d = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
                    if d < args.min_sep - 1e-6:
                        errs.append(f"{tag}: objects {i},{j} only {100 * d:.1f} cm apart")
            if math.hypot(pts[0][0] - BASKET_XY[0],
                          pts[0][1] - BASKET_XY[1]) < MIN_BASKET_SEP - 1e-6:
                errs.append(f"{tag}: target too close to the basket")
            if tuple(lay["cell"]) in seen:
                errs.append(f"{tag}: duplicate target cell {lay['cell']}")
            seen.add(tuple(lay["cell"]))
            if item["category"] in lay["distractors"]:
                errs.append(f"{tag}: target category used as a distractor")
            if len(set(lay["distractors"])) != 5:
                errs.append(f"{tag}: distractors not 5 distinct: {lay['distractors']}")
            bddl = os.path.join(idir, lay["name"], "task.bddl")
            if os.path.isfile(bddl):
                rr = region_ranges(open(bddl).read())
                got = [float(v) for v in rr["target_object_region"].split()]
                cx, cy = (got[0] + got[2]) / 2, (got[1] + got[3]) / 2
                if abs(cx - lay["cell"][0]) > 1e-6 or abs(cy - lay["cell"][1]) > 1e-6:
                    errs.append(f"{tag}: BDDL target centre {(cx, cy)} != plan {lay['cell']}")
                if abs((got[2] - got[0]) - args.region_size) > 1e-6:
                    errs.append(f"{tag}: region box is not {args.region_size} m")
    if errs:
        print(f"\nFAILED: {len(errs)} geometry violations", file=sys.stderr)
        for e in errs[:30]:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)
    print(f"[verify] OK: geometry holds over "
          f"{sum(len(json.load(open(os.path.join(args.out, i['key'], 'layouts.json')))['layouts']) for i in items)}"
          f" layouts")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="dataset folder root")
    ap.add_argument("--suite-dir", default=SUITE_DIR_DEFAULT,
                    help="eval suite the parent BDDLs come from (read-only)")
    ap.add_argument("--grasp-report", default=GRASP_REPORT_DEFAULT,
                    help="collect report supplying each item's reusable grasp")
    ap.add_argument("--rank-report", default=RANK_REPORT_DEFAULT,
                    help="collect report used to rank items by layout robustness")
    # selection
    ap.add_argument("--items", nargs="*", default=None, help="explicit target keys")
    ap.add_argument("--items-file", default=None, help="file of target keys, one per line")
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--per-category", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--min-parent-yield", type=float, default=80.0)
    # layout plan
    ap.add_argument("--n-layouts", type=int, default=50, help="layouts to keep per item")
    ap.add_argument("--n-candidates", type=int, default=0,
                    help="candidates to generate per item (0 = every grid cell)")
    ap.add_argument("--grid-box", type=float, nargs=4, default=list(GRID_BOX),
                    metavar=("X0", "X1", "Y0", "Y1"))
    ap.add_argument("--grid-pitch", type=float, default=GRID_PITCH)
    ap.add_argument("--region-size", type=float, default=REGION_SIZE)
    ap.add_argument("--min-sep", type=float, default=MIN_SEP)
    ap.add_argument("--no-shuffle-cells", action="store_true",
                    help="use row-major cell order for every item instead of a per-item "
                         "seeded permutation")
    # collection
    ap.add_argument("--demos-per-layout", type=int, default=1)
    ap.add_argument("--max-attempts", type=int, default=20)
    ap.add_argument("--build-tries", type=int, default=40)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--task-timeout", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-grasp-reuse", action="store_true",
                    help="calibrate per layout instead of reusing the recorded grasp "
                         "(~10x slower; only for validating the reuse)")
    # flow
    ap.add_argument("--stage", default="all",
                    choices=["generate", "collect", "package", "verify", "all"])
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and touch nothing")
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.suite_dir, "manifest.json")))
    items = select_items(args, manifest)
    if not items:
        raise SystemExit("no items selected")
    grasp_tasks = (load_report_tasks(args.grasp_report)
                   if os.path.isfile(args.grasp_report) else {})

    # distractor pool = every distractor category the suite's two stock bases use
    pool = sorted({d for t in manifest["tasks"] for d in t["distractors"]})

    print(f"[items] {len(items)} selected | grid "
          f"{len(grid_cells(tuple(args.grid_box), args.grid_pitch))} cells @ "
          f"{100 * args.grid_pitch:.0f} cm | {args.n_layouts} layouts x "
          f"{args.demos_per_layout} demo(s)")
    for it in items:
        g = resolve_grasp(it, grasp_tasks)
        it["grasp_entry"] = None if args.no_grasp_reuse else g
        print(f"  {it['key']:30s} {it['category']:16s} "
              f"robustness={it['robustness'] if it['robustness'] is not None else 'n/a':>6}"
              f"  grasp={'reused' if it['grasp_entry'] else 'CALIBRATE'}")
    n_recal = sum(1 for it in items if not it["grasp_entry"])
    if n_recal:
        print(f"[grasp] {n_recal} item(s) have no recorded grasp -> full calibration "
              f"(~10x slower per layout)", file=sys.stderr)

    if args.dry_run:
        it = items[0]
        lays = plan_layouts(it["key"], it["category"], it["distractors"], pool, args)
        print(f"\n[dry-run] {it['key']}: {len(lays)} candidate layouts")
        for lay in lays[:8]:
            print(f"  {lay['name']}  cell=({lay['cell'][0]:+.3f}, {lay['cell'][1]:+.3f})  "
                  f"distractors={','.join(lay['distractors'])}")
        print(f"  ... ({len(lays)} total); nothing written")
        return

    os.makedirs(args.out, exist_ok=True)
    jobs = []
    for it in items:
        idir = os.path.join(args.out, it["key"])
        os.makedirs(idir, exist_ok=True)
        lays = plan_layouts(it["key"], it["category"], it["distractors"], pool, args)
        if args.stage in ("generate", "all"):
            parent_raw = open(os.path.join(args.suite_dir,
                                           it["parent_task"] + ".bddl")).read()
            for lay in lays:
                ldir = os.path.join(idir, lay["name"])
                os.makedirs(ldir, exist_ok=True)
                with open(os.path.join(ldir, "task.bddl"), "w") as f:
                    f.write(render_layout(parent_raw, lay, args.region_size))
            if it["grasp_entry"]:
                it["grasp_path"] = os.path.join(idir, "grasp.json")
                json.dump(it["grasp_entry"], open(it["grasp_path"], "w"), indent=2)
            json.dump({"target_key": it["key"], "category": it["category"],
                       "parent_task": it["parent_task"],
                       "geometry": {"grid_box": list(args.grid_box),
                                    "grid_pitch": args.grid_pitch,
                                    "region_size": args.region_size,
                                    "min_sep": args.min_sep,
                                    "shuffled": not args.no_shuffle_cells},
                       "seed": args.seed, "layouts": lays},
                      open(os.path.join(idir, "layouts.json"), "w"), indent=2)
        gp = os.path.join(idir, "grasp.json")
        it["grasp_path"] = gp if os.path.isfile(gp) else None
        for lay in lays:
            ldir = os.path.join(idir, lay["name"])
            jobs.append({"item": it["key"], "name": lay["name"], "dir": ldir,
                         "bddl": os.path.join(ldir, "task.bddl"),
                         "grasp": it["grasp_path"],
                         "log": os.path.join(args.out, "logs",
                                             f"{it['key']}__{lay['name']}.log")})
    os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)
    if args.stage in ("generate", "all"):
        print(f"[generate] {len(jobs)} candidate layouts written under {args.out}")

    if args.stage in ("collect", "all"):
        collect_all(jobs, args)
    # package/verify cover every item in the folder, not just this run's, so incremental
    # expansion never shrinks index.json or skips checking what is already there
    if args.stage in ("package", "all"):
        package(discover_items(args.out), args)
    if args.stage in ("verify", "all"):
        verify(discover_items(args.out), args)


if __name__ == "__main__":
    main()
