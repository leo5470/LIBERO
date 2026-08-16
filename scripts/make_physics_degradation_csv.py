"""Emit physics_degradation.csv -- the object instances whose demos degrade in eval physics.

"Affected" is defined by the DELTA between the yield recorded at collection (heavier,
wrapper physics) and the yield the same stored grasp achieves in the plain build eval uses.
Absolute eval yield is deliberately not the criterion: 46 of the 71 tasks reading 0% simply
collected below 30%, so a low level says nothing on its own.

Degenerate-mass objects are carried too -- a separate asset defect, flagged in its own column
rather than folded into severity.
"""
import csv, json, os, collections

SP = os.path.dirname(os.path.abspath(__file__))
REPO = "/tmp2/leocheng/forks/LIBERO"
V2 = "/tmp2/leocheng/ricl_scratch/stockbg_v2/collect"
OUT = os.path.join(REPO, "eval", "physics_degradation.csv")

rows = list(csv.DictReader(open(os.path.join(REPO, "eval", "eval_repro_flags.csv"))))
for r in rows:
    for k in ("v2_collect_yield", "eval_yield", "delta", "eval_mass_g", "demo_mass_g",
              "mass_ratio"):
        r[k] = float(r[k])

# how many demos each object actually contributes, and how healthy its siblings are
cat = collections.defaultdict(list)
for r in rows:
    cat[r["category"]].append(r)
healthy = {c: sum(1 for x in v if x["eval_yield"] >= 50) for c, v in cat.items()}


def n_demos(task):
    try:
        return json.load(open(os.path.join(V2, task, "result.json"))).get("n_demos", "")
    except Exception:
        return ""


def severity(r):
    d = r["delta"]
    if r["v2_collect_yield"] >= 50 and r["eval_yield"] < 10:
        return "unreproducible"
    if d < -50:
        return "lost"
    if d < -25:
        return "badly_degraded"
    if d < -10:
        return "degraded"
    return ""


affected = [r for r in rows if severity(r) or r["eval_mass_g"] < 1.0]
affected.sort(key=lambda r: (r["delta"], -r["v2_collect_yield"]))

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["object", "category", "severity", "degenerate_mass",
                "collect_yield_demo_physics", "eval_yield", "delta_pts",
                "n_demos_affected", "eval_mass_g", "demo_mass_g", "mass_ratio",
                "instances_in_category", "healthy_instances_in_category", "task"])
    for r in affected:
        obj = r["task"].replace("pick_up_the_", "").replace("_and_place_it_in_the_basket", "")
        w.writerow([obj, r["category"], severity(r) or "mass_only",
                    "yes" if r["eval_mass_g"] < 1.0 else "no",
                    f"{r['v2_collect_yield']:.1f}", f"{r['eval_yield']:.1f}",
                    f"{r['delta']:.1f}", n_demos(r["task"]),
                    f"{r['eval_mass_g']:.4f}", f"{r['demo_mass_g']:.4f}",
                    f"{r['mass_ratio']:.2f}",
                    len(cat[r["category"]]), healthy[r["category"]], r["task"]])

by = collections.Counter(severity(r) or "mass_only" for r in affected)
dem = sum(int(n_demos(r["task"]) or 0) for r in affected)
print(f"wrote {OUT}")
print(f"  {len(affected)} affected objects across "
      f"{len({r['category'] for r in affected})} categories")
for k in ("unreproducible", "lost", "badly_degraded", "degraded", "mass_only"):
    if by.get(k):
        print(f"    {k:16s} {by[k]:4d}")
print(f"  demos implicated: {dem:,} of 67,615 ({100*dem/67615:.1f}%)")
