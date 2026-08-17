"""Emit eval/physics_impact_by_category.csv -- per-category object counts by impact level.

Counts only, one row per category. `n_affected + n_healthy == n_objects` by construction, and
the five impact-level columns sum to `n_affected`, so the table adds up in both directions.

Severity is recomputed here from eval/eval_repro_flags.csv rather than read out of
eval/physics_degradation.csv, so this script stands alone.

  unreproducible  collected >=50% but reproduces <10% in the env eval builds
  lost            >50 pt drop
  badly_degraded  25-50 pt drop
  degraded        10-25 pt drop
  mass_only       degenerate eval-physics mass (<1 g) with no yield drop of its own

`n_eval_yield_ge50` is deliberately a SEPARATE axis from healthy/affected: an object can be
degraded (e.g. 100% -> 80%) and still be perfectly usable, so the two must not be conflated.
"""
import csv
import collections
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "eval", "eval_repro_flags.csv")
OUT = os.path.join(REPO, "eval", "physics_impact_by_category.csv")

LEVELS = ["unreproducible", "lost", "badly_degraded", "degraded", "mass_only"]


def severity(collect, ev, delta, mass_g):
    """Impact level, or None when the object is unaffected."""
    if collect >= 50 and ev < 10:
        return "unreproducible"
    if delta < -50:
        return "lost"
    if delta < -25:
        return "badly_degraded"
    if delta < -10:
        return "degraded"
    return "mass_only" if mass_g < 1.0 else None


def main():
    cat = collections.defaultdict(list)
    with open(SRC) as f:
        for r in csv.DictReader(f):
            r["sev"] = severity(float(r["v2_collect_yield"]), float(r["eval_yield"]),
                                float(r["delta"]), float(r["eval_mass_g"]))
            cat[r["category"]].append(r)

    rows = []
    for c, v in cat.items():
        counts = collections.Counter(x["sev"] for x in v if x["sev"])
        n_aff = sum(counts.values())
        rows.append({
            "category": c,
            "n_objects": len(v),
            "n_affected": n_aff,
            "n_healthy": len(v) - n_aff,
            **{lv: counts.get(lv, 0) for lv in LEVELS},
            "pct_affected": round(100.0 * n_aff / len(v), 1),
            "n_eval_yield_ge50": sum(1 for x in v if float(x["eval_yield"]) >= 50),
        })
    rows.sort(key=lambda r: (-r["n_affected"], -r["pct_affected"], r["category"]))

    cols = (["category", "n_objects", "n_affected", "n_healthy"] + LEVELS
            + ["pct_affected", "n_eval_yield_ge50"])
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    tot = {k: sum(r[k] for r in rows) for k in ["n_objects", "n_affected", "n_healthy"] + LEVELS}
    print(f"wrote {OUT}  ({len(rows)} categories)")
    print(f"  objects {tot['n_objects']}  =  affected {tot['n_affected']} "
          f"+ healthy {tot['n_healthy']}")
    print("  impact levels: " + "  ".join(f"{lv} {tot[lv]}" for lv in LEVELS)
          + f"  (sum {sum(tot[lv] for lv in LEVELS)})")
    print(f"  categories with zero affected: {sum(1 for r in rows if r['n_affected'] == 0)}")


if __name__ == "__main__":
    main()
