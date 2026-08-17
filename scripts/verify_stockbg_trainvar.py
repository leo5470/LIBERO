"""Constraint verifier for ``libero_object_unseen_stockbg_trainvar`` (pure text, no robosuite).

Companion to ``scripts/verify_stockbg_suite.py``, which proves the *eval* suite is "a stock
file with only the target changed". This one proves each train-time variant is "its parent
eval task with only the target SLOT and the distractor IDENTITIES changed" -- i.e. the
object, its language, the goal predicate and the region geometry are all untouched, so a
yield difference against the parent can only come from the layout.

Per variant it hard-checks:

  1. **region swap** -- ``target_object_region``'s ranges equal the parent's
     ``other_object_region_<slot-1>`` and vice versa; every other region is byte-identical.
     Slot 0 means the whole region block is byte-identical to the parent's;
  2. **target untouched** -- ``:language``, the target ``:objects`` line, ``:obj_of_interest``,
     the target ``(On ...)`` init line and the ``:goal`` are byte-identical to the parent's;
  3. **distractors** -- exactly 5, all distinct, none equal to the target category, each
     drawn from the manifest's declared pool, one per ``floor_other_object_region_i``;
  4. **nothing else moved** -- with the region block and the distractor lines masked out,
     the variant is byte-identical to its parent;
  5. **slot coverage** -- an object's variants use distinct slots, and with
     ``variants_per_object == 6`` they cover all of 0..5;
  6. **bijection** -- one file per manifest task, no orphan BDDLs.

Also reports how far each slot moves the target (centre-to-centre against the parent), so a
"variant" that barely moves the object shows up rather than passing silently.

Exits non-zero on any violation.

Example:
    python scripts/verify_stockbg_trainvar.py \
        --manifest libero/libero/bddl_files/libero_object_unseen_stockbg_trainvar/manifest.json
"""

import argparse
import collections
import json
import math
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_DEFAULT = os.path.join(
    REPO, "libero/libero/bddl_files/libero_object_unseen_stockbg_trainvar/manifest.json")
PARENT_DIR_DEFAULT = os.path.join(
    REPO, "libero/libero/bddl_files/libero_object_unseen_stockbg")

# one (:ranges ...) entry per named region, e.g. "(-0.145 -0.265 -0.095 -0.215)"
_REGION_RE = re.compile(
    r"\((target_object_region|other_object_region_\d)\s*"
    r"\(:target floor\)\s*\(:ranges \(\s*\(([^()]+)\)", re.S)
_ON_RE = re.compile(r"^\s*\(On (\S+)_1 (floor_other_object_region_\d)\)$", re.M)


def region_ranges(raw):
    return {m.group(1): m.group(2).strip() for m in _REGION_RE.finditer(raw)}


def region_block(raw):
    return raw[raw.index("(:regions"): raw.index("(:fixtures")]


def centre(coords):
    x0, y0, x1, y1 = (float(v) for v in coords.split())
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def mask(raw, distractors):
    """`raw` with the region block and every distractor mention replaced by a placeholder."""
    out = raw[:raw.index("(:regions")] + "<REGIONS>" + raw[raw.index("(:fixtures"):]
    for d in distractors:
        out = re.sub(rf"^(\s*){re.escape(d)}_1 - {re.escape(d)}$", r"\g<1><DISTRACTOR>",
                     out, flags=re.M)
        out = re.sub(rf"^(\s*)\(On {re.escape(d)}_1 (floor_other_object_region_\d)\)$",
                     r"\g<1>(On <DISTRACTOR> \g<2>)", out, flags=re.M)
    return out


def check_variant(task, raw, parent_raw, parent_task, pool, errors):
    name, slot = task["name"], task["target_slot"]

    def bad(msg):
        errors.append(f"{name}: {msg}")

    # 1. region swap
    pr, vr = region_ranges(parent_raw), region_ranges(raw)
    if set(pr) != set(vr):
        bad(f"region names changed: {sorted(set(pr) ^ set(vr))}")
        return None
    other = f"other_object_region_{slot - 1}" if slot else None
    for rname in pr:
        want = pr[rname]
        if slot and rname == "target_object_region":
            want = pr[other]
        elif slot and rname == other:
            want = pr["target_object_region"]
        if vr[rname] != want:
            bad(f"region {rname}: expected ranges {want!r}, got {vr[rname]!r}")
    if slot == 0 and region_block(raw) != region_block(parent_raw):
        bad("slot 0 must leave the region block byte-identical")

    # 2. target untouched
    cat, key = task["target_category"], task["target_key"]
    for label, pat in (("language", r"\(:language[^\)]*\)"),
                       ("obj_of_interest", r"\(:obj_of_interest.*?\)"),
                       ("goal", r"\(:goal.*?\n\s*\)")):
        pm = re.search(pat, parent_raw, re.S)
        vm = re.search(pat, raw, re.S)
        if not vm or not pm or vm.group(0) != pm.group(0):
            bad(f"{label} block differs from parent")
    if f"{key}_1 - {cat}" not in raw:
        bad(f"target objects line for {key}_1 - {cat} missing")
    if f"(On {key}_1 floor_target_object_region)" not in raw:
        bad(f"target init line for {key}_1 missing")

    # 3. distractors
    dist = task["distractors"]
    if len(dist) != 5:
        bad(f"expected 5 distractors, manifest has {len(dist)}")
    if len(set(dist)) != len(dist):
        bad(f"duplicate distractors: {dist}")
    if cat in dist:
        bad(f"target category {cat!r} used as a distractor")
    outside = [d for d in dist if d not in pool]
    if outside:
        bad(f"distractors outside the declared pool: {outside}")
    on_lines = _ON_RE.findall(raw)
    got = {inst: reg for inst, reg in on_lines if inst != key}
    if sorted(got) != sorted(dist):
        bad(f"(On ...) distractors {sorted(got)} != manifest {sorted(dist)}")
    if len(set(got.values())) != len(got):
        bad(f"two distractors share a region: {got}")
    for d in dist:
        if f"{d}_1 - {d}" not in raw:
            bad(f"distractor objects line for {d}_1 missing")

    # 4. nothing else moved
    if mask(raw, dist) != mask(parent_raw, parent_task["distractors"]):
        bad("differs from parent outside the region block and the distractor lines")

    # displacement of the target
    (px, py), (vx, vy) = centre(pr["target_object_region"]), centre(vr["target_object_region"])
    return math.hypot(vx - px, vy - py)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST_DEFAULT)
    ap.add_argument("--parent-dir", default=PARENT_DIR_DEFAULT)
    ap.add_argument("--min-move", type=float, default=0.05,
                    help="metres; warn when a non-zero slot moves the target less than this")
    args = ap.parse_args()

    suite_dir = os.path.dirname(os.path.abspath(args.manifest))
    manifest = json.load(open(args.manifest))
    pool = set(manifest["distractor_pool"])
    parents = {t["name"]: t for t in
               json.load(open(os.path.join(args.parent_dir, "manifest.json")))["tasks"]}

    errors, moves, by_key = [], collections.defaultdict(list), collections.defaultdict(list)
    for task in manifest["tasks"]:
        path = os.path.join(suite_dir, task["name"] + ".bddl")
        if not os.path.isfile(path):
            errors.append(f"{task['name']}: no BDDL emitted")
            continue
        parent_task = parents.get(task["parent_task"])
        if parent_task is None:
            errors.append(f"{task['name']}: parent {task['parent_task']} not in the eval manifest")
            continue
        parent_raw = open(os.path.join(args.parent_dir, task["parent_task"] + ".bddl")).read()
        d = check_variant(task, open(path).read(), parent_raw, parent_task, pool, errors)
        if d is not None:
            moves[task["target_slot"]].append(d)
        by_key[task["target_key"]].append(task["target_slot"])

    # 5. slot coverage
    npo = manifest["variants_per_object"]
    for key, slots in by_key.items():
        if len(set(slots)) != len(slots):
            errors.append(f"{key}: repeated slots {sorted(slots)}")
        if npo == 6 and set(slots) != set(range(6)):
            errors.append(f"{key}: slots {sorted(slots)} do not cover 0..5")

    # 6. bijection
    emitted = {f[:-5] for f in os.listdir(suite_dir) if f.endswith(".bddl")}
    orphans = emitted - {t["name"] for t in manifest["tasks"]}
    if orphans:
        errors.append(f"{len(orphans)} BDDLs not in the manifest: {sorted(orphans)[:5]}")

    print(f"suite      : {suite_dir}")
    print(f"tasks      : {len(manifest['tasks'])} over {len(by_key)} objects "
          f"({npo} variants each)")
    print(f"parents    : {len(parents)} in {args.parent_dir}")
    print("\ntarget displacement vs parent, by slot:")
    print(f"  {'slot':>4s} {'n':>4s} {'min cm':>7s} {'med cm':>7s} {'max cm':>7s}")
    for slot in sorted(moves):
        v = sorted(moves[slot])
        print(f"  {slot:4d} {len(v):4d} {100 * v[0]:7.1f} {100 * v[len(v) // 2]:7.1f} "
              f"{100 * v[-1]:7.1f}")
    for slot, v in sorted(moves.items()):
        if slot and min(v) < args.min_move:
            print(f"[warn] slot {slot} moves the target only {100 * min(v):.1f} cm "
                  f"for some objects", file=sys.stderr)

    if errors:
        print(f"\nFAILED: {len(errors)} violations", file=sys.stderr)
        for e in errors[:40]:
            print(f"  - {e}", file=sys.stderr)
        if len(errors) > 40:
            print(f"  ... and {len(errors) - 40} more", file=sys.stderr)
        sys.exit(1)
    print("\nOK: all constraints hold")


if __name__ == "__main__":
    main()
