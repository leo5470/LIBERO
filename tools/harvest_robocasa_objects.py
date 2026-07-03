"""Harvest RoboCasa object metadata -> robocasa_object_meta.json  (run in the ROBOCASA env).

The LIBERO side never imports robocasa; it only consumes this JSON + the on-disk meshes.

At import time RoboCasa turns each ``OBJ_CATEGORIES[cat]`` into ``{source: ObjCat}`` where
each ``ObjCat`` exposes ``graspable``, ``types``, ``scale``, ``exclude`` and ``mjcf_paths``
(the concrete instance ``model.xml`` files for the assets that are actually on disk). We
read those directly -- authoritative, no filesystem guessing -- and record the per-source
scale (RoboCasa applies it at load time, so the converter must bake it in).

Usage (robocasa env):
    python tools/harvest_robocasa_objects.py --out robocasa_object_meta.json
"""

import argparse
import json
import os

import robocasa  # noqa: F401
from robocasa.models.objects.kitchen_objects import OBJ_CATEGORIES


def to_list(t):
    if t is None:
        return []
    if isinstance(t, str):
        return [t]
    return list(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="robocasa_object_meta.json")
    args = ap.parse_args()

    cats = {}
    instances = []
    for cat_name, sources in OBJ_CATEGORIES.items():
        cat_graspable = None
        cat_types = []
        scale_by_source = {}
        for src_name, blk in sources.items():
            g = getattr(blk, "graspable", None)
            if cat_graspable is None:
                cat_graspable = g
            if not cat_types:
                cat_types = to_list(getattr(blk, "types", ()))
            scale = getattr(blk, "scale", 1.0)
            scale_by_source[src_name] = scale
            excl = set(getattr(blk, "exclude", []) or [])
            for p in (getattr(blk, "mjcf_paths", []) or []):
                if not os.path.exists(p):
                    continue  # asset subset not downloaded
                inst = os.path.basename(os.path.dirname(p))
                instances.append({
                    "instance": inst,
                    "category": cat_name,
                    "source": src_name,
                    "scale": scale,
                    "graspable": bool(g) if g is not None else None,
                    "types": to_list(getattr(blk, "types", ())),
                    "model_xml": p,
                    "excluded": inst in excl,
                })
        cats[cat_name] = {
            "graspable": cat_graspable,
            "types": cat_types,
            "scale_by_source": scale_by_source,
        }

    meta = {
        "assets_root": robocasa.models.assets_root,
        "categories": cats,
        "instances": instances,
    }
    with open(args.out, "w") as f:
        json.dump(meta, f, indent=2)

    n_grasp_cat = sum(1 for c in cats.values() if c["graspable"] is True)
    n_grasp_inst = sum(1 for i in instances if i["graspable"] and not i["excluded"])
    print(f"[harvest] {len(cats)} categories ({n_grasp_cat} graspable); "
          f"{len(instances)} instances on disk ({n_grasp_inst} graspable, non-excluded)")
    print(f"[harvest] wrote {args.out}")


if __name__ == "__main__":
    main()
