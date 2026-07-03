"""Generate one pick-and-place BDDL task per RoboCasa manipuland.

Mirrors ``scripts/create_libero_task_example.py`` but builds, programmatically, one
``kitchen_table`` scene per manipuland: table + a fixed receptacle + that one object,
with goal ``(In <obj>_1 <receptacle>_1_contain_region)``. Only the grasped object varies
across tasks (skill/scene/receptacle/camera held fixed), which is what isolates *object*
generalization for the downstream RICL experiment.

The manipuland + receptacle names are LIBERO registry keys (the converted RoboCasa object
category dirs under assets/robocasa_objects/). The receptacle must carry a ``contain_region``
site in its converted XML so it exposes an ``(In ...)`` affordance (see basket.xml).

Example:
    python scripts/create_robocasa_pickplace_tasks.py \
        --manipulanda apple lemon lime orange bell_pepper \
        --receptacle rc_tray \
        --out-dir libero/libero/bddl_files/robocasa_pickplace
"""

import argparse
import os
import re

import numpy as np

import init_path  # noqa: F401
from libero.libero.envs.objects import get_object_dict
from libero.libero.utils.bddl_generation_utils import (
    get_xy_region_kwargs_list_from_regions_info,
)
from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates, get_scene_dict
from libero.libero.utils.task_generation_utils import (
    register_task_info,
    generate_bddl_from_task_info,
)


def _mu_key(class_name: str) -> str:
    """Reproduce mu_utils.register_mu's class-name -> registry-key mangling."""
    return "_".join(re.sub(r"([A-Z])", r" \1", class_name).split()).lower()


def make_scene(manipuland: str, receptacle: str):
    """Define + register a kitchen_table scene holding {manipuland, receptacle}.

    Returns the scene registry key for use as register_task_info(scene_name=...).
    """
    cls_name = "RoboCasaPickPlace" + "".join(p.capitalize() for p in manipuland.split("_"))

    class _Scene(InitialSceneTemplates):
        def __init__(self):
            super().__init__(
                workspace_name="kitchen_table",
                fixture_num_info={"kitchen_table": 1},
                object_num_info={manipuland: 1, receptacle: 1},
            )

        def define_regions(self):
            # manipuland and receptacle on separate, non-overlapping table patches
            self.regions.update(self.get_region_dict(
                region_centroid_xy=[0.0, -0.10], region_name=f"{manipuland}_init_region",
                target_name=self.workspace_name, region_half_len=0.025))
            self.regions.update(self.get_region_dict(
                region_centroid_xy=[0.0, 0.15], region_name=f"{receptacle}_init_region",
                target_name=self.workspace_name, region_half_len=0.03))
            self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

        @property
        def init_states(self):
            return [
                ("On", f"{manipuland}_1", f"kitchen_table_{manipuland}_init_region"),
                ("On", f"{receptacle}_1", f"kitchen_table_{receptacle}_init_region"),
            ]

    _Scene.__name__ = cls_name
    _Scene.__qualname__ = cls_name
    register_mu(scene_type="kitchen")(_Scene)
    return _mu_key(cls_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manipulanda", nargs="+", required=True,
                    help="LIBERO registry keys of the objects to pick")
    ap.add_argument("--receptacle", required=True,
                    help="LIBERO registry key of the place-target (needs contain_region site)")
    ap.add_argument("--out-dir", required=True,
                    help="folder for generated .bddl files (keep 'bddl_files' in the path)")
    args = ap.parse_args()

    registry = get_object_dict()
    missing = [o for o in args.manipulanda + [args.receptacle] if o not in registry]
    if missing:
        raise SystemExit(f"[error] not registered in OBJECTS_DICT: {missing}\n"
                         f"        (did the converter copy assets/robocasa_objects/<name>/model.xml?)")

    os.makedirs(args.out_dir, exist_ok=True)
    for m in args.manipulanda:
        scene_key = make_scene(m, args.receptacle)
        register_task_info(
            language=f"pick up the {m.replace('_', ' ')} and place it in the {args.receptacle.replace('_', ' ')}",
            scene_name=scene_key,
            objects_of_interest=[f"{m}_1", f"{args.receptacle}_1"],
            goal_states=[("In", f"{m}_1", f"{args.receptacle}_1_contain_region")],
        )

    bddl_files, failures = generate_bddl_from_task_info(folder=args.out_dir)
    print(f"[generated] {len(bddl_files)} bddl files in {args.out_dir}")
    for b in bddl_files:
        print("   ", b)
    if failures:
        raise SystemExit(f"[error] {len(failures)} task(s) FAILED to generate: {failures}")


if __name__ == "__main__":
    main()
