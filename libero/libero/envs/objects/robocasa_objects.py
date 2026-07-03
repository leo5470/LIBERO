"""RoboCasa objects imported into LIBERO as robosuite ``MujocoXMLObject`` manipulanda.

Each converted RoboCasa object lives at::

    libero/libero/assets/robocasa_objects/<category>/model.xml   (+ meshes/textures)

where ``<category>`` is a clean LIBERO registry key (e.g. ``apple``, ``bell_pepper``).
At import time we scan that directory and register one object class per subdirectory that
contains a ``model.xml``. This mirrors ``google_scanned_objects.py`` but registers
programmatically (we import hundreds eventually), and is safe to import even when the
assets directory is empty or absent (it simply registers nothing).

The XML itself must already be robosuite-1.4 / MuJoCo loadable -- that conversion is done
offline by ``tools/convert_robocasa_xml.py``; this module only loads + registers.
"""

import os
import re
import pathlib

import numpy as np

from robosuite.models.objects import MujocoXMLObject

from libero.libero.envs.base_object import register_object, OBJECTS_DICT

_ASSETS = pathlib.Path(__file__).parent.parent.parent.absolute() / "assets" / "robocasa_objects"


class RoboCasaObject(MujocoXMLObject):
    """A single RoboCasa object instance loaded from a converted ``model.xml``."""

    # default placement rotation; override per-object via _ROTATION_OVERRIDES if a mesh
    # spawns in an awkward pose (validated visually in agentview during Phase 0).
    rotation = (0.0, 0.0)
    rotation_axis = "z"

    def __init__(self, name, obj_name, joints=[dict(type="free", damping="0.0005")]):
        super().__init__(
            str(_ASSETS / obj_name / "model.xml"),
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", type(self).__name__).split()
        ).lower()
        self.object_properties = {"vis_site_names": {}}


# Per-object placement-rotation overrides, keyed by category dir name. Populate during
# Phase 0 if an object spawns on its side (value is ((min, max), axis)).
_ROTATION_OVERRIDES = {}


def _camel(snake: str) -> str:
    return "".join(part.capitalize() for part in snake.split("_"))


def _make_class(obj_name: str):
    """Create a RoboCasaObject subclass whose registry key == ``obj_name``."""
    rot, axis = _ROTATION_OVERRIDES.get(obj_name, (RoboCasaObject.rotation, RoboCasaObject.rotation_axis))

    def __init__(self, name=None, obj_name=obj_name):
        RoboCasaObject.__init__(self, name if name is not None else obj_name, obj_name)
        self.rotation, self.rotation_axis = rot, axis

    return type(_camel(obj_name), (RoboCasaObject,), {"__init__": __init__})


def register_robocasa_objects():
    """Scan the assets dir and register one object class per ``<cat>/model.xml``.

    Returns the list of registry keys that were (or already are) available.
    """
    registered = []
    if not _ASSETS.is_dir():
        return registered
    for d in sorted(p for p in _ASSETS.iterdir() if p.is_dir()):
        if not (d / "model.xml").is_file():
            continue
        obj_name = d.name
        if obj_name in OBJECTS_DICT:  # already registered (re-import / name clash)
            registered.append(obj_name)
            continue
        # Register under the EXACT dir name. We cannot use register_object() here: it
        # derives the key from the class name via a camel<->snake round-trip that mangles
        # per-instance keys (double underscores collapse, multi-digit indices split, e.g.
        # ``apple__objaverse_10`` -> ``apple_objaverse_1_0``). Keying by the dir name keeps
        # the registry key == manifest key == BDDL object name.
        OBJECTS_DICT[obj_name] = _make_class(obj_name)
        registered.append(obj_name)
    return registered


# register at import time (side effect, like the other object modules)
register_robocasa_objects()
