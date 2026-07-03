"""Convert RoboCasa object model.xml -> robosuite-1.4 / LIBERO-loadable model.xml.

Pure-XML (no robocasa import needed) so it runs in the LIBERO env. For each selected
instance it:

  1. copies model.xml + referenced meshes/textures into
     libero/libero/assets/robocasa_objects/<key>/  and localizes ``file=`` paths;
  2. bakes RoboCasa's category *scale* into the geometry (RoboCasa applies it at load
     time via MJCFObject.set_scale; LIBERO's plain MujocoXMLObject does not, so we must);
  3. synthesizes the three sites robosuite uses for placement
     (``bottom_site`` / ``top_site`` / ``horizontal_radius_site``) from the ``reg_bbox``
     region geom, since RoboCasa objects expose extents via reg_* geoms, not sites;
  4. drops the reg_* marker geoms (keeping the real visual + collision geoms);
  5. (receptacle only, --contain-region) adds a ``contain_region`` box site over the
     interior so the object is a valid ``(In ...)`` place-target (cf. basket.xml).

Validate by LOADING the result in a LIBERO env, not by eyeballing XML.

Usage:
    python tools/convert_robocasa_xml.py --meta robocasa_object_meta.json \
        --map apple=apple_3 lemon=lemon_1 ... rc_tray=tray_2 \
        --contain-region rc_tray
"""

import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET

import numpy as np

DEST_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "libero", "libero", "assets", "robocasa_objects")


def _vec(s, n=3, default=1.0):
    if s is None:
        return np.array([default] * n, dtype=float)
    a = np.array([float(x) for x in str(s).split()], dtype=float)
    return a


def _as3(scale):
    if scale is None:
        return np.ones(3)
    if isinstance(scale, (int, float)):
        return np.array([scale] * 3, dtype=float)
    a = np.array(scale, dtype=float).reshape(-1)
    if a.size == 1:
        a = np.array([a[0]] * 3)
    assert a.size == 3, scale
    return a


def _scale_size(size, s, gtype):
    """Scale a geom/site ``size`` per its type. Approximate (axis-aligned) for Phase 0."""
    size = np.array(size, dtype=float)
    if gtype in ("box", "ellipsoid") and size.size == 3:
        return size * s
    if gtype == "sphere" and size.size >= 1:
        out = size.copy(); out[0] *= float(np.mean(s)); return out
    if gtype in ("cylinder", "capsule") and size.size == 2:
        return np.array([size[0] * float(np.mean(s[:2])), size[1] * float(s[2])])
    # default: scale all components by mean
    return size * float(np.mean(s))


def _bake_scale(root, s):
    """Multiply positions/sizes/mesh-scales by per-axis scale ``s`` (3-vec)."""
    asset = root.find("asset")
    if asset is not None:
        for mesh in asset.findall("mesh"):
            ms = _vec(mesh.get("scale"), default=1.0)
            mesh.set("scale", " ".join(f"{v:.8f}" for v in ms * s))
    wb = root.find("worldbody")
    for elem in wb.iter():
        tag = elem.tag
        if tag in ("body", "joint", "geom", "site"):
            pos = elem.get("pos")
            if pos is not None:
                elem.set("pos", " ".join(f"{v:.8f}" for v in _vec(pos) * s))
        if tag in ("geom", "site"):
            size = elem.get("size")
            if size is not None:
                gtype = elem.get("type", "sphere")
                ns = _scale_size(_vec(size, n=len(size.split())), s, gtype)
                elem.set("size", " ".join(f"{v:.8f}" for v in ns))


def _find_reg_bbox(root):
    """Return (pos, half) of the reg_bbox geom (already scaled), or None."""
    wb = root.find("worldbody")
    for geom in wb.iter("geom"):
        name = geom.get("name") or ""
        if name.endswith("reg_bbox") or name == "reg_bbox" or "reg_bbox" in name:
            pos = _vec(geom.get("pos", "0 0 0"))
            half = _vec(geom.get("size", "0 0 0"))
            return pos, half
    return None


def _outer_body_and_inner_pos(root):
    """robosuite reads placement sites (bottom/top/horizontal_radius) at
    ``worldbody/body/site`` (direct children of the OUTER wrapper body, siblings of
    ``<body name='object'>``), while LIBERO discovers region sites like ``contain_region``
    *inside* an inner ``<body>`` (problems/*_manipulation.py: worldbody/body -> .//body ->
    .//site). So we return the outer body, the inner object body, and the inner pos offset
    (bbox coords are in the inner frame)."""
    wb = root.find("worldbody")
    outer = wb.find("./body")
    inner = outer.find("./body[@name='object']") if outer is not None else None
    inner_pos = _vec(inner.get("pos", "0 0 0")) if inner is not None else np.zeros(3)
    return outer, inner, inner_pos


def _add_site(body, name, pos, size=(0.005,), rgba=(0, 0, 0, 0), stype=None, quat=None):
    s = ET.SubElement(body, "site")
    s.set("name", name)
    s.set("pos", " ".join(f"{v:.8f}" for v in pos))
    s.set("size", " ".join(f"{v:.6f}" for v in size))
    s.set("rgba", " ".join(str(v) for v in rgba))
    if stype:
        s.set("type", stype)
    if quat is not None:  # SiteObject (region sites) requires a quat; placement sites don't
        s.set("quat", quat)


def _drop_reg_geoms(root):
    wb = root.find("worldbody")
    for parent in wb.iter():
        for geom in list(parent.findall("geom")):
            name = geom.get("name") or ""
            if "reg_" in name:
                parent.remove(geom)


def _default_search_dirs():
    """Local dirs to resolve assets referenced by an absolute path baked on another machine
    (aigen XMLs point <texture> at robosuite's shared textures via the generator's abs path)."""
    dirs = []
    try:
        import robosuite
        dirs.append(os.path.join(os.path.dirname(robosuite.__file__), "models", "assets", "textures"))
    except Exception:
        pass
    return [d for d in dirs if os.path.isdir(d)]


def _find_by_basename(base, search_dirs):
    for d in search_dirs:
        cand = os.path.join(d, base)
        if os.path.exists(cand):
            return cand
        for r, _, files in os.walk(d):
            if base in files:
                return os.path.join(r, base)
    return None


def _localize_assets(root, src_dir, dest_dir, search_dirs=()):
    """Copy referenced meshes/textures next to the model and rewrite ``file=`` to basenames.
    Returns the list of asset files that could not be resolved (empty == fully localized)."""
    asset = root.find("asset")
    unresolved = []
    if asset is None:
        return unresolved
    for elem in list(asset.findall("mesh")) + list(asset.findall("texture")):
        fp = elem.get("file")
        if fp is None:
            continue
        src = fp if os.path.isabs(fp) else os.path.normpath(os.path.join(src_dir, fp))
        base = os.path.basename(src)
        if not os.path.exists(src):  # abs path from another machine -> resolve by basename
            alt = _find_by_basename(base, search_dirs)
            if alt is not None:
                src = alt
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest_dir, base))
            elem.set("file", base)
        else:
            print(f"    [warn] missing asset file: {src}")
            unresolved.append(base)
    return unresolved


def convert_one(model_xml, scale, key, dest_root, contain_region=False, search_dirs=()):
    src_dir = os.path.dirname(model_xml)
    dest_dir = os.path.join(dest_root, key)
    os.makedirs(dest_dir, exist_ok=True)

    tree = ET.parse(model_xml)
    root = tree.getroot()

    s = _as3(scale)
    _bake_scale(root, s)

    bbox = _find_reg_bbox(root)
    outer, inner, inner_pos = _outer_body_and_inner_pos(root)
    if bbox is not None and outer is not None:
        pos, half = bbox  # bbox center is in the INNER (object) body frame
        c = inner_pos + pos  # same point expressed in the OUTER-body frame
        # placement sites: OUTER body (robosuite bottom/top/horizontal_radius)
        _add_site(outer, "bottom_site", [c[0], c[1], c[2] - half[2]])
        _add_site(outer, "top_site", [c[0], c[1], c[2] + half[2]])
        _add_site(outer, "horizontal_radius_site", [float(np.hypot(half[0], half[1])), 0.0, 0.0])
        if contain_region and inner is not None:
            # region site goes INSIDE the object body (LIBERO discovers it via .//body/.//site),
            # in the inner frame. Capture box sits ABOVE the receptacle surface so a placed
            # object's center falls inside it (works for flat trays/plates, not just baskets).
            top_z = pos[2] + half[2]
            _add_site(inner, "contain_region",
                      [pos[0], pos[1], top_z + 0.04],
                      size=[max(half[0], 0.05), max(half[1], 0.05), 0.055],
                      rgba=(0.8, 0, 0, 0), stype="box", quat="0 0 0 1")
    else:
        print(f"    [warn] {key}: no reg_bbox/object body found -- sites not added")

    _drop_reg_geoms(root)
    unresolved = _localize_assets(root, src_dir, dest_dir, search_dirs=search_dirs)

    out = os.path.join(dest_dir, "model.xml")
    tree.write(out)
    print(f"    wrote {out}  (scale={list(np.round(s,3))}, bbox={'yes' if bbox else 'NO'}"
          f"{', UNRESOLVED=' + str(unresolved) if unresolved else ''})")
    return out, unresolved


def _autofit_scale(model_xml, scale, max_xy_half):
    """Shrink ``scale`` uniformly so the baked horizontal half-extent <= ``max_xy_half``
    (metres), keeping oversize meshes placeable on the table region. Returns
    (scale_vec, did_fit)."""
    s = _as3(scale)
    if not max_xy_half or max_xy_half <= 0:
        return s, False
    try:
        wb = ET.parse(model_xml).getroot().find("worldbody")
    except (ET.ParseError, OSError):
        return s, False
    half = None
    if wb is not None:
        for g in wb.iter("geom"):
            if "reg_bbox" in (g.get("name") or ""):
                half = _vec(g.get("size", "0 0 0"))
                break
    if half is None:
        return s, False
    baked_xy = max(half[0] * s[0], half[1] * s[1])
    if baked_xy > max_xy_half:
        return s * (max_xy_half / baked_xy), True
    return s, False


def run_manifest(args, search_dirs):
    """Batch-convert every key in a scale_pool_manifest.json (from select_scale_pool.py).

    Robust for scale: skip-on-failure + quarantine (a bad key never leaves a broken object
    behind), auto-fit-scale for oversize meshes, resumable (skip keys already converted),
    and a per-key conversion report."""
    manifest = json.load(open(args.manifest))
    contain = set(args.contain_region)
    report, n_ok, n_q, n_skip = [], 0, 0, 0
    for rec in manifest["keys"]:
        key, mx = rec["key"], rec["model_xml"]
        dest_dir = os.path.join(args.dest, key)
        entry = {"key": key, "category": rec.get("category"), "source": rec.get("source"),
                 "role": rec.get("role_candidate")}
        if not args.overwrite and os.path.isfile(os.path.join(dest_dir, "model.xml")):
            entry["status"] = "exists"
            report.append(entry); n_skip += 1
            continue
        adj, fit = _autofit_scale(mx, rec.get("applied_scale", 1.0), args.max_xy_half)
        print(f"[convert] {key} <- {rec.get('instance')} (cat={rec.get('category')}, "
              f"scale={list(np.round(adj, 3))}{' [auto-fit]' if fit else ''})")
        try:
            _, unresolved = convert_one(mx, adj, key, args.dest,
                                        contain_region=(key in contain), search_dirs=search_dirs)
            if unresolved:  # a missing asset would break MuJoCo load -> don't ship it
                shutil.rmtree(dest_dir, ignore_errors=True)
                entry.update(status="quarantined", reason=f"unresolved_assets:{unresolved}")
                n_q += 1
            else:
                entry.update(status="converted", applied_scale=list(np.round(adj, 5)),
                             auto_fit=fit, bbox_half=rec.get("bbox_half"))
                n_ok += 1
        except Exception as e:  # noqa: BLE001 - any XML/parse failure quarantines the key
            shutil.rmtree(dest_dir, ignore_errors=True)
            entry.update(status="quarantined", reason=f"{type(e).__name__}: {e}")
            n_q += 1
        report.append(entry)

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        json.dump({"n_converted": n_ok, "n_quarantined": n_q, "n_skipped_existing": n_skip,
                   "keys": report}, open(args.report, "w"), indent=2)
    print(f"[manifest] converted={n_ok} quarantined={n_q} skipped_existing={n_skip} -> {args.dest}")
    if args.report:
        print(f"[manifest] report -> {args.report}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", help="required for --map mode")
    ap.add_argument("--map", nargs="+",
                    help="key=instance pairs, e.g. apple=apple_3 rc_tray=tray_2")
    ap.add_argument("--manifest", help="scale_pool_manifest.json for batch conversion")
    ap.add_argument("--report", help="(manifest mode) write per-key conversion report JSON here")
    ap.add_argument("--max-xy-half", type=float, default=0.09,
                    help="(manifest mode) auto-fit-scale: shrink so baked horizontal "
                         "half-extent <= this many metres (0 disables)")
    ap.add_argument("--overwrite", action="store_true",
                    help="(manifest mode) re-convert keys that already exist (default: skip)")
    ap.add_argument("--contain-region", nargs="*", default=[],
                    help="keys that should get a contain_region site (receptacles)")
    ap.add_argument("--scale", nargs="*", default=[],
                    help="key=scale overrides (e.g. rc_tray=0.8) -- absolute scale, replaces meta scale")
    ap.add_argument("--dest", default=DEST_DEFAULT)
    args = ap.parse_args()

    search_dirs = _default_search_dirs()

    if args.manifest:
        run_manifest(args, search_dirs)
        return

    if not (args.meta and args.map):
        raise SystemExit("[error] use either --manifest, or --meta with --map")
    meta = json.load(open(args.meta))
    by_inst = {i["instance"]: i for i in meta["instances"]}
    overrides = dict(kv.split("=", 1) for kv in args.scale)

    pairs = [kv.split("=", 1) for kv in args.map]
    for key, inst in pairs:
        if inst not in by_inst:
            raise SystemExit(f"[error] instance {inst} not in meta")
        rec = by_inst[inst]
        scale = float(overrides[key]) if key in overrides else rec["scale"]
        print(f"[convert] {key} <- {inst} (cat={rec['category']}, scale={scale}"
              f"{' [override]' if key in overrides else ''})")
        convert_one(rec["model_xml"], scale, key, args.dest,
                    contain_region=(key in args.contain_region), search_dirs=search_dirs)
    print("[done] converted", len(pairs), "objects ->", args.dest)


if __name__ == "__main__":
    main()
