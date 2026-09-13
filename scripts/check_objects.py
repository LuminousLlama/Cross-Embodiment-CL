# Copyright (c) 2026, Cross-Embodiment CL Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Headless CPU sanity check of the reviewed offline collision assets under ``assets/objects/``.

For each object's ``textured_collision.usda``, prints the hull count, total hull vertex count,
bounding-box size (from the union of hull points), authored mass, and confirms every hull prim is
collision-enabled and invisible while the visual mesh stays visible. Flags any object whose
bounding-box size suggests wrong units (larger than 0.5 m or smaller than 1 cm on any axis).

Usage:
    uv run python scripts/check_objects.py
    uv run python scripts/check_objects.py assets/objects/YcbApple assets/objects/YcbHammer
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

_OBJECT_PATH = "/Object"
_VISUAL_MESH_PATH = "/Object/visualGeometry"
_OUTPUT_FILENAME = "textured_collision.usda"
_MIN_EXTENT_M = 0.01
_MAX_EXTENT_M = 0.5


def _hull_prims(stage: Usd.Stage) -> list[Usd.Prim]:
    """Every ``/Object/hull*`` collision mesh prim, in authoring order."""
    object_prim = stage.GetPrimAtPath(_OBJECT_PATH)
    return [child for child in object_prim.GetChildren() if child.GetName().startswith("hull")]


def _check_object(object_dir: Path) -> bool:
    """Print one object's summary and return whether it looks correct."""
    usd_path = object_dir / _OUTPUT_FILENAME
    if not usd_path.is_file():
        print(f"{object_dir.name}: MISSING {_OUTPUT_FILENAME}")
        return False

    stage = Usd.Stage.Open(str(usd_path))
    hull_prims = _hull_prims(stage)
    if not hull_prims:
        print(f"{object_dir.name}: no hull prims found")
        return False

    all_points = []
    invisible_ok = True
    collision_ok = True
    total_vertices = 0
    for prim in hull_prims:
        mesh = UsdGeom.Mesh(prim)
        points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
        total_vertices += len(points)
        all_points.append(points)
        if mesh.GetVisibilityAttr().Get() != UsdGeom.Tokens.invisible:
            invisible_ok = False
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_ok = False

    bbox = np.concatenate(all_points, axis=0)
    bbox_size = bbox.max(axis=0) - bbox.min(axis=0)

    visual_prim = stage.GetPrimAtPath(_VISUAL_MESH_PATH)
    visual_visibility = UsdGeom.Mesh(visual_prim).GetVisibilityAttr().Get() if visual_prim.IsValid() else None
    visual_kept = visual_prim.IsValid() and visual_visibility != UsdGeom.Tokens.invisible

    mass = UsdPhysics.MassAPI(stage.GetPrimAtPath(_OBJECT_PATH)).GetMassAttr().Get()

    bad_units = bool((bbox_size > _MAX_EXTENT_M).any() or (bbox_size < _MIN_EXTENT_M).any())
    ok = invisible_ok and collision_ok and visual_kept and not bad_units

    print(
        f"{object_dir.name}: hulls={len(hull_prims)} vertices={total_vertices} "
        f"bbox_m=({bbox_size[0]:.4f}, {bbox_size[1]:.4f}, {bbox_size[2]:.4f}) mass_kg={mass} "
        f"collision_invisible={invisible_ok} collision_enabled={collision_ok} visual_kept={visual_kept}"
    )
    if bad_units:
        print(f"{object_dir.name}: FLAG bbox size suggests wrong units")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "object_dirs",
        type=Path,
        nargs="*",
        help="Object directories to check; defaults to every assets/objects/Ycb* directory.",
    )
    args = parser.parse_args()

    object_dirs = args.object_dirs or sorted((Path(__file__).resolve().parents[1] / "assets/objects").glob("Ycb*"))
    all_ok = True
    for object_dir in object_dirs:
        all_ok &= _check_object(object_dir.resolve())
    if not all_ok:
        raise SystemExit("One or more objects failed the check.")


if __name__ == "__main__":
    main()
