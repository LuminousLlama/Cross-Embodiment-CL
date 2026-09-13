# Copyright (c) 2026, Cross-Embodiment CL Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Build the reviewed offline collision asset for one task object.

The source object directory must contain ``textured.usda`` and a
``collision_spec.json``. The spec selects either one convex hull for a
near-convex object or a CoACD decomposition for a concave one. The generated
``textured_collision.usda`` keeps the source visual mesh, physics material,
and explicit mass, while replacing its collision geometry with invisible,
explicit convex hulls.

Usage:
    uv run python scripts/make_object_decomposition.py assets/objects/YcbApple
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import coacd
import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

_COLLISION_MESH_PATH = "/Object/geometry"
_OBJECT_PATH = "/Object"
_MATERIAL_PATH = "/Object/physicsMaterial"
_SPEC_FILENAME = "collision_spec.json"
_SOURCE_FILENAME = "textured.usda"
_OUTPUT_FILENAME = "textured_collision.usda"


def _triangulate(face_vertex_counts: list[int], face_vertex_indices: list[int]) -> np.ndarray:
    """Fan-triangulate a polygonal mesh's face-vertex arrays into an (F, 3) index array."""
    faces = []
    cursor = 0
    for count in face_vertex_counts:
        face = face_vertex_indices[cursor : cursor + count]
        for i in range(1, count - 1):
            faces.append((face[0], face[i], face[i + 1]))
        cursor += count
    return np.array(faces, dtype=np.int64)


def _load_collision_mesh(stage: Usd.Stage) -> tuple[np.ndarray, np.ndarray]:
    """Read the source collision mesh's points and triangle faces in the ``/Object`` frame."""
    mesh_prim = stage.GetPrimAtPath(_COLLISION_MESH_PATH)
    if not mesh_prim.IsValid():
        raise RuntimeError(f"No prim at {_COLLISION_MESH_PATH} in {stage.GetRootLayer().identifier}")
    mesh = UsdGeom.Mesh(mesh_prim)
    points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    face_vertex_counts = list(mesh.GetFaceVertexCountsAttr().Get())
    face_vertex_indices = list(mesh.GetFaceVertexIndicesAttr().Get())
    if set(face_vertex_counts) == {3}:
        faces = np.array(face_vertex_indices, dtype=np.int64).reshape(-1, 3)
    else:
        faces = _triangulate(face_vertex_counts, face_vertex_indices)

    object_prim = stage.GetPrimAtPath(_OBJECT_PATH)
    mesh_to_world = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    object_to_world = UsdGeom.Xformable(object_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    mesh_to_object = mesh_to_world * object_to_world.GetInverse()
    identity = type(mesh_to_object)(1.0)
    if mesh_to_object != identity:
        points = np.array([list(mesh_to_object.Transform(point)) for point in points], dtype=np.float64)
    return points, faces


def _load_spec(object_dir: Path) -> dict[str, object]:
    """Read and validate the collision policy authored next to an object asset."""
    spec_path = object_dir / _SPEC_FILENAME
    try:
        spec = json.loads(spec_path.read_text())
    except FileNotFoundError as error:
        raise RuntimeError(f"Expected collision spec at {spec_path}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON in {spec_path}: {error}") from error
    if not isinstance(spec, dict) or spec.get("mode") not in {"hull", "coacd"}:
        raise RuntimeError(f"{spec_path} must set mode to 'hull' or 'coacd'")
    if "threshold" in spec and (not isinstance(spec["threshold"], (int, float)) or isinstance(spec["threshold"], bool)):
        raise RuntimeError(f"{spec_path} threshold must be a number")
    return spec


def _write_decomposed_asset(
    src_usd_path: Path, output_path: Path, hulls: list[tuple[np.ndarray, np.ndarray]]
) -> None:
    """Author a new USD with the source visual setup and explicit convex-hull colliders."""
    src_stage = Usd.Stage.Open(str(src_usd_path))
    src_flat = src_stage.Flatten()

    mass = UsdPhysics.MassAPI(src_stage.GetPrimAtPath(_OBJECT_PATH)).GetMassAttr().Get()
    object_name_attr = src_stage.GetPrimAtPath(_OBJECT_PATH).GetAttribute("crossEmbodiment:objectName")
    object_name = object_name_attr.Get() if object_name_attr.IsValid() else None

    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.GetRootLayer().subLayerPaths.append("./configuration/appearance.usda")

    object_prim = UsdGeom.Xform.Define(stage, _OBJECT_PATH).GetPrim()
    stage.SetDefaultPrim(object_prim)
    UsdPhysics.RigidBodyAPI.Apply(object_prim)
    UsdPhysics.MassAPI.Apply(object_prim).CreateMassAttr(mass)
    if object_name is not None:
        object_prim.CreateAttribute(
            "crossEmbodiment:objectName", Sdf.ValueTypeNames.String, custom=True
        ).Set(object_name)

    if not Sdf.CopySpec(src_flat, Sdf.Path(_MATERIAL_PATH), stage.GetRootLayer(), Sdf.Path(_MATERIAL_PATH)):
        raise RuntimeError(f"Failed to copy {_MATERIAL_PATH} from {src_usd_path}")
    material = UsdShade.Material(stage.GetPrimAtPath(_MATERIAL_PATH))

    for index, (vertices, faces) in enumerate(hulls):
        mesh = UsdGeom.Mesh.Define(stage, f"{_OBJECT_PATH}/hull{index}")
        mesh.CreatePointsAttr([tuple(vertex) for vertex in vertices])
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr([int(vertex) for face in faces for vertex in face])
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        prim = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")

    stage.GetRootLayer().Save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "object_dir", type=Path, help="Object directory containing textured.usda and collision_spec.json."
    )
    args = parser.parse_args()

    object_dir = args.object_dir.resolve()
    src_usd_path = object_dir / _SOURCE_FILENAME
    if not src_usd_path.is_file():
        raise RuntimeError(f"Expected source asset at {src_usd_path}")
    spec = _load_spec(object_dir)
    points, faces = _load_collision_mesh(Usd.Stage.Open(str(src_usd_path)))
    mode = spec["mode"]
    hulls = coacd.run_coacd(
        coacd.Mesh(points, faces),
        threshold=spec.get("threshold", 0.05),
        max_convex_hull=1 if mode == "hull" else -1,
        merge=True,
        decimate=True,
        max_ch_vertex=64,
        seed=0,
    )
    output_path = object_dir / _OUTPUT_FILENAME
    _write_decomposed_asset(src_usd_path, output_path, hulls)
    print(f"Hulls: {len(hulls)}")
    print("Vertices per hull: " + ", ".join(str(len(vertices)) for vertices, _ in hulls))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
