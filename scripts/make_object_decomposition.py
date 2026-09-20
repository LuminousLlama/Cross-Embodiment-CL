# Copyright (c) 2026, Cross-Embodiment CL Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Build the reviewed offline collision asset for one task object.

The source object directory must contain ``textured.usda`` and a
``collision_spec.json``. The spec selects one convex hull for a near-convex
object, a CoACD decomposition for a concave one, or overlapping longitudinal
slices for an elongated object whose CoACD boundaries leave visible divots.
The generated
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
import trimesh
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from scipy.spatial import ConvexHull

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
    if not isinstance(spec, dict) or spec.get("mode") not in {"hull", "coacd", "slices"}:
        raise RuntimeError(f"{spec_path} must set mode to 'hull', 'coacd', or 'slices'")
    if "threshold" in spec and (not isinstance(spec["threshold"], (int, float)) or isinstance(spec["threshold"], bool)):
        raise RuntimeError(f"{spec_path} threshold must be a number")
    max_hulls = spec.get("max_hulls")
    if max_hulls is not None and (not isinstance(max_hulls, int) or isinstance(max_hulls, bool) or max_hulls < 1):
        raise RuntimeError(f"{spec_path} max_hulls must be a positive integer")
    slice_count = spec.get("slice_count")
    if slice_count is not None and (
        not isinstance(slice_count, int) or isinstance(slice_count, bool) or slice_count < 2
    ):
        raise RuntimeError(f"{spec_path} slice_count must be an integer of at least 2")
    slice_overlap = spec.get("slice_overlap")
    if slice_overlap is not None and (
        not isinstance(slice_overlap, (int, float)) or isinstance(slice_overlap, bool) or not 0.0 <= slice_overlap < 0.5
    ):
        raise RuntimeError(f"{spec_path} slice_overlap must be in [0, 0.5)")
    slice_boundaries = spec.get("slice_boundaries")
    if slice_boundaries is not None and (
        not isinstance(slice_boundaries, list)
        or not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in slice_boundaries)
        or slice_boundaries != sorted(slice_boundaries)
        or any(not 0.0 < value < 1.0 for value in slice_boundaries)
    ):
        raise RuntimeError(f"{spec_path} slice_boundaries must be an increasing list of fractions in (0, 1)")
    slice_overlaps = spec.get("slice_overlaps")
    if slice_overlaps is not None and (
        not isinstance(slice_overlaps, list)
        or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= value < 0.5
            for value in slice_overlaps
        )
        or slice_boundaries is None
        or len(slice_overlaps) != len(slice_boundaries)
    ):
        raise RuntimeError(f"{spec_path} slice_overlaps must match slice_boundaries with values in [0, 0.5)")
    return spec


def _slice_convex_hulls(
    points: np.ndarray,
    count: int,
    overlap: float,
    boundary_fractions: list[float] | None = None,
    boundary_overlaps: list[float] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build overlapping convex hulls along the mesh's principal axis."""
    principal_axis = np.linalg.eigh(np.cov(points.T))[1][:, -1]
    projections = points @ principal_axis
    if boundary_fractions is None:
        boundary_fractions = np.linspace(0.0, 1.0, count + 1)[1:-1].tolist()
    fractions = np.array([0.0, *boundary_fractions, 1.0])
    boundaries = projections.min() + fractions * np.ptp(projections)
    widths = np.diff(boundaries)
    overlaps = boundary_overlaps or [overlap] * (len(boundaries) - 2)
    hulls = []
    for index in range(len(boundaries) - 1):
        lower_overlap = 0.0 if index == 0 else overlaps[index - 1] * min(widths[index - 1 : index + 1])
        upper_overlap = 0.0 if index == len(boundaries) - 2 else overlaps[index] * min(widths[index : index + 2])
        lower = boundaries[index] - lower_overlap
        upper = boundaries[index + 1] + upper_overlap
        slice_points = points[(projections >= lower) & (projections <= upper)]
        source_hull = ConvexHull(slice_points)
        source_vertices = slice_points[source_hull.vertices]
        source_remap = {old: new for new, old in enumerate(source_hull.vertices)}
        source_faces = np.array(
            [[source_remap[vertex] for vertex in face] for face in source_hull.simplices], dtype=np.int64
        )

        simplified = trimesh.Trimesh(source_vertices, source_faces, process=False).simplify_quadric_decimation(
            face_count=124
        )
        final_hull = ConvexHull(simplified.vertices)
        final_vertices = np.asarray(simplified.vertices)[final_hull.vertices]
        final_remap = {old: new for new, old in enumerate(final_hull.vertices)}
        final_faces = np.array(
            [[final_remap[vertex] for vertex in face] for face in final_hull.simplices], dtype=np.int64
        )
        hulls.append((final_vertices, final_faces))
    return hulls


def _write_decomposed_asset(src_usd_path: Path, output_path: Path, hulls: list[tuple[np.ndarray, np.ndarray]]) -> None:
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
        object_prim.CreateAttribute("crossEmbodiment:objectName", Sdf.ValueTypeNames.String, custom=True).Set(
            object_name
        )

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
    if mode == "slices":
        hulls = _slice_convex_hulls(
            points,
            spec.get("slice_count", 5),
            spec.get("slice_overlap", 0.15),
            spec.get("slice_boundaries"),
            spec.get("slice_overlaps"),
        )
    else:
        max_convex_hull = 1 if mode == "hull" else spec.get("max_hulls", -1)
        hulls = coacd.run_coacd(
            coacd.Mesh(points, faces),
            threshold=spec.get("threshold", 0.05),
            max_convex_hull=max_convex_hull,
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
