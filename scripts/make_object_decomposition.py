# Copyright (c) 2026, Cross-Embodiment CL Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Offline CoACD convex decomposition of a task object's collision mesh (YCB apple, hammer, ...).

Newton's runtime ``convexDecomposition`` importer calls CoACD with ``merge=False``, so its own
hull-count cap never applies and the apple ends up as ~60 hulls -- too many contacts/constraints
for large environment counts (see ``AGENTS.md``). This script instead runs CoACD once, offline,
with ``merge=True`` (so the hull-count cap actually merges down to it) and ``decimate=True`` (so
``max_ch_vertex`` actually caps each hull's vertex count -- without it, CoACD's returned "hulls"
keep the full source tessellation of each convex region, thousands of vertices apiece, and the
vertex cap is a no-op). The result is authored as a new USD asset next to the original: N child
Mesh prims, each already a convex hull, so any backend just imports them -- no importer-side
decomposition, no importer-side cap that does not do what it says.

Decimation shrinks each hull inside the true surface, which leaves visible valleys between hulls
at low hull and vertex counts. ``--extrude-margin`` enables CoACD's extrusion of neighbouring hulls
across their shared faces to close them, and the script prints how far the source surface lies
outside the hull union so the fit can be judged without a viewer.

``--sdf-max-resolution R`` authors each hull as a triangle mesh (``physics:approximation = none``)
with ``NewtonSDFCollisionAPI``, so Newton builds an SDF per hull. That only takes effect with
Newton's collision pipeline (``env.sim.physics.solver_cfg.use_mujoco_contacts=false``); MuJoCo's own
contacts and PhysX convexify the hull meshes instead, which is harmless because each is convex.

The new asset keeps the original's visual mesh, material, and rigid-body mass setup (via its
``configuration/appearance.usda`` sublayer, plus a byte-for-byte ``Sdf.CopySpec`` of the physics
material and the source's explicit ``physics:mass``); only the collision geometry is replaced.

Usage:
    uv run python scripts/make_object_decomposition.py --usd-path assets/objects/YcbHammer/textured.usda \
        [--hulls N] [--max-verts V] [--threshold T] [--extrude-margin M] [--sdf-max-resolution R] [--output PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import coacd
import numpy as np
import trimesh
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from scipy.spatial import ConvexHull

_DEFAULT_USD_PATH = Path(__file__).resolve().parents[1] / "assets/objects/YcbApple/textured.usda"
_COLLISION_MESH_PATH = "/Object/geometry"
_OBJECT_PATH = "/Object"
_MATERIAL_PATH = "/Object/physicsMaterial"


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

    # Apply the mesh's local transform relative to the object root consistently, so the emitted
    # hulls line up even if a future source asset nests the collision mesh under a transform.
    object_prim = stage.GetPrimAtPath(_OBJECT_PATH)
    mesh_to_world = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    object_to_world = UsdGeom.Xformable(object_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    mesh_to_object = mesh_to_world * object_to_world.GetInverse()
    identity = type(mesh_to_object)(1.0)
    if mesh_to_object != identity:
        points = np.array([list(mesh_to_object.Transform(p)) for p in points], dtype=np.float64)
    return points, faces


def _report_fit(points: np.ndarray, faces: np.ndarray, hulls: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """Print how far the source surface lies outside the hull union, and how much the hulls bloat it.

    A surface sample's distance outside a convex hull is bounded below by its largest facet-plane
    distance; the minimum over hulls is its distance outside the union. Large values are the
    valleys a fingertip can sink into.
    """
    source = trimesh.Trimesh(points, faces, process=False)
    samples, _ = trimesh.sample.sample_surface(source, 20000, seed=0)
    outside = np.full(len(samples), np.inf)
    hull_volume = 0.0
    for verts, _ in hulls:
        hull = ConvexHull(verts)
        hull_volume += hull.volume
        plane_distance = samples @ hull.equations[:, :3].T + hull.equations[:, 3]
        outside = np.minimum(outside, plane_distance.max(axis=1))
    gap_mm = np.clip(outside, 0.0, None) * 1000.0
    extent_mm = (points.max(axis=0) - points.min(axis=0)) * 1000.0
    print(f"Source extent [mm]: {np.array2string(extent_mm, precision=1)}")
    print(
        f"Surface outside hull union [mm]: mean {gap_mm.mean():.2f}, p95 {np.percentile(gap_mm, 95):.2f}, "
        f"max {gap_mm.max():.2f}; within 1 mm: {100.0 * (gap_mm <= 1.0).mean():.1f}%"
    )
    if source.is_volume:
        print(f"Hull volume sum / source volume: {hull_volume / source.volume:.3f} (overlaps count twice)")


def _write_decomposed_asset(
    src_usd_path: Path,
    output_path: Path,
    hulls: list[tuple[np.ndarray, np.ndarray]],
    sdf_max_resolution: int,
) -> None:
    """Author a new USD next to the original: same visuals/material/mass, N convex hull colliders."""
    src_stage = Usd.Stage.Open(str(src_usd_path))
    src_flat = src_stage.Flatten()

    mass = UsdPhysics.MassAPI(src_stage.GetPrimAtPath(_OBJECT_PATH)).GetMassAttr().Get()
    object_name_attr = src_stage.GetPrimAtPath(_OBJECT_PATH).GetAttribute("crossEmbodiment:objectName")
    object_name = object_name_attr.Get() if object_name_attr.IsValid() else None

    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    # Same appearance sublayer as the original: the visual mesh, its material, and the "over
    # geometry { visibility = invisible }" opinion (harmless here, since this file never
    # ``def``s a "geometry" prim for it to apply to). Only valid when the new asset is written
    # next to the source, which is the default and the only case exercised here.
    appearance_path = src_usd_path.parent / "configuration/appearance.usda"
    if not appearance_path.is_file():
        raise RuntimeError(f"Expected sibling appearance sublayer at {appearance_path}")
    if output_path.parent != src_usd_path.parent:
        raise RuntimeError("Output must be written next to the source asset for the relative sublayer to resolve.")
    stage.GetRootLayer().subLayerPaths.append("./configuration/appearance.usda")

    object_prim = UsdGeom.Xform.Define(stage, _OBJECT_PATH).GetPrim()
    stage.SetDefaultPrim(object_prim)
    UsdPhysics.RigidBodyAPI.Apply(object_prim)
    mass_api = UsdPhysics.MassAPI.Apply(object_prim)
    mass_api.CreateMassAttr(mass)
    if object_name is not None:
        object_prim.CreateAttribute("crossEmbodiment:objectName", Sdf.ValueTypeNames.String, custom=True).Set(
            object_name
        )

    # Byte-for-byte copy of the physics material (friction/restitution + PhysX combine modes),
    # since the PhysxMaterialAPI python bindings are not available in this environment.
    if not Sdf.CopySpec(src_flat, Sdf.Path(_MATERIAL_PATH), stage.GetRootLayer(), Sdf.Path(_MATERIAL_PATH)):
        raise RuntimeError(f"Failed to copy {_MATERIAL_PATH} from {src_usd_path}")
    material = UsdShade.Material(stage.GetPrimAtPath(_MATERIAL_PATH))

    print(f"CoACD produced {len(hulls)} hull(s):")
    for i, (verts, faces) in enumerate(hulls):
        name = f"hull{i}"
        mesh = UsdGeom.Mesh.Define(stage, f"{_OBJECT_PATH}/{name}")
        mesh.CreatePointsAttr([tuple(v) for v in verts])
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr([int(i) for face in faces for i in face])
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        # Collision-only, like the source ``geometry`` mesh (kept invisible by the sublayered
        # appearance.usda's "over" opinion): without this, Newton's importer treats a hull as
        # viewport geometry too (default purpose is visible) and keeps an extra visual-only
        # copy of it alongside its collision shape, doubling the shape count for no reason.
        mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        prim = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
        if sdf_max_resolution > 0:
            # Newton's importer ignores physics:approximation on an SDF prim but warns unless it is "none".
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
            prim.AddAppliedSchema("NewtonSDFCollisionAPI")
            prim.CreateAttribute("newton:sdfMaxResolution", Sdf.ValueTypeNames.Int).Set(sdf_max_resolution)
        else:
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
        print(f"  {name}: {len(verts)} vertices, {len(faces)} faces")

    stage.GetRootLayer().Save()
    print(f"Wrote {output_path}")
    print(f"Source mass preserved: {mass} kg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--usd-path", type=Path, default=_DEFAULT_USD_PATH)
    parser.add_argument("--hulls", type=int, default=8, help="Cap on the number of convex hulls.")
    parser.add_argument("--max-verts", type=int, default=32, help="Cap on vertices per convex hull.")
    parser.add_argument("--threshold", type=float, default=0.05, help="CoACD concavity threshold.")
    parser.add_argument(
        "--extrude-margin", type=float, default=0.0, help="Extrude hulls across shared faces by this much [m]; 0 is off."
    )
    parser.add_argument(
        "--sdf-max-resolution", type=int, default=0, help="Author Newton SDF hull meshes at this resolution; 0 is off."
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Defaults to <name>_coacd<N>[_sdf<R>].usda next to the source."
    )
    args = parser.parse_args()

    usd_path = args.usd_path.resolve()
    output_path = args.output.resolve() if args.output is not None else None
    if output_path is None:
        suffix = f"_coacd{args.hulls}" + (f"_sdf{args.sdf_max_resolution}" if args.sdf_max_resolution > 0 else "")
        output_path = usd_path.with_name(f"{usd_path.stem}{suffix}{usd_path.suffix}")

    stage = Usd.Stage.Open(str(usd_path))
    points, faces = _load_collision_mesh(stage)
    mesh = coacd.Mesh(points, faces)
    # decimate=True is required for max_ch_vertex to actually cap vertex count (see module
    # docstring); without it CoACD leaves each convex piece at its full source tessellation.
    result = coacd.run_coacd(
        mesh,
        threshold=args.threshold,
        max_convex_hull=args.hulls,
        merge=True,
        decimate=True,
        max_ch_vertex=args.max_verts,
        extrude=args.extrude_margin > 0.0,
        extrude_margin=args.extrude_margin,
    )
    _report_fit(points, faces, result)
    _write_decomposed_asset(usd_path, output_path, result, args.sdf_max_resolution)


if __name__ == "__main__":
    main()
