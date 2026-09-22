"""Build reviewed offline convex colliders for a standalone hand URDF."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import coacd
import numpy as np
import trimesh
from hand_asset_specs import HAND_SPECS, HandAssetSpec

MIN_HULL_VOLUME_M3 = 1.0e-7


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Expected one triangle mesh in {path}")
    mesh.remove_unreferenced_vertices()
    return mesh


def _prepare_inspire_visuals(spec: HandAssetSpec) -> None:
    mesh_dir = spec.urdf_path.parent / "meshes"
    for link_name in spec.mesh_collision_links:
        path = mesh_dir / f"{link_name}{spec.mesh_suffix}"
        mesh = _load_mesh(path)
        if link_name == "right_little_2" and mesh.extents[0] > 2.0 * mesh.extents[2]:
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2.0, (0.0, 1.0, 0.0)))
        mesh.export(path)


def _generate_hulls(spec: HandAssetSpec, link_name: str, threshold: float, max_hulls: int) -> list[Path]:
    mesh_dir = spec.urdf_path.parent / "meshes"
    collider_dir = mesh_dir / "collision_hulls"
    source_path = mesh_dir / f"{link_name}{spec.mesh_suffix}"
    mesh = _load_mesh(source_path)
    hulls = coacd.run_coacd(
        coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)),
        threshold=threshold,
        max_convex_hull=max_hulls,
        merge=True,
        decimate=True,
        max_ch_vertex=64,
        seed=0,
    )
    for stale_path in collider_dir.glob(f"{link_name}_hull_*.obj"):
        stale_path.unlink()

    paths = []
    for vertices, faces in hulls:
        hull = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        if hull.volume < MIN_HULL_VOLUME_M3:
            continue
        if not hull.is_watertight or not hull.is_convex or len(hull.vertices) > 64:
            raise RuntimeError(
                f"Invalid hull for {link_name}: watertight={hull.is_watertight}, "
                f"convex={hull.is_convex}, vertices={len(hull.vertices)}"
            )
        path = collider_dir / f"{link_name}_hull_{len(paths)}.obj"
        hull.export(path)
        paths.append(path)
    if not paths:
        raise RuntimeError(f"CoACD returned no usable hulls for {link_name}")
    return paths


def _collision_xml(spec: HandAssetSpec, link_name: str, hull_paths: list[Path]) -> str:
    blocks = []
    for index, path in enumerate(hull_paths):
        relative_path = path.relative_to(spec.urdf_path.parent).as_posix()
        prefix = "./" if spec.key == "inspire" else ""
        blocks.append(
            f'    <collision name="{link_name}_hull_{index}">\n'
            '      <origin xyz="0 0 0" rpy="0 0 0"/>\n'
            "      <geometry>\n"
            f'        <mesh filename="{prefix}{relative_path}" scale="1 1 1"/>\n'
            "      </geometry>\n"
            "    </collision>"
        )
    return "\n".join(blocks)


def _rewrite_urdf(spec: HandAssetSpec, colliders: dict[str, list[Path]]) -> None:
    text = spec.urdf_path.read_text()
    for link_name, hull_paths in colliders.items():
        pattern = re.compile(rf'(<link name="{re.escape(link_name)}">)(.*?)(\n  </link>)', re.DOTALL)
        match = pattern.search(text)
        if match is None:
            raise RuntimeError(f"Could not find URDF link {link_name}")
        body = match.group(2)
        if spec.key == "inspire":
            visuals = re.findall(r"\n    <visual>.*?</visual>", body, flags=re.DOTALL)
            render_visuals = [block for block in visuals if "_visuals.obj" in block]
            if len(render_visuals) != 1:
                raise RuntimeError(f"Expected one render visual for {link_name}, found {len(render_visuals)}")
            body = re.sub(r"\n    <visual>.*?</visual>", "", body, flags=re.DOTALL)
            body = f"{body}\n{render_visuals[0].lstrip()}"
        body = re.sub(r'\n    <collision(?:\s+name="[^"]+")?>.*?</collision>', "", body, flags=re.DOTALL)
        body = f"{body}\n{_collision_xml(spec, link_name, hull_paths)}"
        text = f"{text[: match.start()]}{match.group(1)}{body}{match.group(3)}{text[match.end() :]}"
    spec.urdf_path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", required=True, choices=HAND_SPECS)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--palm-max-hulls", type=int, default=6)
    parser.add_argument("--finger-max-hulls", type=int, default=1)
    args = parser.parse_args()
    if args.threshold <= 0.0 or min(args.palm_max_hulls, args.finger_max_hulls) < 1:
        raise ValueError("Threshold must be positive and hull limits must be at least one")

    spec = HAND_SPECS[args.hand]
    (spec.urdf_path.parent / "meshes/collision_hulls").mkdir(parents=True, exist_ok=True)
    if spec.key == "inspire":
        _prepare_inspire_visuals(spec)
    colliders = {
        name: _generate_hulls(
            spec,
            name,
            args.threshold,
            args.palm_max_hulls if name == spec.palm_link_name else args.finger_max_hulls,
        )
        for name in spec.mesh_collision_links
    }
    _rewrite_urdf(spec, colliders)
    print(f"Generated {sum(map(len, colliders.values()))} convex hulls for {len(colliders)} links")
    print(f"Updated {spec.urdf_path}")


if __name__ == "__main__":
    main()
