"""Rebuild and validate a layered PhysX/Newton standalone hand USD."""

from __future__ import annotations

import argparse
import math
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from hand_asset_specs import HAND_SPECS, HandAssetSpec

from isaaclab.app import AppLauncher, add_launcher_args, launch_simulation
from isaaclab.utils.version import standalone_importers_available

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--hand", required=True, choices=HAND_SPECS)
parser.add_argument("--output-dir", type=Path)
add_launcher_args(parser)
args = parser.parse_args()
args.require_kit = not standalone_importers_available()
args.physics = "isaacsim_physx" if args.require_kit else "newton_mjwarp"
if args.require_kit and not AppLauncher.is_available():
    raise ImportError("Export requires Isaac Sim or the standalone Isaac asset importer.")

from isaaclab.physics import PhysicsCfg  # noqa: E402


def _ensure_api_schema(prim: Usd.Prim, schema_name: str) -> None:
    if schema_name in prim.GetAppliedSchemas():
        return
    local_schemas = prim.GetMetadata("apiSchemas") or Sdf.TokenListOp()
    prepended = list(local_schemas.prependedItems)
    if schema_name not in prepended:
        prepended.append(schema_name)
    local_schemas.prependedItems = prepended
    prim.SetMetadata("apiSchemas", local_schemas)


def _author_portable_defaults(spec: HandAssetSpec, generated_dir: Path) -> None:
    stage = Usd.Stage.Open(str(generated_dir / "payloads/Physics/physics.usda"))
    if stage is None:
        raise RuntimeError("Could not open generated common physics layer")
    for joint_name in spec.joint_names:
        prim = stage.GetPrimAtPath(f"{spec.root_prim}/Physics/{joint_name}")
        if not prim:
            raise RuntimeError(f"Missing imported joint {joint_name}")
        joint = UsdPhysics.RevoluteJoint(prim)
        lower, upper = spec.joint_limits[joint_name]
        joint.GetLowerLimitAttr().Set(math.degrees(lower))
        joint.GetUpperLimitAttr().Set(math.degrees(upper))
        position_attr = prim.GetAttribute("state:angular:physics:position")
        if not position_attr:
            raise RuntimeError(f"Missing angular joint state on {joint_name}")
        position_attr.Set(math.degrees(spec.open_joint_pos[joint_name]))
        prim.CreateAttribute("newton:armature", Sdf.ValueTypeNames.Double).Set(spec.armature)
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        drive.GetStiffnessAttr().Set(math.radians(spec.stiffness[joint_name]))
        drive.GetDampingAttr().Set(math.radians(spec.damping[joint_name]))
        if joint_name in spec.active_joint_names:
            drive.GetTargetPositionAttr().Set(math.degrees(spec.open_joint_pos[joint_name]))
        else:
            drive.GetTargetPositionAttr().Clear()

    root_joint_prim = stage.GetPrimAtPath(f"{spec.root_prim}/Physics/{spec.root_joint_name}")
    root_joint = UsdPhysics.FixedJoint(root_joint_prim)
    if not root_joint:
        raise RuntimeError(f"Missing fixed root joint {spec.root_joint_name}")
    # A world-fixed joint is represented by an empty body0. Targeting the
    # package Xform happens to simulate as a world weld, but Robot Assembler
    # cannot recognize and disable that non-canonical form.
    root_joint.GetBody0Rel().SetTargets([])
    root_joint.GetBody1Rel().SetTargets([Sdf.Path(spec.root_body_path)])

    for body_path, filtered_body_path in spec.collision_filter_pairs:
        body = stage.GetPrimAtPath(body_path)
        filtered_body = stage.GetPrimAtPath(filtered_body_path)
        if not body or not filtered_body:
            raise RuntimeError(f"Missing self-collision filter body: {body_path}, {filtered_body_path}")
        UsdPhysics.FilteredPairsAPI.Apply(body).CreateFilteredPairsRel().SetTargets([Sdf.Path(filtered_body_path)])
    stage.GetRootLayer().customLayerData = {
        **stage.GetRootLayer().customLayerData,
        spec.rom_contract_key: spec.rom_contract,
    }
    stage.GetRootLayer().Save()


def _author_physx_defaults(spec: HandAssetSpec, generated_dir: Path) -> None:
    stage = Usd.Stage.Open(str(generated_dir / "payloads/Physics/physx.usda"))
    if stage is None:
        raise RuntimeError("Could not open generated PhysX layer")
    root = stage.GetPrimAtPath(spec.root_body_path)
    if not root:
        raise RuntimeError(f"Missing articulation root body {spec.root_body_path}")
    _ensure_api_schema(root, "PhysxArticulationAPI")
    root.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(True)
    for joint_name in spec.joint_names:
        prim = stage.GetPrimAtPath(f"{spec.root_prim}/Physics/{joint_name}")
        _ensure_api_schema(prim, "PhysxJointAPI")
        prim.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float).Set(spec.armature)
    stage.GetRootLayer().Save()


def _author_display_pose(spec: HandAssetSpec, generated_dir: Path) -> None:
    """Bake nonzero initial joint positions into stopped-timeline link transforms."""
    base_stage = Usd.Stage.Open(str(generated_dir / "payloads/base.usda"))
    physics_stage = Usd.Stage.Open(str(generated_dir / "payloads/Physics/physics.usda"))
    if base_stage is None or physics_stage is None:
        raise RuntimeError("Could not open generated layers for display-pose authoring")

    axis_vectors = {
        "X": Gf.Vec3d(1.0, 0.0, 0.0),
        "Y": Gf.Vec3d(0.0, 1.0, 0.0),
        "Z": Gf.Vec3d(0.0, 0.0, 1.0),
    }
    for joint_name, position in spec.open_joint_pos.items():
        if abs(position) < 1.0e-12:
            continue
        joint = UsdPhysics.RevoluteJoint(physics_stage.GetPrimAtPath(f"{spec.root_prim}/Physics/{joint_name}"))
        body1_targets = joint.GetBody1Rel().GetTargets()
        if len(body1_targets) != 1:
            raise RuntimeError(f"Expected one child body for {joint_name}, got {body1_targets}")
        local_pos1 = joint.GetLocalPos1Attr().Get()
        local_rot1 = Gf.Quatd(joint.GetLocalRot1Attr().Get())
        local_rot1_is_identity = (
            abs(abs(local_rot1.GetReal()) - 1.0) <= 1.0e-9 and local_rot1.GetImaginary().GetLength() <= 1.0e-9
        )
        if local_pos1.GetLength() > 1.0e-9 or not local_rot1_is_identity:
            raise RuntimeError(f"Display-pose authoring requires an identity child joint frame: {joint_name}")

        body = base_stage.GetPrimAtPath(body1_targets[0])
        orientation = body.GetAttribute("xformOp:orient")
        if not orientation:
            raise RuntimeError(f"Missing xformOp:orient on child body for {joint_name}")
        imported_orientation = Gf.Quatd(orientation.Get())
        axis = axis_vectors[str(joint.GetAxisAttr().Get())]
        delta = Gf.Rotation(axis, math.degrees(position)).GetQuat()
        orientation.Set(Gf.Quatf(imported_orientation * delta))

    base_stage.GetRootLayer().Save()


def _author_robot_relationships(spec: HandAssetSpec, generated_path: Path) -> None:
    composed_stage = Usd.Stage.Open(str(generated_path))
    if composed_stage is None:
        raise RuntimeError(f"Could not open generated asset {generated_path}")

    def paths_with_schema(schema_name: str) -> list[Sdf.Path]:
        paths = []
        for prim in composed_stage.Traverse():
            schemas = prim.GetMetadata("apiSchemas")
            if schemas and schema_name in schemas.GetAppliedItems():
                paths.append(prim.GetPath())
        return paths

    link_paths = paths_with_schema("IsaacLinkAPI")
    joint_paths = paths_with_schema("IsaacJointAPI")
    if not link_paths or not joint_paths:
        raise RuntimeError(
            f"Importer produced incomplete Robot API metadata: {len(link_paths)} links, {len(joint_paths)} joints"
        )

    base_stage = Usd.Stage.Open(str(generated_path.parent / "payloads/base.usda"))
    if base_stage is None:
        raise RuntimeError("Could not open generated base layer")
    root = base_stage.GetPrimAtPath(spec.root_prim)
    root.GetRelationship("isaac:physics:robotLinks").SetTargets(link_paths)
    root.GetRelationship("isaac:physics:robotJoints").SetTargets(joint_paths)
    # The importer may embed its disposable temporary directory in this field.
    base_stage.GetRootLayer().documentation = ""
    base_stage.GetRootLayer().Save()


def _normalize_usda_files(directory: Path) -> None:
    for path in directory.rglob("*.usda"):
        path.write_text(path.read_text().rstrip() + "\n")


def _validate_asset(spec: HandAssetSpec, asset_path: Path) -> None:
    stage = Usd.Stage.Open(str(asset_path))
    if stage is None:
        raise RuntimeError(f"Could not open generated asset {asset_path}")
    root = stage.GetDefaultPrim()
    selection = root.GetVariantSets().GetVariantSet("Physics").GetVariantSelection()
    if selection != "physx":
        raise RuntimeError(f"Expected default Physics variant 'physx', got {selection!r}")

    for joint_name in spec.joint_names:
        prim = stage.GetPrimAtPath(f"{spec.root_prim}/Physics/{joint_name}")
        joint = UsdPhysics.RevoluteJoint(prim)
        actual_limits = (
            math.radians(joint.GetLowerLimitAttr().Get()),
            math.radians(joint.GetUpperLimitAttr().Get()),
        )
        if any(
            abs(actual - expected) > 1.0e-6
            for actual, expected in zip(actual_limits, spec.joint_limits[joint_name], strict=True)
        ):
            raise RuntimeError(f"Incorrect limits for {joint_name}: {actual_limits}")
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        actual_k = math.degrees(drive.GetStiffnessAttr().Get())
        actual_d = math.degrees(drive.GetDampingAttr().Get())
        if abs(actual_k - spec.stiffness[joint_name]) > 1.0e-4 or abs(actual_d - spec.damping[joint_name]) > 1.0e-4:
            raise RuntimeError(f"Incorrect drive gains for {joint_name}: K={actual_k}, D={actual_d}")
        for attribute in ("physxJoint:armature", "newton:armature"):
            value = prim.GetAttribute(attribute).Get()
            if value is None or abs(value - spec.armature) > 1.0e-9:
                raise RuntimeError(f"Incorrect {attribute} for {joint_name}: {value}")

    root_body = stage.GetPrimAtPath(spec.root_body_path)
    if root_body.GetAttribute("physxArticulation:enabledSelfCollisions").Get() is not True:
        raise RuntimeError("PhysX self-collision is not enabled")
    robot_links = root.GetRelationship("isaac:physics:robotLinks").GetTargets()
    robot_joints = root.GetRelationship("isaac:physics:robotJoints").GetTargets()
    if len(robot_links) != len(spec.joint_names) + 1 or Sdf.Path(spec.root_body_path) not in robot_links:
        raise RuntimeError(f"Incomplete Robot API link targets: {robot_links}")
    if len(robot_joints) != len(spec.joint_names) + 1:
        raise RuntimeError(f"Incomplete Robot API joint targets: {robot_joints}")
    root_joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(f"{spec.root_prim}/Physics/{spec.root_joint_name}"))
    if root_joint.GetBody0Rel().GetTargets() or root_joint.GetBody1Rel().GetTargets() != [
        Sdf.Path(spec.root_body_path)
    ]:
        raise RuntimeError(f"Root joint {spec.root_joint_name} is not a canonical world-fixed joint")
    for body_path, filtered_body_path in spec.collision_filter_pairs:
        targets = UsdPhysics.FilteredPairsAPI.Get(stage, body_path).GetFilteredPairsRel().GetTargets()
        if Sdf.Path(filtered_body_path) not in targets:
            raise RuntimeError(f"Missing self-collision filter: {body_path} -> {filtered_body_path}")

    urdf_root = ET.parse(spec.urdf_path).getroot()
    expected_hulls = sum(
        "collision_hulls" in (mesh.get("filename") or "")
        for collision in urdf_root.findall(".//collision")
        for mesh in collision.findall(".//mesh")
    )
    imported_hulls = [prim for prim in stage.Traverse() if "_hull_" in prim.GetName()]
    if len(imported_hulls) != expected_hulls:
        raise RuntimeError(f"Imported {len(imported_hulls)} mesh colliders, expected {expected_hulls}")
    if any("collisions_1" in prim.GetName() for prim in stage.Traverse()):
        raise RuntimeError("Found legacy collisions_1 prim naming in generated asset")
    if spec.key == "dex3":
        thumb_boxes = [
            prim
            for prim in stage.Traverse()
            if prim.GetTypeName() == "Cube" and "right_hand_thumb_1_link" in str(prim.GetPath())
        ]
        if len(thumb_boxes) != 1:
            raise RuntimeError("The authored Dex3 thumb box collider was not preserved")


def main() -> None:
    global Gf, Sdf, UrdfConverter, UrdfConverterCfg, Usd, UsdPhysics  # noqa: PLW0603
    from pxr import Gf, Sdf, Usd, UsdPhysics

    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    spec = HAND_SPECS[args.hand]
    output_dir = (args.output_dir or spec.asset_dir).expanduser().resolve()
    if output_dir.name != spec.asset_dir.name:
        raise ValueError(f"--output-dir must name the generated {spec.asset_dir.name!r} package directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"{spec.key}_export_", dir=output_dir.parent) as temp_dir:
        converter = UrdfConverter(
            UrdfConverterCfg(
                asset_path=str(spec.urdf_path),
                usd_dir=temp_dir,
                fix_base=True,
                merge_fixed_joints=False,
                force_usd_conversion=True,
                make_instanceable=True,
                physics_variant="physx",
                self_collision=True,
                robot_type="End Effector",
                run_asset_transformer=True,
                run_multi_physics_conversion=True,
                debug_mode=False,
                joint_drive=UrdfConverterCfg.JointDriveCfg(
                    drive_type="force",
                    target_type="position",
                    gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                        stiffness=spec.stiffness,
                        damping=spec.damping,
                    ),
                ),
            )
        )
        generated_path = Path(converter.usd_path)
        generated_dir = generated_path.parent
        _author_portable_defaults(spec, generated_dir)
        _author_physx_defaults(spec, generated_dir)
        _author_display_pose(spec, generated_dir)
        _author_robot_relationships(spec, generated_path)
        _validate_asset(spec, generated_path)

        configuration_dir = output_dir / "configuration"
        if configuration_dir.exists():
            shutil.rmtree(configuration_dir)
        shutil.copytree(generated_dir / "payloads", configuration_dir)
        _normalize_usda_files(configuration_dir)
        root_text = generated_path.read_text().replace("@./payloads/", "@./configuration/")
        (output_dir / spec.usd_name).write_text(root_text.rstrip() + "\n")

    _validate_asset(spec, output_dir / spec.usd_name)
    print(f"PASS exported {output_dir / spec.usd_name}")
    print(f"ROM contract: {spec.rom_contract}")
    print(f"PhysX/Newton armature default: {spec.armature:g}")


if __name__ == "__main__":
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args):
        main()
