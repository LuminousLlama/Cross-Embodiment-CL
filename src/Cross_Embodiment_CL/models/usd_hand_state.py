# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read authored hand reset positions instead of inferring them from drives."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pxr import Usd

from .hand_registry import HandSpec


def _mimic_relation(prim: Usd.Prim, schemas: tuple[str, ...]) -> tuple[str, float, float] | None:
    relations = []

    def attribute(name: str, default: float) -> float:
        value = prim.GetAttribute(name).Get()
        return default if value is None else float(value)

    def relation(name: str, coefficient: float, offset: float) -> None:
        targets = prim.GetRelationship(name).GetTargets()
        if len(targets) != 1 or not all(math.isfinite(value) for value in (coefficient, offset)):
            raise ValueError(f"Invalid linear mimic relation on {prim.GetPath()}.")
        target = targets[0]
        if not target.IsAbsolutePath():
            target = target.MakeAbsolutePath(prim.GetPath().GetParentPath())
        relations.append((str(target), coefficient, offset))

    if "NewtonMimicAPI" in schemas:
        if any(attribute(f"newton:mimicCoef{i}", 0.0) != 0.0 for i in range(2, 5)):
            raise ValueError(f"Nonlinear hand mimic is unsupported on {prim.GetPath()}.")
        relation(
            "newton:mimicJoint", attribute("newton:mimicCoef1", 1.0), math.radians(attribute("newton:mimicCoef0", 0.0))
        )
    for schema in schemas:
        if schema.startswith("PhysxMimicJointAPI:"):
            prefix = f"physxMimicJoint:{schema.split(':', 1)[1]}"
            # PhysX defines follower + gearing * leader + offset = 0;
            # its angular offset is radians (Newton authors offsets in degrees).
            relation(
                f"{prefix}:referenceJoint", -attribute(f"{prefix}:gearing", 1.0), -attribute(f"{prefix}:offset", 0.0)
            )
    if not relations:
        return None
    leader, coefficient, offset = relations[0]
    if any(
        target != leader
        or not math.isclose(gain, coefficient, abs_tol=1.0e-6)
        or not math.isclose(bias, offset, abs_tol=1.0e-6)
        for target, gain, bias in relations[1:]
    ):
        raise ValueError(f"Backend mimic relations disagree on {prim.GetPath()}.")
    return leader, coefficient, offset


def read_hand_joint_positions(usd_path: str | Path, hand_spec: HandSpec) -> dict[str, float]:
    """Return all independent and follower authored joint-state positions [rad]."""
    # Register before the first stage builds USD's schema registry, but only
    # at runtime: importing USD before AppLauncher breaks Kit's USD ABI.
    import newton_usd_schemas  # noqa: F401
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise ValueError(f"Cannot open hand assembly USD: {usd_path}.")
    root = stage.GetDefaultPrim().GetPath().AppendPath(hand_spec.hand_prim_path)
    required = set(hand_spec.joint_names + hand_spec.follower_joint_names)
    positions: dict[str, float] = {}
    independent: set[str] = set()
    joint_paths: dict[str, str] = {}
    limits: dict[str, tuple[float, float]] = {}
    mimics: dict[str, tuple[str, float, float]] = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            if prim.IsA(UsdPhysics.Joint) and not prim.IsA(UsdPhysics.FixedJoint):
                raise ValueError(f"Unsupported movable hand joint {prim.GetPath()} ({prim.GetTypeName()}).")
            continue
        name = prim.GetName()
        if name not in required or name in positions:
            raise ValueError(f"Unexpected or duplicate hand joint {prim.GetPath()} in {usd_path}.")
        # Unknown backend schemas are filtered out by GetAppliedSchemas in a
        # plain USD process; authored API metadata still carries the contract.
        schemas = prim.GetMetadata("apiSchemas")
        authored_schemas = schemas.GetAppliedItems() if schemas is not None else ()
        mimic = _mimic_relation(prim, authored_schemas)
        if mimic is None:
            independent.add(name)
        else:
            mimics[name] = mimic
            for field in ("stiffness", "damping"):
                value = prim.GetAttribute(f"drive:angular:physics:{field}").Get()
                if value is not None and (not math.isfinite(float(value)) or value != 0.0):
                    raise ValueError(f"Mimic follower {prim.GetPath()} must have passive USD drive gains.")
        position = prim.GetAttribute("state:angular:physics:position").Get()
        if position is None or not math.isfinite(float(position)):
            raise ValueError(f"Hand joint {prim.GetPath()} needs a finite authored angular state position.")
        joint = UsdPhysics.RevoluteJoint(prim)
        lower, upper = joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()
        if (
            lower is None
            or upper is None
            or not all(math.isfinite(float(value)) for value in (lower, upper))
            or lower >= upper
        ):
            raise ValueError(f"Hand joint {prim.GetPath()} needs finite ordered limits.")
        if not lower - 1.0e-4 <= position <= upper + 1.0e-4:
            raise ValueError(f"Authored hand state {position} deg lies outside limits on {prim.GetPath()}.")
        positions[name] = math.radians(float(position))
        limits[name] = (math.radians(lower), math.radians(upper))
        joint_paths[str(prim.GetPath())] = name
    if set(positions) != required:
        raise ValueError(f"Hand USD lacks joint state for {sorted(required - set(positions))}.")
    if independent != set(hand_spec.joint_names):
        raise ValueError(f"USD independent hand joints disagree with {hand_spec.name} mapping: {sorted(independent)}.")

    def mapped_interval(name: str, visiting: tuple[str, ...] = ()) -> tuple[float, float]:
        if name in visiting:
            raise ValueError(f"Cyclic hand mimic relation at {name}.")
        if name not in mimics:
            return limits[name]
        path, coefficient, offset = mimics[name]
        if path not in joint_paths:
            raise ValueError(f"Hand mimic {name} references missing or external leader {path}.")
        leader = joint_paths[path]
        if abs(positions[name] - (coefficient * positions[leader] + offset)) > 1.0e-5:
            raise ValueError(f"Authored hand reset state violates mimic relation at {name}.")
        interval = tuple(sorted(coefficient * value + offset for value in mapped_interval(leader, visiting + (name,))))
        if interval[0] < limits[name][0] - 1.0e-5 or interval[1] > limits[name][1] + 1.0e-5:
            raise ValueError(f"Driver ROM maps outside follower limits at {name}.")
        return interval

    for name in mimics:
        mapped_interval(name)
    return positions
