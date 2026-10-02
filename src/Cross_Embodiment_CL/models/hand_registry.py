# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hand semantics shared by retargeting and the fixed-size task observation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .hand_observation import HAND_SLOT_COUNT

FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")


@dataclass(frozen=True)
class HandSpec:
    """Independent joints and semantic points; physical properties belong to USD."""

    name: str
    joint_names: tuple[str, ...]
    model_joint_names: tuple[str, ...]
    slot_indices: tuple[int, ...]
    hand_prim_path: str
    contact_body_groups: dict[str, tuple[str, ...]]
    palm_body_name: str
    tip_body_names: tuple[str, ...]
    tip_frame_paths: tuple[str, ...]
    palm_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    zero_floor_names: tuple[str, ...] = ()
    follower_joint_names: tuple[str, ...] = ()
    supports_force: bool = False
    retargeter_directory: str = ""

    def __post_init__(self) -> None:
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("Independent joint names must be nonempty and unique.")
        if len(set(self.model_joint_names)) != len(self.model_joint_names):
            raise ValueError("Retargeter joint names must be unique.")
        if len(self.model_joint_names) != len(self.joint_names) or len(self.slot_indices) != len(self.joint_names):
            raise ValueError("Every independent joint needs one model name and one semantic slot.")
        if len(set(self.slot_indices)) != len(self.slot_indices) or any(
            type(slot) is not int or not 0 <= slot < HAND_SLOT_COUNT for slot in self.slot_indices
        ):
            raise ValueError("Semantic hand slots must be unique integers in [0, 20).")
        if len(self.tip_body_names) != len(self.tip_frame_paths):
            raise ValueError("Every fingertip body needs a local point offset.")
        groups = set(self.contact_body_groups)
        if groups - {"palm", *FINGER_NAMES}:
            raise ValueError("Contact groups must use canonical palm and finger names.")
        fingers = groups - {"palm"}
        if not {"palm", "thumb"} <= groups or not fingers - {"thumb"}:
            raise ValueError("Hand contacts require a palm, thumb, and at least one other finger.")
        if any(not bodies for bodies in self.contact_body_groups.values()):
            raise ValueError("Every contact group must contain at least one body.")
        bodies = tuple(body for group in self.contact_body_groups.values() for body in group)
        if len(set(bodies)) != len(bodies):
            raise ValueError("Contact bodies must be unique across all hand groups.")
        if len(self.tip_body_names) != len(fingers):
            raise ValueError("Fingertip points must match the active contact finger groups.")
        if any(FINGER_NAMES[slot // 4] not in fingers for slot in self.slot_indices):
            raise ValueError("Every semantic joint slot must belong to an active contact finger group.")
        if set(self.zero_floor_names) - set(self.joint_names):
            raise ValueError("Command lower-bound overrides must refer to independent joints.")
        if len(set(self.follower_joint_names)) != len(self.follower_joint_names) or set(
            self.follower_joint_names
        ) & set(self.joint_names):
            raise ValueError("Follower joints must be unique and separate from independent joints.")

    def read_tip_offsets(self, usd_path: str | Path) -> tuple[tuple[float, float, float], ...]:
        """Read fingertip offsets [m] from the selected assembly, including asset overrides."""
        import newton_usd_schemas  # noqa: F401
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.Open(str(usd_path))
        if stage is None:
            raise ValueError(f"Cannot open fingertip assembly USD: {usd_path}.")
        cache = UsdGeom.XformCache()
        offsets = []
        for body_name, frame_path in zip(self.tip_body_names, self.tip_frame_paths, strict=True):
            frame = stage.GetPrimAtPath(stage.GetDefaultPrim().GetPath().AppendPath(frame_path))
            if not frame.IsValid():
                raise ValueError(f"Missing fingertip frame {frame_path} in {usd_path}.")
            body = frame
            while body.IsValid() and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                body = body.GetParent()
            if not body.IsValid() or body.GetName() != body_name:
                raise ValueError(f"Fingertip {frame_path} must belong to rigid body {body_name}.")
            transform = cache.GetLocalToWorldTransform(frame) * cache.GetLocalToWorldTransform(body).GetInverse()
            offsets.append(tuple(float(value) for value in transform.ExtractTranslation()))
        return tuple(offsets)

    def model_order(self, names: tuple[str, ...]) -> tuple[int, ...]:
        """Map manifest output order to independent command order, rejecting ambiguous mappings."""
        if len(set(names)) != len(names) or set(names) != set(self.model_joint_names):
            raise ValueError(
                f"{self.name} retargeter independent joint names do not match the hand definition: {names}."
            )
        return tuple(names.index(name) for name in self.model_joint_names)


_WUJI_NAMES = tuple(f"right_finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5))
_INSPIRE_ROOT = "inspire_hand_right/Geometry/inspire_hand_base/hand_base_link"
_DEX3_ROOT = "dex3_1_right/Geometry/right_hand_wrist_link/right_hand_palm_link"

HAND_SPECS = {
    "wuji": HandSpec(
        name="wuji",
        joint_names=_WUJI_NAMES,
        model_joint_names=tuple(name.removeprefix("right_") for name in _WUJI_NAMES),
        slot_indices=tuple(range(20)),
        hand_prim_path="wujihand",
        contact_body_groups={
            "palm": ("wujihand/right_palm_link",),
            "thumb": tuple(
                f"wujihand/right_finger1_{link}" for link in ("link1", "link2", "link2_softbody", "link3", "link4")
            ),
            **{
                finger_name: tuple(f"wujihand/right_finger{finger}_link{link}" for link in (2, 3, 4))
                for finger, finger_name in enumerate(FINGER_NAMES[1:], 2)
            },
        },
        palm_body_name="right_palm_link",
        tip_body_names=tuple(f"right_finger{finger}_tip_link" for finger in range(1, 6)),
        tip_frame_paths=tuple(f"wujihand/right_finger{finger}_tip_link" for finger in range(1, 6)),
        zero_floor_names=tuple(
            name for name in _WUJI_NAMES if name not in tuple(f"right_finger{f}_joint2" for f in range(2, 6))
        ),
        supports_force=True,
        retargeter_directory="wuji",
    ),
    "inspire": HandSpec(
        name="inspire",
        joint_names=(
            "right_index_1_joint",
            "right_middle_1_joint",
            "right_ring_1_joint",
            "right_little_1_joint",
            "right_thumb_1_joint",
            "right_thumb_2_joint",
        ),
        model_joint_names=(
            "right_index_1_joint",
            "right_middle_1_joint",
            "right_ring_1_joint",
            "right_little_1_joint",
            "right_thumb_1_joint",
            "right_thumb_2_joint",
        ),
        slot_indices=(4, 8, 12, 16, 0, 1),
        hand_prim_path="inspire_hand_right",
        contact_body_groups={
            "palm": (_INSPIRE_ROOT,),
            "thumb": tuple(
                f"{_INSPIRE_ROOT}/" + "/".join(f"right_thumb_{i}" for i in range(1, j + 1)) for j in range(1, 5)
            ),
            **{
                name: (f"{_INSPIRE_ROOT}/right_{name}_1", f"{_INSPIRE_ROOT}/right_{name}_1/right_{name}_2")
                for name in FINGER_NAMES[1:]
            },
        },
        palm_body_name="hand_base_link",
        tip_body_names=("right_thumb_4", "right_index_2", "right_middle_2", "right_ring_2", "right_little_2"),
        tip_frame_paths=(f"{_INSPIRE_ROOT}/right_thumb_1/right_thumb_2/right_thumb_3/right_thumb_4/thumb_tip",)
        + tuple(f"{_INSPIRE_ROOT}/right_{name}_1/right_{name}_2/{name}_tip" for name in FINGER_NAMES[1:]),
        follower_joint_names=(
            "right_index_2_joint",
            "right_middle_2_joint",
            "right_ring_2_joint",
            "right_little_2_joint",
            "right_thumb_3_joint",
            "right_thumb_4_joint",
        ),
        retargeter_directory="inspire",
    ),
    "dex3": HandSpec(
        name="dex3",
        joint_names=(
            "right_hand_thumb_0_joint",
            "right_hand_thumb_1_joint",
            "right_hand_thumb_2_joint",
            "right_hand_index_0_joint",
            "right_hand_index_1_joint",
            "right_hand_middle_0_joint",
            "right_hand_middle_1_joint",
        ),
        model_joint_names=(
            "right_hand_thumb_0_joint",
            "right_hand_thumb_1_joint",
            "right_hand_thumb_2_joint",
            "right_hand_index_0_joint",
            "right_hand_index_1_joint",
            "right_hand_middle_0_joint",
            "right_hand_middle_1_joint",
        ),
        slot_indices=(0, 1, 2, 4, 5, 8, 9),
        hand_prim_path="dex3_1_right",
        contact_body_groups={
            "palm": (_DEX3_ROOT,),
            "thumb": tuple(
                f"{_DEX3_ROOT}/" + "/".join(f"right_hand_thumb_{i}_link" for i in range(j + 1)) for j in (0, 2)
            ),
            **{
                name: (
                    f"{_DEX3_ROOT}/right_hand_{name}_0_link",
                    f"{_DEX3_ROOT}/right_hand_{name}_0_link/right_hand_{name}_1_link",
                )
                for name in ("index", "middle")
            },
        },
        palm_body_name="right_hand_palm_link",
        tip_body_names=("right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link"),
        tip_frame_paths=(
            f"{_DEX3_ROOT}/right_hand_thumb_0_link/right_hand_thumb_1_link/right_hand_thumb_2_link/right_hand_thumb_tip_head",
        )
        + tuple(
            f"{_DEX3_ROOT}/right_hand_{name}_0_link/right_hand_{name}_1_link/right_hand_{name}_tip_head"
            for name in ("index", "middle")
        ),
        retargeter_directory="dex3_1",
    ),
}


def get_hand_spec(name: str) -> HandSpec:
    """Return the selected hand's semantic contract."""
    try:
        return HAND_SPECS[name]
    except KeyError:
        raise ValueError(f"Unknown hand {name!r}; choose one of {tuple(HAND_SPECS)}.") from None
