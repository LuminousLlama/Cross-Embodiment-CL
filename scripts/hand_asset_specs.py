"""Source-of-truth metadata for standalone hand asset generation and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class HandAssetSpec:
    key: str
    asset_dir: Path
    urdf_path: Path
    usd_name: str
    root_prim: str
    root_body_path: str
    root_body_name: str
    root_joint_name: str
    rom_contract_key: str
    rom_contract: str
    joint_names: tuple[str, ...]
    active_joint_names: tuple[str, ...]
    active_joint_pattern: str
    follower_joint_pattern: str | None
    joint_limits: dict[str, tuple[float, float]]
    open_joint_pos: dict[str, float]
    stiffness: dict[str, float]
    damping: dict[str, float]
    armature: float
    mimic_joints: dict[str, tuple[str, float]]
    collision_filter_pairs: tuple[tuple[str, str], ...]
    mesh_collision_links: tuple[str, ...]
    palm_link_name: str
    mesh_suffix: str

    @property
    def usd_path(self) -> Path:
        return self.asset_dir / self.usd_name


_INSPIRE_BODY_ROOT = "/inspire_hand_right/Geometry/inspire_hand_base/hand_base_link"
_INSPIRE_ACTIVE = (
    "right_index_1_joint",
    "right_middle_1_joint",
    "right_ring_1_joint",
    "right_little_1_joint",
    "right_thumb_1_joint",
    "right_thumb_2_joint",
)
_INSPIRE_FOLLOWERS = (
    "right_index_2_joint",
    "right_middle_2_joint",
    "right_ring_2_joint",
    "right_little_2_joint",
    "right_thumb_3_joint",
    "right_thumb_4_joint",
)
_INSPIRE_LINKS = (
    "hand_base_link",
    "right_index_1",
    "right_index_2",
    "right_middle_1",
    "right_middle_2",
    "right_ring_1",
    "right_ring_2",
    "right_little_1",
    "right_little_2",
    "right_thumb_1",
    "right_thumb_2",
    "right_thumb_3",
    "right_thumb_4",
)

INSPIRE = HandAssetSpec(
    key="inspire",
    asset_dir=REPO_ROOT / "assets/hands/inspire_hand",
    urdf_path=REPO_ROOT / "assets/hands/inspire_hand/urdf/inspire_hand_right.urdf",
    usd_name="inspire_hand_right.usda",
    root_prim="/inspire_hand_right",
    root_body_path=_INSPIRE_BODY_ROOT,
    root_body_name="hand_base_link",
    root_joint_name="base_joint",
    rom_contract_key="inspireHandRomContract",
    rom_contract="inspire_rh56_safe_rom_v1",
    joint_names=_INSPIRE_ACTIVE + _INSPIRE_FOLLOWERS,
    active_joint_names=_INSPIRE_ACTIVE,
    active_joint_pattern=r"right_(index|middle|ring|little)_1_joint|right_thumb_[12]_joint",
    follower_joint_pattern=r"right_(index|middle|ring|little)_2_joint|right_thumb_[34]_joint",
    joint_limits={
        "right_index_1_joint": (0.0, 1.47),
        "right_index_2_joint": (0.0, 1.5435),
        "right_middle_1_joint": (0.0, 1.47),
        "right_middle_2_joint": (0.0, 1.5435),
        "right_ring_1_joint": (0.0, 1.47),
        "right_ring_2_joint": (0.0, 1.5435),
        "right_little_1_joint": (0.0, 1.47),
        "right_little_2_joint": (0.0, 1.5435),
        "right_thumb_1_joint": (-1.20, -0.30),
        "right_thumb_2_joint": (0.14, 0.56),
        "right_thumb_3_joint": (0.056, 0.224),
        "right_thumb_4_joint": (0.084, 0.336),
    },
    open_joint_pos={
        "right_index_1_joint": 0.0,
        "right_index_2_joint": 0.0,
        "right_middle_1_joint": 0.0,
        "right_middle_2_joint": 0.0,
        "right_ring_1_joint": 0.0,
        "right_ring_2_joint": 0.0,
        "right_little_1_joint": 0.0,
        "right_little_2_joint": 0.0,
        "right_thumb_1_joint": -0.30,
        "right_thumb_2_joint": 0.14,
        "right_thumb_3_joint": 0.056001,
        "right_thumb_4_joint": 0.084001,
    },
    stiffness={name: 10.0 if name in _INSPIRE_ACTIVE else 0.0 for name in _INSPIRE_ACTIVE + _INSPIRE_FOLLOWERS},
    damping={name: 1.0 if name in _INSPIRE_ACTIVE else 0.0 for name in _INSPIRE_ACTIVE + _INSPIRE_FOLLOWERS},
    armature=0.0002,
    mimic_joints={
        "right_index_2_joint": ("right_index_1_joint", 1.05),
        "right_middle_2_joint": ("right_middle_1_joint", 1.05),
        "right_ring_2_joint": ("right_ring_1_joint", 1.05),
        "right_little_2_joint": ("right_little_1_joint", 1.05),
        "right_thumb_3_joint": ("right_thumb_2_joint", 0.4),
        "right_thumb_4_joint": ("right_thumb_2_joint", 0.6),
    },
    collision_filter_pairs=(
        (_INSPIRE_BODY_ROOT, f"{_INSPIRE_BODY_ROOT}/right_thumb_1/right_thumb_2"),
        (f"{_INSPIRE_BODY_ROOT}/right_ring_1/right_ring_2", f"{_INSPIRE_BODY_ROOT}/right_little_1"),
    ),
    mesh_collision_links=_INSPIRE_LINKS,
    palm_link_name="hand_base_link",
    mesh_suffix="_visuals.obj",
)

_DEX3_JOINTS = (
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
)
_DEX3_STIFFNESS = (461.5, 384.67, 124.19, 1154.98, 124.22, 1152.2, 124.22)
_DEX3_DAMPING = (5.13, 4.26, 1.38, 12.8, 1.38, 12.8, 1.38)

DEX3 = HandAssetSpec(
    key="dex3",
    asset_dir=REPO_ROOT / "assets/hands/dex3_1_hand",
    urdf_path=REPO_ROOT / "assets/hands/dex3_1_hand/urdf/dex3_1_r.urdf",
    usd_name="dex3_1_right.usda",
    root_prim="/dex3_1_r",
    root_body_path="/dex3_1_r/Geometry/right_hand_wrist_link/right_hand_palm_link",
    root_body_name="right_hand_palm_link",
    root_joint_name="right_hand_wrist_to_palm_joint",
    rom_contract_key="dex3RomContract",
    rom_contract="unitree_dex3_1_urdf_rom_v1",
    joint_names=_DEX3_JOINTS,
    active_joint_names=_DEX3_JOINTS,
    active_joint_pattern=r"right_hand_(thumb|index|middle)_[0-2]_joint",
    follower_joint_pattern=None,
    joint_limits={
        "right_hand_thumb_0_joint": (-1.04719755, 1.04719755),
        "right_hand_thumb_1_joint": (-1.04719755, 0.61086523),
        "right_hand_thumb_2_joint": (-1.74532925, 0.0),
        "right_hand_index_0_joint": (0.0, 1.57079632),
        "right_hand_index_1_joint": (0.0, 1.74532925),
        "right_hand_middle_0_joint": (0.0, 1.57079632),
        "right_hand_middle_1_joint": (0.0, 1.74532925),
    },
    open_joint_pos={
        "right_hand_thumb_0_joint": 0.0,
        "right_hand_thumb_1_joint": 0.0,
        "right_hand_thumb_2_joint": -0.05,
        "right_hand_index_0_joint": 0.05,
        "right_hand_index_1_joint": 0.05,
        "right_hand_middle_0_joint": 0.05,
        "right_hand_middle_1_joint": 0.05,
    },
    stiffness=dict(zip(_DEX3_JOINTS, _DEX3_STIFFNESS, strict=True)),
    damping=dict(zip(_DEX3_JOINTS, _DEX3_DAMPING, strict=True)),
    armature=0.001,
    mimic_joints={},
    collision_filter_pairs=(),
    mesh_collision_links=(
        "right_hand_palm_link",
        "right_hand_thumb_0_link",
        "right_hand_thumb_2_link",
        "right_hand_middle_0_link",
        "right_hand_middle_1_link",
        "right_hand_index_0_link",
        "right_hand_index_1_link",
    ),
    palm_link_name="right_hand_palm_link",
    mesh_suffix=".STL",
)

HAND_SPECS = {spec.key: spec for spec in (INSPIRE, DEX3)}
