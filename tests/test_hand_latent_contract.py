# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU numerical checks of retargeter order, units, and independent-joint mappings."""

from pathlib import Path

import pytest
import torch
import yaml
from pxr import Usd

from Cross_Embodiment_CL.models.hand_registry import get_hand_spec
from Cross_Embodiment_CL.models.usd_hand_state import read_hand_joint_positions
from Cross_Embodiment_CL.models.wuji_latent import HandLatentActionPipeline


@pytest.mark.parametrize("hand_type", ["wuji", "inspire", "dex3"])
def test_latent_targets_match_checkpoint_matmul_in_radians_and_live_clamp(hand_type):
    """Catch manifest reordering, unintended unit scaling, and omitted live clamps."""
    pipeline = HandLatentActionPipeline(hand_type, "cpu")
    artifact_dir = Path(__file__).resolve().parents[1] / "assets/models/retargeting"
    artifact_dir /= pipeline.hand_spec.retargeter_directory
    manifest = yaml.safe_load(next(artifact_dir.glob("*.yaml")).read_text())
    weights = torch.load(artifact_dir / manifest["state_dict_path"], weights_only=True)
    latent = torch.linspace(-2, 2, 36).reshape(2, 18)
    reference = pipeline.decode_latent_action(latent)
    layers = sorted(int(key.split(".")[1]) for key in weights if key.endswith(".weight"))
    for layer in layers:
        reference = reference @ weights[f"model.{layer}.weight"].T + weights[f"model.{layer}.bias"]
        if layer != layers[-1]:
            reference = reference.clamp_min(0)
    model_names = manifest["robot_joint_names"]
    reference = reference[:, [model_names.index(name) for name in pipeline.hand_spec.model_joint_names]]
    # Force both lower and upper saturation with per-joint asymmetric intervals.
    lower = reference + torch.tensor([0.1, -0.2])[:, None]
    upper = lower + 0.05
    expected = torch.minimum(torch.maximum(reference, lower), upper)
    torch.testing.assert_close(pipeline.latent_action_to_joint_target(latent, lower, upper), expected)
    with pytest.raises(ValueError, match="upper joint limit"):
        pipeline.latent_action_to_joint_target(latent, upper, lower)


@pytest.mark.parametrize("hand_type", ["wuji", "inspire", "dex3"])
def test_manifest_permutation_maps_named_joints_and_rejects_ambiguous_orders(hand_type):
    """A shuffled model output still commands the same joint and semantic slot."""
    spec = get_hand_spec(hand_type)
    reversed_names = tuple(reversed(spec.model_joint_names))
    model_values = torch.tensor([spec.model_joint_names.index(name) + 0.25 for name in reversed_names])
    command_values = model_values[list(spec.model_order(reversed_names))]
    padded = torch.zeros(20)
    padded[list(spec.slot_indices)] = command_values
    for joint_index, slot in enumerate(spec.slot_indices):
        assert padded[slot].item() == joint_index + 0.25
    absent_slots = [slot for slot in range(20) if slot not in spec.slot_indices]
    assert torch.count_nonzero(padded[absent_slots]) == 0
    for invalid in (reversed_names[:-1], reversed_names[:-1] + (reversed_names[0],)):
        with pytest.raises(ValueError, match="do not match"):
            spec.model_order(invalid)


def test_inspire_reset_rejects_in_range_follower_pose_that_violates_authored_mimic(tmp_path):
    """An independently legal follower reset must not teleport when mimic constraints start."""
    spec = get_hand_spec("inspire")
    source = Path(__file__).resolve().parents[1] / "assets/g1/g1_with_hands/g1_inspire.usda"
    layer_path = tmp_path / "inspire_reset.usda"
    stage = Usd.Stage.CreateNew(str(layer_path))
    stage.GetRootLayer().subLayerPaths = [str(source)]
    stage.SetDefaultPrim(stage.GetPrimAtPath(Usd.Stage.Open(str(source)).GetDefaultPrim().GetPath()))
    stage.GetRootLayer().Save()
    follower = next(prim for prim in stage.Traverse() if prim.GetName() in spec.follower_joint_names)
    leader = stage.GetPrimAtPath(follower.GetRelationship("newton:mimicJoint").GetTargets()[0])
    coefficient = follower.GetAttribute("newton:mimicCoef1").Get()
    expected_degrees = coefficient * leader.GetAttribute("state:angular:physics:position").Get()
    positions = read_hand_joint_positions(layer_path, spec)
    assert positions[follower.GetName()] == pytest.approx(torch.deg2rad(torch.tensor(expected_degrees)).item())
    upper = follower.GetAttribute("physics:upperLimit").Get()
    invalid_degrees = expected_degrees + 0.1 * (upper - expected_degrees)
    assert invalid_degrees < upper
    follower.GetAttribute("state:angular:physics:position").Set(invalid_degrees)
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="reset state violates mimic relation"):
        read_hand_joint_positions(layer_path, spec)
