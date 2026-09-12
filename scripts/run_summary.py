#!/usr/bin/env python3
"""Summarise the key scalars of one or two RSL-RL runs as a binned markdown table.

`analyze_run.py` answers "one tag in detail"; this answers "every tag I care about,
at a glance", which is what deciding the next experiment actually needs.

Usage:
    scripts/run_summary.py RUN [RUN_B] [--bins N] [--tags a,b,c]
"""
from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_TAGS = [
    "Train/mean_reward", "Train/mean_episode_length",
    "Task/success", "Task/object_height_ep_max",
    "Task/keypoint_error_ep_min", "Task/keypoint_error_ep_final",
    "Reach/hand_distance_farthest_ep_min", "Reach/hand_distance_nearest_step",
    "Reward/reach_ep_return", "Reward/goal_ep_return", "Reward/contact_ep_return",
    "Contact/gate_frac_ep",
    "Contact/touch_frac_any_step", "Contact/force_max_step", "Contact/force_thumb_step",
    "Contact/groups_over_threshold_step",
    "Policy/mean_std", "Loss/entropy", "Loss/value", "Loss/learning_rate",
    "Control/action_saturation_frac_step",
    "Terminations/timeout", "Terminations/workspace_exit", "Terminations/below_table",
    "Perf/total_fps",
]
SHORT = {"Task/": "Tk/", "Reach/": "R/", "Reward/": "W/", "Train/": "T/", "Control/": "C/",
         "Terminations/": "X/", "Policy/": "P/", "Loss/": "L/", "Perf/": "", "Contact/": "K/"}
# Runs logged before the tag rename, read under their new names so old and new runs compare.
LEGACY_TAGS = {
    "Metrics/success": "Task/success",
    "Metrics/final_keypoint_error": "Task/keypoint_error_ep_final",
    "Metrics/min_keypoint_error": "Task/keypoint_error_ep_min",
    "Metrics/step_keypoint_error": "Task/keypoint_error_step",
    "Metrics/max_object_height": "Task/object_height_ep_max",
    "Metrics/step_object_height": "Task/object_height_step",
    "Metrics/min_hand_distance": "Reach/hand_distance_farthest_ep_min",
    "Metrics/step_hand_distance": "Reach/hand_distance_farthest_step",
    "Metrics/step_nearest_hand_distance": "Reach/hand_distance_nearest_step",
    "Contact/step_max_force": "Contact/force_max_step",
    "Contact/step_thumb_force": "Contact/force_thumb_step",
    "Metrics/contact_gate_fraction": "Contact/gate_frac_ep",
    "Metrics/step_contact_gate_fraction": "Contact/gate_frac_step",
    "Contact/step_bodies_over_threshold": "Contact/groups_over_threshold_step",
    "Contact/step_any_touch_fraction": "Contact/touch_frac_any_step",
    **{f"Contact/step_touch_fraction_{group}": f"Contact/touch_frac_{group}_step"
       for group in ("palm", "finger1", "finger2", "finger3", "finger4", "finger5")},
    **{f"Metrics/{term}_return": f"Reward/{term}_ep_return" for term in ("reach", "goal", "contact", "lift")},
    **{f"Metrics/step_{term}_reward": f"Reward/{term}_step" for term in ("reach", "goal", "contact", "lift")},
    "Control/step_action_saturation_fraction": "Control/action_saturation_frac_step",
    "Control/arm_tracking_error": "Control/arm_tracking_error_ep",
    "Control/step_arm_tracking_error": "Control/arm_tracking_error_step",
    "Control/wuji_tracking_error": "Control/wuji_tracking_error_ep",
    "Control/step_wuji_tracking_error": "Control/wuji_tracking_error_step",
}


def load(run: Path, tags: set[str]) -> dict[str, dict[int, float]]:
    """Read the cached scalars, preparing them via analyze_run.py when stale."""
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("analyze_run.py")), "prepare", str(run)],
        check=True, capture_output=True,
    )
    out: dict[str, dict[int, float]] = defaultdict(dict)
    with (run / "analysis/scalars.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            tag = LEGACY_TAGS.get(row["tag"], row["tag"])
            if tag in tags:
                value = float(row["value"])
                if math.isfinite(value):
                    out[tag][int(row["step"])] = value
    return out


def bins(hi: int, count: int) -> list[tuple[int, int]]:
    width = max(1, math.ceil((hi + 1) / count))
    return [(left, min(hi, left + width - 1)) for left in range(0, hi + 1, width)]


def mean(values: dict[int, float], lo: int, hi: int) -> str:
    chosen = [v for s, v in values.items() if lo <= s <= hi]
    return format(sum(chosen) / len(chosen), ".4g") if chosen else ""


def short(tag: str) -> str:
    for long_prefix, abbreviation in SHORT.items():
        if tag.startswith(long_prefix):
            return abbreviation + tag[len(long_prefix):]
    return tag


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--tags", default="")
    args = parser.parse_args()

    tags = args.tags.split(",") if args.tags else DEFAULT_TAGS
    loaded = [load(run, set(tags)) for run in args.runs]
    for run, data in zip(args.runs, loaded, strict=True):
        steps = [s for values in data.values() for s in values]
        print(f"{run.name}: {max(steps) + 1 if steps else 0} iterations")

    hi = max((s for data in loaded for values in data.values() for s in values), default=0)
    edges = bins(hi, args.bins)
    header = ["tag"] + [f"{lo}-{high}" for lo, high in edges]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for tag in tags:
        for index, data in enumerate(loaded):
            if tag not in data:
                continue
            label = short(tag) + (f" [{index}]" if len(loaded) > 1 else "")
            cells = [mean(data[tag], lo, high) for lo, high in edges]
            print("| " + " | ".join([label] + cells) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
