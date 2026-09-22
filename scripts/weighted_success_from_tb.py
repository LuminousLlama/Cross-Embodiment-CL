#!/usr/bin/env python3
"""Print the per-episode weighted success rate from a run's Task/success_count_step and
Task/episodes_done_step scalars.

rsl_rl's Logger averages a tag unweighted over the 32-step rollout window, so
`Task/success` (mean of the per-step success mean) is a mean-of-ratios, not the weighted
per-episode success rate eval_policy.py reports. Since both tags are logged on every step
(0 on steps with no resets), dividing their per-iteration means recovers the weighted rate:
mean(success_count_step) / mean(episodes_done_step) = sum(successes) / sum(episodes done).

Usage:
    scripts/weighted_success_from_tb.py RUN_DIR [--window N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

COUNT_TAG = "Task/success_count_step"
DONE_TAG = "Task/episodes_done_step"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run directory containing the tfevents file.")
    parser.add_argument("--window", type=int, default=50, help="Rolling-mean window, in iterations.")
    args = parser.parse_args()

    accumulator = EventAccumulator(str(args.run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    counts = {event.step: event.value for event in accumulator.Scalars(COUNT_TAG)}
    dones = {event.step: event.value for event in accumulator.Scalars(DONE_TAG)}

    steps = sorted(set(counts) & set(dones))
    if not steps:
        print(f"No overlapping {COUNT_TAG!r} / {DONE_TAG!r} scalars found under {args.run_dir}")
        return

    ratios: list[float] = []
    print(f"{'step':>8}  {'success':>10}  {'done':>10}  {'ratio':>8}  {'rolling':>8}")
    for step in steps:
        count, done = counts[step], dones[step]
        ratio = count / done if done > 0 else float("nan")
        ratios.append(ratio)
        window = [r for r in ratios[-args.window :] if r == r]  # drop NaNs
        rolling = sum(window) / len(window) if window else float("nan")
        print(f"{step:>8}  {count:>10.1f}  {done:>10.1f}  {ratio:>8.3f}  {rolling:>8.3f}")


if __name__ == "__main__":
    main()
