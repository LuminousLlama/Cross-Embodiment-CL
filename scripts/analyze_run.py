#!/usr/bin/env python3
"""Prepare and query TensorBoard scalars without accumulator deduplication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

VERSION = 2
FIELDS = ["tag", "step", "wall_time", "value"]


def event_files(run):
    return sorted((p for p in run.glob("events.out.tfevents.*") if p.is_file()), key=lambda p: p.name)


def fingerprint(run, files):
    return [
        {"path": p.relative_to(run).as_posix(), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        for p in files
    ]


def scalar_value(summary):
    if summary.HasField("simple_value"):
        return float(summary.simple_value)
    if summary.HasField("tensor"):
        try:
            from tensorboard.util import tensor_util

            value = tensor_util.make_ndarray(summary.tensor)
            if getattr(value, "ndim", None) == 0:
                return float(value)
        except (ImportError, TypeError, ValueError):
            pass
    return None


def read_records(files):
    records = []
    for path in files:
        for event in EventFileLoader(str(path)).Load():
            for summary in event.summary.value:
                value = scalar_value(summary)
                if value is not None:
                    records.append(
                        {
                            "tag": summary.tag,
                            "step": int(event.step),
                            "wall_time": float(event.wall_time),
                            "value": value,
                        }
                    )
    return records


def prepare(run, quiet=False):
    files = event_files(run)
    if not files:
        raise ValueError(f"no TensorBoard event files found directly under {run}")
    before = fingerprint(run, files)
    records = read_records(files)
    after = fingerprint(run, event_files(run))
    if before != after:
        raise RuntimeError("event files changed while parsing; run prepare again")
    analysis = run / "analysis"
    analysis.mkdir(exist_ok=True)
    with (analysis / "scalars.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    tags = {}
    for record in records:
        item = tags.setdefault(record["tag"], {"records": 0, "min_step": record["step"], "max_step": record["step"]})
        item["records"] += 1
        item["min_step"] = min(item["min_step"], record["step"])
        item["max_step"] = max(item["max_step"], record["step"])
    digest = hashlib.sha256(json.dumps(after, sort_keys=True).encode()).hexdigest()
    with (analysis / "index.json").open("w") as stream:
        json.dump(
            {"parser_version": VERSION, "source": after, "source_fingerprint": digest, "tags": tags},
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
    if not quiet:
        print(f"prepared {run}: {len(records)} scalar records, {len(tags)} tags")


def ensure_prepared(run):
    files = event_files(run)
    current = fingerprint(run, files)
    try:
        index = json.loads((run / "analysis/index.json").read_text())
        stale = (
            index.get("parser_version") != VERSION
            or index.get("source") != current
            or not (run / "analysis/scalars.csv").is_file()
        )
    except (OSError, ValueError, TypeError):
        stale = True
    if stale:
        prepare(run, quiet=True)


def load_csv(run, tag):
    records = []
    with (run / "analysis/scalars.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["tag"] == tag:
                records.append(
                    {"step": int(row["step"]), "wall_time": float(row["wall_time"]), "value": float(row["value"])}
                )
    return records


def parse_range(text):
    if not text:
        return None, None
    try:
        start, end = (int(part) for part in text.split(":", 1))
    except ValueError as exc:
        raise ValueError("--steps must be START:END") from exc
    if start > end:
        raise ValueError("--steps START must be <= END")
    return start, end


def select(records, start, end):
    return [r for r in records if (start is None or r["step"] >= start) and (end is None or r["step"] <= end)]


def dedup(records):
    latest, duplicates = {}, 0
    for record in records:
        if record["step"] in latest:
            duplicates += 1
        latest[record["step"]] = record
    nonfinite = sum(not math.isfinite(r["value"]) for r in latest.values())
    finite = [r for r in sorted(latest.values(), key=lambda r: r["step"]) if math.isfinite(r["value"])]
    return finite, duplicates, nonfinite


def bounds(start, end, requested):
    count = min(requested or 20, end - start + 1)
    width = math.ceil((end - start + 1) / count)
    return [(left, min(end, left + width - 1)) for left in range(start, end + 1, width)]


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows)
    return "\n".join(lines)


def num(value):
    return format(value, ".6g")


def curve(run, tag, step_text, bins, exact, output):
    if exact and bins is not None:
        raise ValueError("--exact and --bins are mutually exclusive")
    if bins is not None and bins < 1:
        raise ValueError("--bins must be positive")
    start, end = parse_range(step_text)
    records = select(load_csv(run, tag), start, end)
    if not records:
        raise ValueError(f"scalar tag {tag!r} has no records in the selected range")
    finite, duplicates, nonfinite = dedup(records)
    if exact:
        headers = ["step", "value", "wall_time"]
        rows = [[r["step"], repr(r["value"]), r["wall_time"]] for r in records]
    else:
        lo = start if start is not None else min(r["step"] for r in records)
        hi = end if end is not None else max(r["step"] for r in records)
        headers = [
            "interval",
            "count",
            "first_step",
            "first",
            "last_step",
            "last",
            "mean",
            "min",
            "min_step",
            "max",
            "max_step",
            "duplicates",
            "nonfinite",
        ]
        rows = []
        for left, right in bounds(lo, hi, bins):
            values, duplicates, nonfinite = dedup([r for r in records if left <= r["step"] <= right])
            if not values:
                rows.append([f"{left}:{right}", 0, "", "", "", "", "", "", "", "", "", duplicates, nonfinite])
                continue
            minimum = min(values, key=lambda r: r["value"])
            maximum = max(values, key=lambda r: r["value"])
            rows.append(
                [
                    f"{left}:{right}",
                    len(values),
                    values[0]["step"],
                    num(values[0]["value"]),
                    values[-1]["step"],
                    num(values[-1]["value"]),
                    num(sum(r["value"] for r in values) / len(values)),
                    num(minimum["value"]),
                    minimum["step"],
                    num(maximum["value"]),
                    maximum["step"],
                    duplicates,
                    nonfinite,
                ]
            )
    print(table(headers, rows))
    if exact:
        print(f"records={len(records)} duplicates={duplicates} nonfinite={nonfinite}")
    if output:
        with output.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(rows)


def stats(values, steps):
    first, last = steps[0], steps[-1]
    minimum = min(steps, key=lambda s: values[s])
    maximum = max(steps, key=lambda s: values[s])
    return [
        len(steps),
        num(sum(values[s] for s in steps) / len(steps)),
        num(values[first]),
        first,
        num(values[last]),
        last,
        num(values[minimum]),
        minimum,
        num(values[maximum]),
        maximum,
    ]


def compare(run_a, run_b, tag, step_text, bins, output):
    if bins is not None and bins < 1:
        raise ValueError("--bins must be positive")
    start, end = parse_range(step_text)
    raw_a, raw_b = select(load_csv(run_a, tag), start, end), select(load_csv(run_b, tag), start, end)
    if not raw_a or not raw_b:
        raise ValueError(f"scalar tag {tag!r} is missing from one selected range")
    finite_a, dup_a, nonfinite_a = dedup(raw_a)
    finite_b, dup_b, nonfinite_b = dedup(raw_b)
    values_a = {r["step"]: r["value"] for r in finite_a}
    values_b = {r["step"]: r["value"] for r in finite_b}
    lo = start if start is not None else min(r["step"] for r in raw_a + raw_b)
    hi = end if end is not None else max(r["step"] for r in raw_a + raw_b)
    common = sorted(set(values_a) & set(values_b))
    headers = [
        "interval",
        "common",
        "A_count",
        "A_mean",
        "A_first",
        "A_first_step",
        "A_last",
        "A_last_step",
        "A_min",
        "A_min_step",
        "A_max",
        "A_max_step",
        "B_count",
        "B_mean",
        "B_first",
        "B_first_step",
        "B_last",
        "B_last_step",
        "B_min",
        "B_min_step",
        "B_max",
        "B_max_step",
        "delta_B-A",
        "A_unmatched",
        "B_unmatched",
    ]
    rows = []
    for left, right in bounds(lo, hi, bins):
        common_steps = [s for s in common if left <= s <= right]
        a_steps = [s for s in values_a if left <= s <= right]
        b_steps = [s for s in values_b if left <= s <= right]
        a = stats(values_a, common_steps) if common_steps else [0] + [""] * 9
        b = stats(values_b, common_steps) if common_steps else [0] + [""] * 9
        if common_steps:
            mean_a = sum(values_a[s] for s in common_steps) / len(common_steps)
            mean_b = sum(values_b[s] for s in common_steps) / len(common_steps)
            delta = num(mean_b - mean_a)
        else:
            delta = ""
        rows.append(
            [
                f"{left}:{right}",
                len(common_steps),
                *a,
                *b,
                delta,
                len(a_steps) - len(common_steps),
                len(b_steps) - len(common_steps),
            ]
        )
    print(f"A={run_a} duplicates={dup_a} nonfinite={nonfinite_a}")
    print(f"B={run_b} duplicates={dup_b} nonfinite={nonfinite_b}")
    print(table(headers, rows))
    if output:
        with output.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("run", type=Path)
    p = sub.add_parser("tags")
    p.add_argument("run", type=Path)
    p.add_argument("--match", default="")
    p = sub.add_parser("curve")
    p.add_argument("run", type=Path)
    p.add_argument("--tag", required=True)
    p.add_argument("--steps")
    p.add_argument("--bins", type=int)
    p.add_argument("--exact", action="store_true")
    p.add_argument("--output", type=Path)
    p = sub.add_parser("compare")
    p.add_argument("run_a", type=Path)
    p.add_argument("run_b", type=Path)
    p.add_argument("--tag", required=True)
    p.add_argument("--steps")
    p.add_argument("--bins", type=int)
    p.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare(args.run)
        elif args.command == "tags":
            ensure_prepared(args.run)
            index = json.loads((args.run / "analysis/index.json").read_text())
            rows = [
                [tag, data["records"], data["min_step"], data["max_step"]]
                for tag, data in index["tags"].items()
                if args.match in tag
            ]
            print(table(["tag", "records", "min_step", "max_step"], rows))
        elif args.command == "curve":
            ensure_prepared(args.run)
            curve(args.run, args.tag, args.steps, args.bins, args.exact, args.output)
        else:
            ensure_prepared(args.run_a)
            ensure_prepared(args.run_b)
            compare(args.run_a, args.run_b, args.tag, args.steps, args.bins, args.output)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
