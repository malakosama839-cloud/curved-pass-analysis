"""Build the World Cup 2022 pass table (worldcup2022_passes.csv) from PFF data.

This is the data-processing workflow that produced the tournament pass table
used by the curvature analysis (originally a Google Colab notebook,
``fifa2022_worldcup_passes_colab.ipynb``; logic unchanged).

For every match it needs two authorized PFF FC inputs (see docs/data_access.md):

  * event data:        <match_id>.json          (possession-event JSON array)
  * raw tracking:      <match_id>.jsonl          (one JSON frame per line; may be
                       provided zipped as .zip/.bz2/.gz/.tar/.tgz and is
                       extracted automatically)

and writes one row per pass to the output CSV. Each row carries the event
metadata plus the full ball trajectory of the pass, embedded as stringified
lists in the columns ``frame_numbers``, ``frame_timestamps_ms``, ``ball_x``,
``ball_y``, ``ball_z``, ``ball_visibility``. Trajectory frames are selected
as every tracking frame whose timestamp (videoTimeMs) falls inside the pass
event's [startTime, endTime] interval; if the interval is degenerate, the
nearest frame to the event start is used ("nearest_event_frame").

The script is resumable: matches already present in the output CSV are
skipped, so an interrupted build can simply be re-run.

Usage
-----
    python scripts/build_worldcup_passes_csv.py \
        --tracking-dir /path/to/pff/tracking \
        --event-dir    /path/to/pff/events \
        --output       data/derived/worldcup2022_passes.csv

Defaults read from the environment variable CURVED_PASSES_PFF_DATA (or
data/pff_match_data) looking for ``tracking/`` and ``events/`` subfolders.
"""
import argparse
import bz2
import csv
import gzip
import json
import math
import re
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from curved_passes.config import PFF_DATA_DIR, DERIVED_DATA_DIR  # noqa: E402

ARCHIVE_SUFFIXES = {".zip", ".bz2", ".gz", ".tar", ".tgz"}

OUTCOME_LABELS = {
    "C": "complete",
    "D": "deflected",
    "B": "blocked",
    "O": "out of play",
    "S": "stopped",
}

CSV_HEADERS = [
    "match_id", "game_event_id", "possession_event_id", "period",
    "start_time", "end_time", "event_time", "game_clock", "team_id",
    "passer_id", "receiver_id", "pass_success", "pass_outcome_code",
    "pass_outcome", "pass_type", "pass_category", "frame_count",
    "frame_selection", "valid_coordinate_count", "curve_status",
    "path_length_3d", "chord_length_3d", "straightness_3d",
    "max_deviation_3d", "max_deviation_x", "max_deviation_y",
    "max_deviation_z", "x_range", "y_range", "z_range",
    "frame_numbers", "frame_timestamps_ms",
    "ball_x", "ball_y", "ball_z", "ball_visibility",
]


# ---------------------------------------------------------------
# archive handling: the PFF download ships tracking as .bz2/.zip
# ---------------------------------------------------------------
def _decompress_single(src, dst_dir):
    """Decompress a single-file archive (.bz2 / .gz) into dst_dir."""
    dst = dst_dir / src.stem          # 3812.jsonl.bz2 -> 3812.jsonl
    opener = bz2.open if src.suffix == ".bz2" else gzip.open
    with opener(src, "rb") as fin, open(dst, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    return dst


def extract_one(src, dst_dir):
    """Extract one archive; returns list of produced files (may include nested archives)."""
    suffix = src.suffix.lower()
    produced = []
    if suffix == ".zip":
        with zipfile.ZipFile(src) as zf:
            zf.extractall(dst_dir)
            produced.extend(dst_dir / n for n in zf.namelist() if not n.endswith("/"))
    elif suffix in (".bz2", ".gz"):
        produced.append(_decompress_single(src, dst_dir))
    elif suffix in (".tar", ".tgz") or src.name.lower().endswith(".tar.gz"):
        with tarfile.open(src) as tf:
            tf.extractall(dst_dir)
            produced.extend(dst_dir / m.name for m in tf.getmembers() if m.isfile())
    return produced


def extract_all(src_dir, dst_dir):
    """Extract every archive under src_dir into dst_dir (recursively)."""
    queue = [p for p in Path(src_dir).rglob("*") if p.is_file()]
    extracted, skipped = [], []
    while queue:
        src = queue.pop(0)
        suffix = src.suffix.lower()
        if suffix in ARCHIVE_SUFFIXES:
            produced = extract_one(src, dst_dir)
            extracted.extend(produced)
            queue.extend(p for p in produced if p.suffix.lower() in ARCHIVE_SUFFIXES)
        elif suffix in (".jsonl", ".json"):
            extracted.append(src)  # already plain
        else:
            skipped.append(src.name)
    return extracted, skipped


# ---------------------------------------------------------------
# PFF parsing
# ---------------------------------------------------------------
def load_pass_events(event_path):
    """Index all pass events ('PA' = possession action) from a PFF event JSON."""
    with open(event_path, "r", encoding="utf-8") as handle:
        events = json.load(handle)

    passes = {}
    for event in events:
        possession = event.get("possessionEvents") or {}
        if possession.get("possessionEventType") != "PA":
            continue

        outcome_code = possession.get("passOutcomeType")
        pass_type = possession.get("passType")
        passes[event.get("possessionEventId")] = {
            "match_id": event.get("gameId"),
            "game_event_id": event.get("gameEventId"),
            "possession_event_id": event.get("possessionEventId"),
            "period": possession.get("period") or (event.get("gameEvents") or {}).get("period"),
            "start_time": event.get("startTime"),
            "end_time": event.get("endTime"),
            "event_time": event.get("eventTime"),
            "game_clock": possession.get("formattedGameClock"),
            "team_id": (event.get("gameEvents") or {}).get("teamId"),
            "passer_id": possession.get("passerPlayerId"),
            "receiver_id": possession.get("receiverPlayerId"),
            "pass_outcome_code": outcome_code,
            "pass_outcome": OUTCOME_LABELS.get(outcome_code, outcome_code),
            "pass_success": outcome_code == "C",
            "pass_type": pass_type,
            "pass_category": "foul" if pass_type == "F" else ("normal" if pass_type else "other"),
        }
    return passes


def load_tracking_frames(tracking_path):
    """Read ball positions from a PFF raw tracking .jsonl (one frame per line)."""
    frames = []
    with open(tracking_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            ball = record.get("ballsSmoothed") or record.get("balls") or {}
            frames.append({
                "timestamp": record.get("videoTimeMs"),
                "frame": record.get("frameNum"),
                "x": ball.get("x"),
                "y": ball.get("y"),
                "z": ball.get("z"),
                "visibility": ball.get("visibility"),
            })
    return frames


def select_frames(event, tracking_frames):
    """Every frame inside the event's [start_ms, end_ms] interval (or nearest)."""
    start_ms = event.get("start_time") * 1000 if event.get("start_time") is not None else None
    end_ms = event.get("end_time") * 1000 if event.get("end_time") is not None else None
    if start_ms is None or end_ms is None:
        return [], "missing_event_time"

    if end_ms > start_ms:
        return [
            frame for frame in tracking_frames
            if frame["timestamp"] is not None and start_ms <= frame["timestamp"] <= end_ms
        ], "event_interval"

    nearest = min(
        (frame for frame in tracking_frames if frame["timestamp"] is not None),
        key=lambda frame: abs(frame["timestamp"] - start_ms),
        default=None,
    )
    return ([nearest] if nearest is not None else []), "nearest_event_frame"


def curve_metrics(frames):
    """Pass-level summary geometry (path length, chord, straightness, deviations)."""
    points = [
        (frame["x"], frame["y"], frame["z"])
        for frame in frames
        if all(frame[axis] is not None for axis in ("x", "y", "z"))
    ]
    result = {
        "valid_coordinate_count": len(points),
        "curve_status": "insufficient_samples",
        "path_length_3d": None,
        "chord_length_3d": None,
        "straightness_3d": None,
        "max_deviation_3d": None,
        "max_deviation_x": None,
        "max_deviation_y": None,
        "max_deviation_z": None,
        "x_range": None,
        "y_range": None,
        "z_range": None,
    }
    if not points:
        return result

    result["x_range"] = max(point[0] for point in points) - min(point[0] for point in points)
    result["y_range"] = max(point[1] for point in points) - min(point[1] for point in points)
    result["z_range"] = max(point[2] for point in points) - min(point[2] for point in points)
    if len(points) < 3:
        return result

    start = points[0]
    end = points[-1]
    chord = tuple(end[index] - start[index] for index in range(3))
    chord_length = math.sqrt(sum(value * value for value in chord))
    path_length = sum(
        math.sqrt(sum((current[index] - previous[index]) ** 2 for index in range(3)))
        for previous, current in zip(points, points[1:])
    )
    deviations = []
    axis_deviations = [[], [], []]
    for point_index, point in enumerate(points):
        fraction = point_index / (len(points) - 1)
        expected = tuple(start[index] + fraction * chord[index] for index in range(3))
        axis_errors = [abs(point[index] - expected[index]) for index in range(3)]
        axis_deviations[0].append(axis_errors[0])
        axis_deviations[1].append(axis_errors[1])
        axis_deviations[2].append(axis_errors[2])
        deviations.append(math.sqrt(sum(error * error for error in axis_errors)))

    result.update({
        "curve_status": "measured",
        "path_length_3d": path_length,
        "chord_length_3d": chord_length,
        "straightness_3d": chord_length / path_length if path_length else 1.0,
        "max_deviation_3d": max(deviations),
        "max_deviation_x": max(axis_deviations[0]),
        "max_deviation_y": max(axis_deviations[1]),
        "max_deviation_z": max(axis_deviations[2]),
    })
    return result


def row_for_pass(event, frames, selection_method):
    metrics = curve_metrics(frames)
    row = dict(event)
    row.update(metrics)
    row.update({
        "frame_count": len(frames),
        "frame_selection": selection_method,
        "frame_numbers": json.dumps([frame["frame"] for frame in frames], separators=(",", ":")),
        "frame_timestamps_ms": json.dumps([frame["timestamp"] for frame in frames], separators=(",", ":")),
        "ball_x": json.dumps([frame["x"] for frame in frames], separators=(",", ":")),
        "ball_y": json.dumps([frame["y"] for frame in frames], separators=(",", ":")),
        "ball_z": json.dumps([frame["z"] for frame in frames], separators=(",", ":")),
        "ball_visibility": json.dumps([frame["visibility"] for frame in frames], separators=(",", ":")),
    })
    return row


# ---------------------------------------------------------------
# main build loop (resumable, one CSV row per pass)
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build worldcup2022_passes.csv from PFF data.")
    default_root = Path(PFF_DATA_DIR)
    parser.add_argument("--tracking-dir", type=Path, default=default_root / "tracking",
                        help="Folder containing PFF tracking .jsonl (or archives of them)")
    parser.add_argument("--event-dir", type=Path, default=default_root / "events",
                        help="Folder containing PFF event <match_id>.json files")
    parser.add_argument("--output", type=Path, default=DERIVED_DATA_DIR / "worldcup2022_passes.csv",
                        help="Output CSV path (default: data/derived/worldcup2022_passes.csv)")
    args = parser.parse_args()

    extract_dir = args.tracking_dir.parent / "_tracking_extracted"

    # If the event/tracking subfolders don't exist, fall back to scanning the
    # data root itself (files placed flat in data/pff_match_data).
    event_dir = args.event_dir if args.event_dir.exists() else default_root
    tracking_dir = args.tracking_dir if args.tracking_dir.exists() else default_root

    extract_dir.mkdir(parents=True, exist_ok=True)
    extracted, skipped = extract_all(tracking_dir, extract_dir)
    jsonl_files = sorted(extract_dir.rglob("*.jsonl"))
    print(f"extracted/decompressed: {len(extracted)} item(s)")
    print(f"tracking (.jsonl) files ready: {len(jsonl_files)}")
    if skipped:
        print("skipped (unknown type):", skipped[:10])

    # index tracking files by match id (matches 3812.jsonl, match_3812.jsonl, ...)
    tracking_index = {}
    for p in jsonl_files:
        m = re.search(r"(\d{4,})", p.stem)
        if m:
            tracking_index[int(m.group(1))] = p

    # event files: numeric-named JSONs in the event folder
    event_files = sorted(
        p for p in Path(event_dir).rglob("*.json")
        if re.fullmatch(r"\d+", p.stem) and "_meta" not in p.stem
    )
    print(f"{len(event_files)} event file(s), {len(tracking_index)} tracking file(s)")

    # resume support: which match ids are already in the output CSV
    done_matches = set()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        with out_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            next(reader, None)  # header
            for r in reader:
                if r:
                    done_matches.add(r[0])
        print(f"resuming: {len(done_matches)} match(es) already in {out_path.name}")

    write_header = not out_path.exists()
    new_matches = total_rows = skipped_matches = 0

    with out_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_HEADERS)
        if write_header:
            writer.writeheader()

        for ev_path in event_files:
            match_id = ev_path.stem
            if match_id in done_matches:
                continue

            tr_path = tracking_index.get(int(match_id))
            if tr_path is None:
                print(f"match {match_id}: NO tracking file found, skipped")
                skipped_matches += 1
                continue

            passes = load_pass_events(ev_path)
            frames = load_tracking_frames(tr_path)

            n_rows = 0
            for event in passes.values():
                sel, method = select_frames(event, frames)
                row = row_for_pass(event, sel, method)
                writer.writerow({h: row.get(h) for h in CSV_HEADERS})
                n_rows += 1
            fh.flush()

            new_matches += 1
            total_rows += n_rows
            print(f"match {match_id}: {n_rows} pass rows  ({new_matches} matches done)")

    print()
    print(f"DONE -> {out_path}")
    print(f"new matches: {new_matches}, new pass rows: {total_rows}, skipped (no tracking): {skipped_matches}")


if __name__ == "__main__":
    main()
