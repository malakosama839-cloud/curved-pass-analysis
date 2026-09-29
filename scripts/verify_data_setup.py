"""Verify that the restricted PFF data is installed correctly.

Checks data/pff_match_data (or CURVED_PASSES_PFF_DATA) for the files each
analysis expects, per match, and prints a PASS/FAIL report:

  Required by the pitch-control pipeline (LaurieOnTracking.load_new_data):
      <id>.json                 possession-event JSON
      match_<id>_players.csv    per-event player positions
      match_<id>_ball.csv       per-event ball positions
  Optional:
      <id>_meta.json            match metadata (team names, fps, pitch size)
      <id>.jsonl                raw tracking frames (WC pass-table builder,
                                DAS 3D notebook)

Usage:
    python scripts/verify_data_setup.py
    python scripts/verify_data_setup.py --match-ids 3812 10517
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from curved_passes.config import PFF_DATA_DIR, pff_event_json, pff_players_csv, pff_ball_csv

# Matches referenced by the analyses shipped in this repository.
DEFAULT_MATCH_IDS = ["3812", "3813", "3814", "10517"]


def check_match(match_id):
    required = {
        "event JSON (<id>.json)": pff_event_json(match_id),
        "player positions (match_<id>_players.csv)": pff_players_csv(match_id),
        "ball positions (match_<id>_ball.csv)": pff_ball_csv(match_id),
    }
    optional = {
        "match metadata (<id>_meta.json)": PFF_DATA_DIR / f"{match_id}_meta.json",
        "raw tracking (<id>.jsonl)": PFF_DATA_DIR / f"{match_id}.jsonl",
    }
    print(f"\nMatch {match_id}")
    ok = True
    for label, path in required.items():
        status = "PASS" if path.exists() else "FAIL (required)"
        ok &= path.exists()
        print(f"  [{status:>15}] {label}: {path.name}")
    for label, path in optional.items():
        status = "present" if path.exists() else "missing (optional)"
        print(f"  [{status:>15}] {label}: {path.name}")

    # Light schema check on the player CSV (the file the pitch-control
    # loader actually parses).
    players_csv = pff_players_csv(match_id)
    if players_csv.exists():
        try:
            import pandas as pd
            head = pd.read_csv(players_csv, nrows=2)
            expected_cols = {"gameid", "gameeventid", "possessioneventid", "eventtime",
                             "playerid", "positiongrouptype", "jerseynum", "team",
                             "x", "y", "period"}
            missing = expected_cols - set(head.columns)
            if missing:
                print(f"  [FAIL (required)] players CSV is missing columns: {sorted(missing)}")
                ok = False
            else:
                print("  [          PASS] players CSV schema check")
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL (required)] players CSV could not be read: {exc}")
            ok = False

    # Meta sanity print
    meta_path = PFF_DATA_DIR / f"{match_id}_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(meta, list):
                meta = meta[0]
            home = meta.get("homeTeam", {}).get("name", "?")
            away = meta.get("awayTeam", {}).get("name", "?")
            fps = meta.get("fps", "?")
            print(f"  info: {home} vs {away}, fps={fps}")
        except Exception:  # noqa: BLE001
            pass
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--match-ids", nargs="*", default=DEFAULT_MATCH_IDS)
    args = parser.parse_args()

    print(f"PFF data directory: {PFF_DATA_DIR}")
    if not PFF_DATA_DIR.exists():
        print("\nThe data directory does not exist yet.")
        print("Create it and place the authorized PFF match data inside; see")
        print("docs/data_access.md for the full step-by-step tutorial.")
        sys.exit(1)

    all_ok = True
    for match_id in args.match_ids:
        all_ok &= check_match(match_id)

    print("\n" + "=" * 60)
    if all_ok:
        print("All required files found. You can run the analyses:")
        print("  pitch control :  python pitch_control/compute_all_events_error_summary.py")
        print("  DAS notebooks :  notebooks/das_curved_pass_study_3D.ipynb")
    else:
        print("Some required files are MISSING.")
        print("Follow docs/data_access.md to obtain and place the PFF data,")
        print("or drop matches from this list if you did not license them.")
    print("=" * 60)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
