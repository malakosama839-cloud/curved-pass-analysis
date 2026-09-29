"""Path configuration for the curved-pass repository.

Every script and notebook resolves data locations through this module (or
through the same convention), so no machine-specific absolute paths are
hard-coded anywhere. Override the PFF data location without editing code by
setting the environment variable ``CURVED_PASSES_PFF_DATA``.

Repository layout (all repo-root-relative):

    data/pff_match_data/    restricted PFF data (NOT in git; you provide it)
    data/derived/           generated pass tables (NOT in git; regenerated)
    data/scenarios/         small scenario JSONs (in git)
    figures/                generated and committed figures
    results/                generated and committed result tables
"""
from __future__ import annotations

import os
from pathlib import Path

# .../curved-pass-analysis/src/curved_passes/config.py -> repo root is 3 up
REPO_ROOT = Path(__file__).resolve().parents[2]

PFF_DATA_DIR = Path(
    os.environ.get("CURVED_PASSES_PFF_DATA", REPO_ROOT / "data" / "pff_match_data")
)
DERIVED_DATA_DIR = REPO_ROOT / "data" / "derived"
SCENARIO_DIR = REPO_ROOT / "data" / "scenarios"
FIGURES_DIR = REPO_ROOT / "figures"
RESULTS_DIR = REPO_ROOT / "results"

# Master pass table built by scripts/build_worldcup_passes_csv.py
WORLDCUP_PASSES_CSV = DERIVED_DATA_DIR / "worldcup2022_passes.csv"


def pff_event_json(match_id) -> Path:
    """PFF possession-event JSON for a match, e.g. data/pff_match_data/3812.json."""
    return PFF_DATA_DIR / f"{match_id}.json"


def pff_tracking_jsonl(match_id) -> Path:
    """PFF raw tracking file for a match, e.g. data/pff_match_data/3812.jsonl."""
    return PFF_DATA_DIR / f"{match_id}.jsonl"


def pff_players_csv(match_id) -> Path:
    """Per-event player positions, e.g. data/pff_match_data/match_3812_players.csv."""
    return PFF_DATA_DIR / f"match_{match_id}_players.csv"


def pff_ball_csv(match_id) -> Path:
    """Per-event ball positions, e.g. data/pff_match_data/match_3812_ball.csv."""
    return PFF_DATA_DIR / f"match_{match_id}_ball.csv"
