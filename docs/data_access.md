# Obtaining and Installing the PFF Match Data (Authorized Users Only)

> **This repository does NOT redistribute the PFF match data.**
> The PFF FC tracking and event data used in this project is licensed
> material. It is excluded from version control via `.gitignore`, and no
> part of it (or any derivative pass table built from it) may be uploaded,
> copied into the repository, or otherwise redistributed. Every user must
> obtain it from PFF under their own license.

---

## 1. What PFF data is required

The analyses in this repository use **PFF FC event and tracking data from
the FIFA World Cup 2022**. Concretely, per match:

| File | Role | Required by |
|---|---|---|
| `<match_id>.json` | Possession-event JSON array (passes, shots, passer/receiver, timestamps) | Pitch-control pipeline, DAS 3D notebook, pass-table builder |
| `<match_id>.jsonl` | Raw tracking, one JSON frame per line (ball + player positions per `frameNum`, `videoTimeMs`) | Pass-table builder (`worldcup2022_passes.csv`), DAS 3D notebook |
| `match_<match_id>_players.csv` | One row **per player per possession event** (x, y in meters, center-origin) | Pitch-control pipeline (`Metrica_IO.load_new_data`) |
| `match_<match_id>_ball.csv` | One row **per possession event** (ball_x/y/z) | Pitch-control pipeline (`Metrica_IO.load_new_data`) |
| `<match_id>_meta.json` | Match metadata: team names/kits, periods, `fps`, stadium pitch size (105 x 68 m) | Team-name normalization in the OBSO GUI (optional but recommended) |

The matches used in the reported work are:

| `match_id` | Fixture | Used for |
|---|---|---|
| `3812` | Netherlands – Senegal (group stage) | Core use case: DAS curved-pass study, pitch-control comparison, curvature statistics |
| `3814` | Qatar – Ecuador | Pitch-control GUIs (multi-match) |
| `10517` | Argentina – France (final) | Pitch-control GUIs (multi-match) |
| `3813` | — | Player/ball CSVs exist but the event JSON was **not** part of the data package; the match is unloadable. Include the event JSON if you license it. |

The full-tournament pass table (`data/derived/worldcup2022_passes.csv`)
additionally requires tracking + event data for **every World Cup 2022
match** you want to include.

---

## 2. How to obtain the data

1. **Identify yourself as an authorized researcher.** PFF FC data is
   distributed under license. Contact PFF (https://www.pff.com/, or the
   PFF FC data representative your institution works with) and request
   research access to the FIFA World Cup 2022 event + tracking dataset,
   or use an existing license through your university/organization.
2. **Sign the applicable data-license agreement.** Note that the license
   typically prohibits redistribution — which is exactly why this
   repository ships no data.
3. **Download per match:**
   - the possession-event JSON (`<match_id>.json`, optionally
     `<match_id>_meta.json`),
   - the raw tracking file (`<match_id>.jsonl`; PFF usually delivers it
     compressed as `.zip`/`.bz2` — both are supported by the builder
     script, which decompresses them automatically).
4. **The `match_<id>_players.csv` / `match_<id>_ball.csv` exports** are
   flattened, per-possession-event extracts of the event JSON. If your
   data package does not include them, they must be exported once from
   `<id>.json` before the pitch-control pipeline can run: one row per
   player per possession event with columns
   `gameid, gameeventid, possessioneventid, starttime, endtime, duration,
   eventtime, sequence, playerid, positiongrouptype, jerseynum, team, x, y,
   visibility, confidence, possessioneventtype, teamattackingdirection,
   period, teamname` (players) and `gameid, gameeventid,
   possessioneventid, starttime, endtime, duration, eventtime, sequence,
   ball_visibility, ball_x, ball_y, ball_z, period, teamname,
   teamattackingdirection, possessioneventtype` (ball).
   Coordinates are **meters with the pitch-center origin**
   (x ∈ [-52.5, 52.5], y ∈ [-34, 34]); `team` is `H`/`A`.

---

## 3. Where to place the data (expected folder structure)

Place everything under **`data/pff_match_data/`** at the repository root
(create it if a fresh clone — it exists only as a `.gitkeep` placeholder):

```text
curved-pass-analysis/
└── data/
    └── pff_match_data/              <- PFF_DATA_DIR (gitignored)
        ├── 3812.json                <- event JSON, match 3812
        ├── 3812.jsonl               <- raw tracking, match 3812
        ├── 3812_meta.json           <- metadata, match 3812 (if provided)
        ├── 3814.json
        ├── 3814_meta.json
        ├── 3814.jsonl
        ├── 10517.json
        ├── 10517_meta.json
        ├── 10517.jsonl
        ├── match_3812_players.csv
        ├── match_3812_ball.csv
        ├── match_3813_players.csv   <- (event JSON 3813.json missing: not loadable)
        ├── match_3813_ball.csv
        ├── match_3814_players.csv
        ├── match_3814_ball.csv
        ├── match_10517_players.csv
        └── match_10517_ball.csv
```

Two supported layout variants for the **full-tournament pass-table build**
(`scripts/build_worldcup_passes_csv.py`):

```text
data/pff_match_data/
├── events/          <- all <match_id>.json event files
└── tracking/        <- all <match_id>.jsonl files (or .zip/.bz2 archives)
```

…or simply keep everything flat in `data/pff_match_data/` as shown above —
the builder falls back to scanning the data root when the subfolders do
not exist.

### Custom data location

Do not want data inside the repository? Set an environment variable and
every script/notebook convention follows it:

```bash
export CURVED_PASSES_PFF_DATA=/path/to/your/pff/data     # Linux/macOS
set CURVED_PASSES_PFF_DATA=D:\data\pff_match_data        # Windows (cmd)
```

`src/curved_passes/config.py` resolves the directory in this order:
environment variable → `data/pff_match_data`.

> **Never hard-code an absolute path** (e.g. `D:\...`) into a script or
> notebook. If your layout differs, use `CURVED_PASSES_PFF_DATA` or adjust
> `src/curved_passes/config.py` once.

---

## 4. File naming and organization rules

- Event and metadata files are named exactly `<match_id>.json` /
  `<match_id>_meta.json`, where `match_id` is PFF's numeric game id
  (`3812`, `3814`, `10517`, …).
- Tracking files are named `<match_id>.jsonl` (the builder's regex also
  accepts `match_3812.jsonl`-style names — any name whose first long
  number is the match id).
- The player/ball exports must be exactly
  `match_<match_id>_players.csv` and `match_<match_id>_ball.csv`
  (`load_new_data` looks these names up verbatim).
- Do not rename, re-indent, or re-serialize the JSON files.

## 5. Preprocessing required before running the analyses

1. **Decompression** of tracking archives: automatic —
   `build_worldcup_passes_csv.py` extracts `.zip/.bz2/.gz/.tar` (including
   nested archives) into `<tracking-dir>/../_tracking_extracted/`.
2. **Player/ball CSV export** from `<id>.json`: only needed if not
   supplied in your data package (see §2.4 for the expected schema).
3. Nothing else — the pass table, curvature values, and figures are all
   generated by the scripts in `scripts/`.

## 6. How the code finds and loads the data

| Component | Loading mechanism |
|---|---|
| Pass-table builder (`scripts/build_worldcup_passes_csv.py`) | `--event-dir` / `--tracking-dir` args (defaults from `src/curved_passes/config.py`); matches tracking files to events by numeric match id |
| Pitch-control pipeline (`pitch_control/*.py`) | `MATCH_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "pff_match_data")`, then `Metrica_IO.load_new_data(MATCH_DATA_PATH, game_id)`, which requires the three files of §1 verbatim and raises `FileNotFoundError` listing what is missing |
| DAS 3D notebook (`notebooks/das_curved_pass_study_3D.ipynb`) | Reads `../data/pff_match_data/3812.json` (events) and `../data/pff_match_data/3812.jsonl` (first tracking frame of selected events) relative to `notebooks/` |
| Curvature scripts (`scripts/curvature_*.py`) | Read the derived pass table `data/derived/worldcup2022_passes.csv` (built from the PFF data by the builder), not the PFF files directly |

The event JSON schema used downstream: each element carries
`gameId, gameEventId, possessionEventId, startTime, endTime, duration,
eventTime, sequence, gameEvents{...}, possessionEvents{...}, ball[...]`
(plus `homePlayers`/`awayPlayers` in some matches). Passes are identified
by `possessionEvents.possessionEventType == "PA"`; completion by
`passOutcomeType == "C"`.

## 7. Verify the installation

```bash
python scripts/verify_data_setup.py
```

This checks every expected file per match (required vs optional), validates
the players-CSV schema, prints match metadata when available, and exits
non-zero if something required is missing.

## 8. Run the analyses

Once `verify_data_setup.py` passes:

```bash
# full-tournament pass table (needed by the curvature analyses)
python scripts/build_worldcup_passes_csv.py

# curvature heat map + histograms
python scripts/curvature_heatmap_breakdown.py
python scripts/curvature_histograms.py

# pitch-control error summary for match 3812 (headless)
python pitch_control/compute_all_events_error_summary.py

# DAS curved-pass study (needs accessible-space)
jupyter lab notebooks/das_curved_pass_study_3D.ipynb
```

See the repository README and `docs/reproducibility.md` for the complete
runbook and per-result provenance.
