# Data Directory

```text
data/
├── pff_match_data/     RESTRICTED - PFF FC match data (NOT in git)
├── derived/            Generated pass tables (NOT in git; regenerated)
└── scenarios/          Small scenario configs (in git)
```

## `pff_match_data/` — licensed PFF data (you provide it)

This folder is where authorized users place the PFF FC World Cup 2022
event/tracking data. It is **gitignored**: the repository does not
redistribute this data, and nothing under this folder (or any derivative
pass table) may be committed. See **[docs/data_access.md](../docs/data_access.md)**
for the full acquisition tutorial, expected filenames, and schemas.

Expected contents (per licensed match `id`):

```
<id>.json                  possession-event JSON        (required by pitch control)
match_<id>_players.csv     per-event player positions   (required by pitch control)
match_<id>_ball.csv        per-event ball positions     (required by pitch control)
<id>_meta.json             match metadata               (optional)
<id>.jsonl                 raw tracking frames          (pass-table builder, DAS 3D)
events/  tracking/         optional subfolders for the tournament-wide build
```

A `.gitkeep` keeps the (empty) folder in git.

## `derived/` — generated tables (regenerable)

Created by `scripts/build_worldcup_passes_csv.py` and the curvature
scripts; gitignored because they are derived from PFF data and large
(the tournament table is ~150 MB):

```
worldcup2022_passes.csv             one row per pass; ball trajectory embedded
                                    as stringified lists (frame_timestamps_ms,
                                    ball_x/y/z, ball_visibility)
3812_passes_with_curvature.csv      match-3812 rows + curvature columns
                                    (written by notebooks/pass_curvature_analysis.ipynb)
```

## `scenarios/` — small configs (committed)

```
best_pass_scenario.json   passer/receiver/defender coordinates of the blocked-lane
                          scenario used by the DAS xC sweep (synthetic study input;
                          contains only position coordinates, no PFF payloads)
```
