# Pitch-Control Module

Pitch-control, xC and OBSO code for the curved-pass project, operating on
**PFF FC World Cup 2022** data (restricted; placed in `../data/pff_match_data/`).

> Run scripts from this directory (`pitch_control/`) — imports resolve
> `LaurieOnTracking` from here.

## Contents

### Vendored library: `LaurieOnTracking/`

A fork of [Laurie Shaw's LaurieOnTracking](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking)
(MIT license retained in `LaurieOnTracking/LICENSE`), extended for PFF data:

- `Metrica_IO.py` — **`load_new_data(data_dir, match_id)`** is the single
  data entry point. Expects per match: `<id>.json`, `match_<id>_players.csv`,
  `match_<id>_ball.csv` (raises `FileNotFoundError` listing missing files).
  Coordinates are converted meters → Metrica-normalized; each script converts
  back to meters as needed. Also contains the original Metrica sample-data
  loaders and the upstream tutorials (`Tutorial1..4`).
- `Metrica_PitchControl.py` — the models:
  - `generate_pitch_control_for_event` — baseline Spearman (2018) model
    (straight-line ball travel, `default_model_params()`).
  - `generate_pitch_control_for_event_curve_based` — **curvature-aware
    variant**: ball travel is a quadratic Bézier whose bulge is derived from
    a discretized curvature class (`params['trajectory_classes']`, 5 classes
    with empirical probabilities; class values were derived from the
    tournament curvature histogram — see `results/curvature/trajectory_classes.csv`
    and `scripts/curvature_histograms.py`), so each cell's control
    probability averages over interception risk along curved ball paths.
  - linearized / Gaussian / exponential approximation models (research
    comparisons).
- `Metrica_Velocities.py`, `Metrica_Viz.py`, `Metrica_EPV.py` — upstream
  modules (velocities, plotting, EPV).
- `EPV_grid.csv`, `Transition_gauss.csv` — OBSO lookup grids.

### Project scripts (this folder)

| Script | Purpose |
|---|---|
| `compute_all_events_error_summary.py` | **Headless.** Baseline vs linearized-model error/timing over all match-3812 events → `../results/pitch_control/all_events_*.csv` |
| `batch_reconstruction_distance_errors.py` | Headless batch for the constraint-1 reverse solver |
| `reverse_pitch_control_solver.py` | Constraint-based reverse solver + `--evaluate-all-events` batch |
| `reverse_pitch_control_lstsq.py` | Gauss-Newton / damped-least-squares reverse solver |
| `run_pitch_control.py` | GUI: baseline vs approximation models for one event |
| `pitch_control_comparison_gui.py` | GUI: **straight vs curve-based PPCF** + squared-error map |
| `obso_pitch_control_comparison_gui.py` | GUI: PPCF + OBSO, straight vs curved, multi-match (team names via `<id>_meta.json`) |
| `run_pitch_control_reconstructed_compare.py` | GUI: PPCF → purge frame → re-solve positions → reconstructed PPCF |
| `run_pitch_control_formation_gui.py` | GUI: fit formation templates to a PPCF heatmap |
| `run_pitch_control_nudge_gui.py` | GUI: animate players toward a target event's PPCF |
| `reverse_pitch_control_gui.py` | GUI: reverse solver on a single event |
| `tune_linearized_params.py` | Hyperparameter sweep (⚠ signature drift — see docs/reproducibility.md §3.4) |
| `obso_player.py` | OBSO function library (`calc_obso`, `calc_obso_curve_based`, player-value extraction) |

Result CSVs live in `../results/pitch_control/`.

## Quick start

```bash
# from the repository root, with PFF data installed (see docs/data_access.md):
cd pitch_control
python compute_all_events_error_summary.py        # headless check
python pitch_control_comparison_gui.py            # straight vs curved PPCF heatmaps
python obso_pitch_control_comparison_gui.py       # PPCF + OBSO comparison
```

Known quirks and their documentation: `../docs/reproducibility.md` §3
(25 fps vs 29.97, +53/106 normalization, missing 3813.json / 3812_meta.json,
broken legacy code paths).
