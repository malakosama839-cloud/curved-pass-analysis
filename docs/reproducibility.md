# Reproducibility Guide

This document is the provenance map of the project: for every major result,
it lists **Input → Code → Processing → Output**, plus the execution order,
parameters, and known caveats. The README is the entry point; this document
is the audit trail.

All paths are relative to the repository root. `data/pff_match_data/` and
`data/derived/` are gitignored (restricted/licensed or regenerable data).

---

## 0. Environment

- Python **3.12** (3.10+ should work; the project was developed on 3.12-3.14).
- `pip install -r requirements.txt`
- The DAS notebooks additionally need the `accessible-space` library
  (see the note in `requirements.txt` — install from GitHub if the PyPI
  wheel is unavailable, and record the commit hash you used).
- No notebook needs Google Colab: all Colab drive mounts and sandbox paths
  (`/content/drive/...`, `/mnt/user-data/...`) were repointed to repository
  paths when the notebooks were imported.

---

## 1. Provenance of every major result

### 1.1 World Cup pass table (`worldcup2022_passes.csv`)

| | |
|---|---|
| **Input** | Authorized PFF event JSONs (`<id>.json`) + raw tracking (`<id>.jsonl`, possibly zipped) in `data/pff_match_data/` |
| **Code** | `scripts/build_worldcup_passes_csv.py` (extracted verbatim from the original `fifa2022_worldcup_passes_colab.ipynb`) |
| **Processing** | per match: parse pass events (`possessionEventType == "PA"`); read ball per tracking frame (`ballsSmoothed`/`balls`, `videoTimeMs`, `frameNum`); select frames inside `[startTime, endTime]` (fallback: nearest frame); compute pass-geometry summary (path length, chord, straightness, max deviation); write one CSV row per pass with the trajectory embedded as stringified lists. Resumable (skips matches already in the CSV) |
| **Output** | `data/derived/worldcup2022_passes.csv` (~66k passes; ~150 MB; gitignored) |

### 1.2 Normalized 3D curvature metric

| | |
|---|---|
| **Input** | per pass: `frame_timestamps_ms`, `ball_x`, `ball_y`, `ball_z` columns of the pass table |
| **Code (canonical)** | `src/curved_passes/curvature.py::curvature_3d_normalized` |
| **Inline copies** | `notebooks/pass_curvature_analysis.ipynb` (cell "calculate_curvature_3d"), `scripts/curvature_heatmap_breakdown.py` (imports the canonical module) |
| **Processing** | `D = ||r_end − r_start||`; perpendicular distances `d_i = ||(r_i − r_0) × chord|| / D`; deviation area `A_dev = Σ 0.5 (d_i + d_{i+1}) · ||r_{i+1} − r_i||`; **C = A_dev / D²** |
| **Output** | `curvature_3d_normalized` column; dimensionless, 0 = straight |

Methodology is preserved exactly as developed — the notebooks keep their
own inline copies of the metric (character-identical math) so the published
workflow can be read end-to-end in one place.

### 1.3 Curvature heat map (spatial distribution)

| | |
|---|---|
| **Input** | `data/derived/worldcup2022_passes.csv` |
| **Code** | `scripts/curvature_heatmap_breakdown.py` |
| **Parameters** | straight threshold 0.02; grid 12 x 8 over 105 x 68 m; fixed color scale vmin 0.07 / vmax 0.29; pitch markings (center circle r 9.15, both penalty areas 16.5 x 40.32) |
| **Output** | `figures/curvature_heatmap/worldcup2022_heatmap_breakdown.png` (committed) |

Runtime: ~20-60 s for the full tournament table.

### 1.4 Curvature histograms (distribution across the match/table)

| | |
|---|---|
| **Input** | `data/derived/worldcup2022_passes.csv` |
| **Code** | `scripts/curvature_histograms.py` (workflow of the original histogram notebook; same analysis also appears in `notebooks/pass_curvature_analysis.ipynb`) |
| **Parameters** | equal-width histograms: 31 bins, threshold 0.02, xlim capped at 5; quantile histograms: threshold 0.01, 20 quantile bins from the pooled success+failure distribution, cumulative-% line on twin axis, xlim capped at 1 (two variants: counts and probability density); statistics threshold 0.06; trajectory classes: 5 quantile bins |
| **Output** | `figures/curvature_histogram/histograms_equal_bins.png`, `histograms_quantile_bins.png`, `histograms_quantile_density.png`; `results/curvature/curved_pass_statistics.csv`, `trajectory_classes.csv` |

The reported statistics (source of the priors used downstream): curved-pass
rate ≈ **41.12%** among successful passes vs **55.70%** among failed passes
(threshold 0.06). `trajectory_classes.csv` is the empirical derivation of
the 5 curvature classes hard-coded as `trajectory_classes` in the
curve-based pitch-control model (`pitch_control/LaurieOnTracking/Metrica_PitchControl.py`).

Note on `histograms_equal_bins.png`: a small number of blocked/deflected
passes travel almost perpendicular to the start-end chord and reach extreme
curvature values (max observed C ≈ 793), so with the original equal-width
binning nearly all mass falls into the first bin and the remaining bins look
empty. This is faithful to the original notebook; the **quantile-bin**
variants (`histograms_quantile_bins.png`, `histograms_quantile_density.png`)
are the informative view of the distribution.

### 1.5 Pass-curvature analysis notebook (interactive)

`notebooks/pass_curvature_analysis.ipynb` — computes the metric per pass,
saves the enriched table, renders an interactive Plotly 3D comparison of a
real curved pass vs synthetic paths, and produces the histograms and
threshold statistics of §1.4. Run top-to-bottom after the pass table exists.

### 1.6 DAS curved-pass xC study

| | |
|---|---|
| **Input** | `data/scenarios/best_pass_scenario.json` (blocked-lane scenario); for the real-match integration: `../data/pff_match_data/3812.json` + `3812.jsonl`; everything else is synthetic |
| **Code** | `notebooks/das_curved_pass_study_3D.ipynb` (canonical; contains the validated 2D module, the 3D height-gated extension, real-match DAS heatmaps, play priors, per-pixel histograms) and `notebooks/das_curved_pass_study_2D.ipynb` (2D-only standalone) |
| **Key library** | `accessible_space.core.simulate_passes`, `accessible_space.motion_models.approx_two_point_time_to_arrive` (+ private constants: pin the library version) |
| **Processing** | synthetic quadratic-Bézier (`bezier_arc`, n=300) / half-ellipsoid 3D paths between passer and receiver; curvature `C = A_dev / D²` via `calculate_curvature_2d`; xC via the competing-risks integral re-implementation `compute_xc_curved` (validated against the library within ~1% for straight paths, seed 0); 3D variant `compute_xc_curved_3d` zeroes defender interception rates when `arc_z > jump_reach (2.2 m)`; DAS heatmaps over 144 angles x library radial grid, curve-weighted via `curve_play_prior(C) = clip(0.4841 · exp(−1.8·C), 0.02, 0.98)` with 0.4841 = mean(0.4112, 0.5570) from §1.4 |
| **Output** | `figures/das_study/*.png` (committed: three scenario sweeps, 3D ellipsoid renders, match-3812 aggregate/prior/per-pixel-histogram figures), `interactive_pass_ellipsoid.html` |

Hard-coded scenario constants (preserved from the original): passer/receiver
coordinates of the three scenarios; bulge grids ±0.05-0.60; tracked event
ids `6497474, 6497478, 6497488`; jersey map de Jong 21, Blind 17, Aké 4.

### 1.7 Pitch control (straight vs curve-based) and OBSO

| | |
|---|---|
| **Input** | `data/pff_match_data/{3812,3814,10517}.json` + `match_<id>_{players,ball}.csv` |
| **Code** | `pitch_control/LaurieOnTracking/Metrica_IO.py::load_new_data` (single entry point), `Metrica_PitchControl.py` (models), `Metrica_Velocities.py`, `Metrica_EPV.py` + EPV/Transition grids; project scripts in `pitch_control/` |
| **Output** | pitch-control heatmaps (GUI figures) and `results/pitch_control/*.csv` (committed) |

Headless, reproducible entry points (run from `pitch_control/`):

```bash
python compute_all_events_error_summary.py            # -> results/pitch_control/all_events_{error,timing}_summary.csv
python batch_reconstruction_distance_errors.py --game-id 3812
python reverse_pitch_control_solver.py --evaluate-all-events   # reverse_constraint1_* CSVs
python tune_linearized_params.py                      # linearized_sigma_features_tuning.csv (see caveat 3.4)
```

Model families in `Metrica_PitchControl.py`:

1. **Baseline Spearman 2018** (`generate_pitch_control_for_event`): 50x32
   grid (default), time-to-intercept sigmoid (`tti_sigma` 0.45 s, reaction
   0.7 s, vmax 5 m/s), ball travel straight-line at 15 m/s, `kappa_def` 1,
   integration dt 0.04 s.
2. **Curve-based** (`generate_pitch_control_for_event_curve_based`): the
   baseline model generalized to curved ball travel — 5 discretized
   curvature classes (from `default_model_params()['trajectory_classes']`,
   derived empirically per §1.4) x bend direction ±1, each drawn as a
   quadratic Bézier whose control-point offset is `h = 3.0 · curvature ·
   chord_length` (inverse of `C = area / chord²` for a circular-segment
   profile), per-cell interception re-evaluated along the trajectory.
   This is how **curvature is incorporated into pitch control**: every
   grid cell's control probability averages over curved ball paths
   instead of assuming a straight pass line.
3. Linearized matrix-form / Gaussian / exponential approximation models
   (research comparisons; see caveats 3.4/3.6).

**OBSO** (`obso_player.py`, used by `obso_pitch_control_comparison_gui.py`):
`OBSO(cell) = PPCF(cell) × Transition(cell→shot) × EPV(cell)`, with
`Transition_gauss.csv` (100x64) and `EPV_grid.csv` (32x50) loaded from
`pitch_control/LaurieOnTracking/`.

Interactive GUIs (Tkinter) for exploration:
`run_pitch_control.py`, `pitch_control_comparison_gui.py` (straight vs
curve-based PPCF + squared error), `obso_pitch_control_comparison_gui.py`
(PPCF + OBSO, multi-match), `run_pitch_control_reconstructed_compare.py`,
`run_pitch_control_formation_gui.py`, `run_pitch_control_nudge_gui.py`,
`reverse_pitch_control_gui.py`.

### 1.8 Project report

`docs/CurvedPass_v1.pdf` — the written project report (methodology,
motivation, and results narrative).

---

## 2. Recommended execution order

1. `pip install -r requirements.txt`
2. Place PFF data → `python scripts/verify_data_setup.py`
3. `python scripts/build_worldcup_passes_csv.py`
4. `python scripts/curvature_heatmap_breakdown.py`
5. `python scripts/curvature_histograms.py`
6. `jupyter lab notebooks/pass_curvature_analysis.ipynb` (run top-to-bottom)
7. `jupyter lab notebooks/das_curved_pass_study_3D.ipynb` (needs `accessible-space` + `data/pff_match_data/3812.{json,jsonl}`)
8. `python pitch_control/compute_all_events_error_summary.py` (and/or the GUIs)

Steps 4-6 are independent of 7-8; step 5's statistics feed the priors used
in step 7 (documented in §1.6).

## 3. Known caveats (documented, NOT silently changed)

These quirks exist in the original methodology. They are preserved as-is
for fidelity; fix them only if you consciously revise the methodology.

1. **25 fps assumption.** `Metrica_IO.load_new_data`/`read_json_event_data`
   convert event times to frame numbers as `int(t * 25)`, while the PFF
   metadata reports `fps: 29.97`. Frame indices are used consistently
   within the pipeline, but they are not true video frames.
2. **Normalization offset.** The loader maps meters to Metrica-normalized
   units via `(x + 53) / 106` (and `(y + 34) / 68`), assuming a 106 m pitch;
   the actual pitch is 105 x 68 m (meta JSON). Downstream scripts convert
   back with `field_dimen=(106, 68)`. Net effect: a small systematic offset
   (< 1 m), consistent across all pitch-control results.
3. **Sparse tracking.** The per-event CSVs contain only possession-event
   frames (not continuous 25 Hz tracking); pitch control is computed per
   event from those frames.
4. **`tune_linearized_params.py` is broken against the current library:**
   it passes `n_features=` to
   `generate_pitch_control_for_event_linearized_matrixform`, whose current
   signature ignores it (and hard-codes `FastGaussianGrid(sigma=38)`). The
   committed tuning CSV predates this drift.
5. **Match coverage gaps:** `match_3813_*.csv` exist but `3813.json` is
   missing (match unloadable); `3812_meta.json` is missing (team-name
   normalization falls back to hard-coded Netherlands→Home / Senegal→Away
   inside `read_json_event_data`, which happens to be correct for 3812).
   Use `obso_pitch_control_comparison_gui.normalize_event_teams` for other
   matches.
6. **Historical model iterations (superseded, kept for the record):**
   `visualization_1/2/3` notebooks of the DAS study (an "always-straight"
   xC lookup bug, then an apex-risk proxy) were superseded by the
   CORRECTED notebooks and are **not** included; the exponential
   approximation model's `NonLinearGaussianGrid` uses a positive exponent
   (`np.exp(r_sq)`), which is almost certainly a bug — it is only used for
   comparison columns, never as a final result.
7. **`calculate_obso.py` is not runnable as shipped** (references a
   `third_party` module and J-League sample data from the original OBSO
   project). The working OBSO pipeline is
   `obso_player.calc_obso` + `obso_pitch_control_comparison_gui.py`, so the
   broken driver was excluded from this repository.

## 4. What was reorganized (and what was not)

- Only **paths** were changed (Colab mounts, sandbox output dirs,
  machine-specific data locations → repository-relative paths and
  `src/curved_passes/config.py`). No metric, model parameter, threshold,
  or dataset was altered.
- Duplicated obsolete artifacts excluded: `histogram notebook.ipynb`
  (workflow preserved as `scripts/curvature_histograms.py`), the three
  superseded visualization notebooks, `run_reconstructed_compare_log.txt`
  (old-machine console log), 0-byte `metrica_comparison_fixed.py`,
  `tmp_inspect_csv.py`, `.vs`/`__pycache__`/git worktree clutter.
- Original notebooks `fifa2022_worldcup_passes_colab.ipynb` and
  `worldcup_heatmap_breakdown.py` remain available in their converted,
  path-corrected form (§1.1, §1.3).
