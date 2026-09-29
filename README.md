# Curved Passes: A Normalized Pass-Curvature Metric and Its Impact on Pitch Control

Reproducible research repository for the MIT Sloan curved-pass project.

The project asks a simple question with a quantitative answer: **does the
shape of a pass matter?** Not where the ball starts or ends — those are
fully described by every event-data feed — but the *path* it takes between
passer and receiver. We develop a compact, physically interpretable
curvature metric from player-tracking data, characterize how curved passes
actually are at the FIFA World Cup 2022, and integrate curvature into
pitch-control models to test whether ignoring ball-path shape misprices
passing options.

---

## 1. Project overview

### Motivation

Event data describes a pass by two points: start and end. But a chipped
around-the-blocker pass and a straight thread-the-needle pass between the
same two points are very different football actions — the curved pass can
render an apparently blocked lane viable. Tracking data now gives the full
3D ball trajectory of every pass, so pass shape can be measured rather than
anecdotally asserted.

### Research questions

1. How curved are passes in practice, and how should curvature be measured
   so that the number is comparable across passes of different lengths?
2. Does curvature relate to pass outcome (success vs failure)?
3. Does curvature change pitch-control / expected-completion (xC) valuations
   of space — i.e., do curved passes "open" regions the straight-pass model
   calls unreachable?

### The new curvature metric

For a ball trajectory sampled at points `r_0 … r_{n−1}` (meters):

1. **Chord** — the straight passer-to-receiver distance
   `D = ‖r_{n−1} − r_0‖`
2. **Perpendicular deviation** of each point from the chord
   `d_i = ‖(r_i − r_0) × (r_{n−1} − r_0)‖ / D`
3. **Deviation area** — trapezoidal integration of the deviation along the
   actual 3D arc length: `A_dev = Σ 0.5 (d_i + d_{i+1}) · ‖r_{i+1} − r_i‖`
4. **Normalized curvature** — `C = A_dev / D²` (dimensionless; 0 = straight)

Dividing by `D²` makes the metric scale-invariant: a 40 m pass bending 5 m
out of line scores like a 10 m pass bending 1.25 m. The area is the
"region swept" between the true path and the straight line; the chord is the
total passer-to-receiver distance. The canonical implementation is
[`src/curved_passes/curvature.py`](src/curved_passes/curvature.py).

### Methodology at a glance

- **Passing data** — PFF FC World Cup 2022 event + player/ball tracking
  (restricted; see below) is flattened into a per-pass table with each
  pass's full ball trajectory embedded (`worldcup2022_passes.csv`, ~66k passes).
- **Curvature** — the metric above is computed for every pass; distributions
  are compared between successful and failed passes (thresholds 0.01 / 0.02 /
  0.06 used across the analyses).
- **Spatial analysis** — P(pass is straight) by starting position on a
  12 × 8 pitch grid (the curvature heat map).
- **Pitch control** — the Spearman (2018) pitch-control model is generalized
  so ball travel follows curved (Bézier) paths; per-cell control
  probabilities average over discretized curvature classes whose values come
  from the empirical curvature histogram.
- **DAS xC study** — defender-adjusted space: expected completion of curved
  passes through a blocked lane using the accessible-space model (including
  a 3D height-gated variant), with curve-play priors derived from the
  empirical curvature rates (41.12% successful / 55.70% failed at C > 0.06).
- **OBSO** — on-ball skill space values (`PPCF × Transition × EPV`) compared
  between straight and curve-based pitch control.

---

## 2. Data

| Dataset | In repo? | Notes |
|---|---|---|
| **PFF FC WC 2022** event JSONs, tracking JSONL, player/ball CSVs | **No — restricted** | Must be obtained under license from PFF; see [`docs/data_access.md`](docs/data_access.md). Goes in `data/pff_match_data/` (gitignored). |
| `data/scenarios/best_pass_scenario.json` | Yes | Passer/receiver/defender coordinates of the blocked-lane DAS scenario (positions only) |
| `data/derived/worldcup2022_passes.csv` | **Regenerate locally** | ~150 MB tournament pass table built from PFF data by `scripts/build_worldcup_passes_csv.py`; gitignored (derived from licensed data) |
| `LaurieOnTracking` EPV/Transition grids | Yes | Upstream OBSO assets (`pitch_control/LaurieOnTracking/`) |

**The repository contains no PFF data and no per-pass PFF-derived table.**
`.gitignore` blocks `data/pff_match_data/*`, `data/derived/*`, and raw
tracking/event patterns anywhere in the tree.

## 3. Installation

```bash
git clone <this-repository>
cd curved-pass-analysis

python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

- Python **3.12** recommended (3.10+ should work).
- The DAS notebooks additionally need the
  [`accessible-space`](https://github.com/jonas-bischofberger/accessible-space)
  library — see the note in `requirements.txt` (its xC code uses private
  constants of that library, so pin/record the version you install).

## 4. Data setup

1. Obtain the PFF FC World Cup 2022 data under your own license — full
   step-by-step tutorial in **[docs/data_access.md](docs/data_access.md)**.
2. Place the files in `data/pff_match_data/`:

```text
data/pff_match_data/
├── 3812.json                  # possession events (Netherlands - Senegal)
├── 3812.jsonl                 # raw tracking
├── 3812_meta.json             # optional metadata
├── match_3812_players.csv     # per-event player positions
├── match_3812_ball.csv        # per-event ball positions
└── ...                        # 3814, 10517, ... same pattern
```

3. Verify:

```bash
python scripts/verify_data_setup.py
```

(Optional) use a data folder outside the repo by setting the environment
variable `CURVED_PASSES_PFF_DATA=/your/path` — no code edits needed.

## 5. Running the project

In order (each step's provenance is documented in
[`docs/reproducibility.md`](docs/reproducibility.md)):

```text
1. Set up the environment                     pip install -r requirements.txt
2. Obtain authorized data and place it        see docs/data_access.md
3. Verify the data installation               python scripts/verify_data_setup.py
4. Build the WC-2022 pass table               python scripts/build_worldcup_passes_csv.py
5. Curvature heat map                         python scripts/curvature_heatmap_breakdown.py
6. Curvature histograms + statistics          python scripts/curvature_histograms.py
7. Interactive curvature analysis             notebooks/pass_curvature_analysis.ipynb
8. DAS curved-pass xC study                   notebooks/das_curved_pass_study_3D.ipynb
9. Pitch-control error summary (headless)     python pitch_control/compute_all_events_error_summary.py
10. Pitch-control heat maps / OBSO GUIs       python pitch_control/pitch_control_comparison_gui.py
                                              python pitch_control/obso_pitch_control_comparison_gui.py
```

(Windows note: GUI scripts use Tkinter; run with a desktop Python. Scripts
in step 9-10 are run from the `pitch_control/` directory.)

### Expected outputs

| Result | Produced by | Output |
|---|---|---|
| Tournament pass table | `scripts/build_worldcup_passes_csv.py` | `data/derived/worldcup2022_passes.csv` |
| Curvature heat map | `scripts/curvature_heatmap_breakdown.py` | `figures/curvature_heatmap/worldcup2022_heatmap_breakdown.png` (committed copy) |
| Curvature histograms & curved-pass stats | `scripts/curvature_histograms.py` | `figures/curvature_histogram/*.png`, `results/curvature/*.csv` |
| DAS scenario sweeps / match-3812 heatmaps / per-pixel histograms | `notebooks/das_curved_pass_study_3D.ipynb` | `figures/das_study/*.png` (committed copies) |
| Pitch-control model errors & timings | `pitch_control/compute_all_events_error_summary.py` | `results/pitch_control/all_events_*.csv` (committed copies) |
| Straight vs curved PPCF / OBSO heat maps | GUIs in `pitch_control/` | interactive windows |

Committed figures/regression CSVs let you check your run against the
reported results without waiting on the licensed data.

## 6. Repository structure

```text
curved-pass-analysis/
├── README.md                        this file
├── requirements.txt
├── LICENSE                          MIT for the code; PFF data is NOT included
├── .gitignore                       blocks all restricted/derived data
├── data/
│   ├── README.md                    what goes where
│   ├── pff_match_data/              (gitignored) you place licensed PFF data here
│   ├── derived/                     (gitignored) generated pass tables
│   └── scenarios/best_pass_scenario.json
├── docs/
│   ├── data_access.md               PFF data acquisition & installation tutorial
│   ├── reproducibility.md           Input → Code → Output map for every result
│   └── CurvedPass_v1.pdf            project report
├── notebooks/
│   ├── pass_curvature_analysis.ipynb        per-pass curvature analysis
│   ├── das_curved_pass_study_3D.ipynb       canonical DAS xC study (2D + 3D + match 3812)
│   └── das_curved_pass_study_2D.ipynb       2D standalone variant
├── scripts/
│   ├── build_worldcup_passes_csv.py         PFF raw → pass table
│   ├── curvature_heatmap_breakdown.py       curvature heat map
│   ├── curvature_histograms.py              curvature histograms & stats
│   └── verify_data_setup.py                 data installation check
├── src/curved_passes/
│   ├── curvature.py                 CANONICAL curvature metric (C = A_dev / D²)
│   └── config.py                    path configuration (env-var overridable)
├── pitch_control/                   LaurieOnTracking fork + project scripts + README
├── figures/                         committed result figures
└── results/                         committed result CSVs
```

## 7. Reproducibility notes

- Every major result has a documented **Input → Code → Processing → Output**
  chain in [`docs/reproducibility.md`](docs/reproducibility.md), including
  parameters, runtime, and known caveats of the original methodology
  (25 fps vs 29.97, normalization offset, superseded model iterations) —
  the research methodology itself is unchanged.
- All Colab mounts and machine-specific paths were replaced with
  repository-relative paths / `src/curved_passes/config.py` (env-var
  overridable).
- Notebooks are ordered top-to-bottom runnable after step 4 above.

## 8. License & attribution

- Repository code: MIT (see `LICENSE`).
- `pitch_control/LaurieOnTracking/` is a fork of
  [Friends-of-Tracking-Data-FoTD/LaurieOnTracking](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking)
  (MIT, license retained).
- The DAS study depends on
  [accessible-space](https://github.com/jonas-bischofberger/accessible-space).
- **PFF FC match data is licensed material; it is not included, and it may
  not be redistributed through this repository.**
