"""World Cup 2022 - P(pass is straight) by starting position, with per-cell breakdown.

Computes the normalized 3D curvature (see src/curved_passes/curvature.py) for
every pass in the tournament pass table, then renders the "curvature heat
map": a 12 x 8 grid over the pitch where color = P(pass is straight) and
each cell is annotated with n / Straight / Curved.

Input
-----
data/derived/worldcup2022_passes.csv  (built by scripts/build_worldcup_passes_csv.py
from authorized PFF tracking + event data; see docs/data_access.md)

Output
------
figures/curvature_heatmap/worldcup2022_heatmap_breakdown.png

Requirements: numpy, pandas, matplotlib
Usage:
    python scripts/curvature_heatmap_breakdown.py
    python scripts/curvature_heatmap_breakdown.py --csv path/to/pass_table.csv
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

from curved_passes.config import WORLDCUP_PASSES_CSV, FIGURES_DIR
from curved_passes.curvature import row_curvature

# ============================================================
# Configuration (parameters used for the reported figure)
# ============================================================
STRAIGHT_THRESHOLD = 0.02   # a pass is "straight" when curvature <= this
CURVED_THRESHOLD = 0.06     # reported in the text: curved when curvature > this
X_BINS, Y_BINS = 12, 8
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
X_MIN, X_MAX = -PITCH_LENGTH / 2, PITCH_LENGTH / 2
Y_MIN, Y_MAX = -PITCH_WIDTH / 2, PITCH_WIDTH / 2
PROB_VMIN, PROB_VMAX = 0.07, 0.29   # fixed color scale (matches the original figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=WORLDCUP_PASSES_CSV,
                        help="Pass table CSV (default: data/derived/worldcup2022_passes.csv)")
    parser.add_argument("--output", type=Path,
                        default=FIGURES_DIR / "curvature_heatmap" / "worldcup2022_heatmap_breakdown.png",
                        help="Output PNG path")
    args = parser.parse_args()

    # ============================================================
    # Load raw passes
    # ============================================================
    df = pd.read_csv(args.csv)
    print(f"Loaded {args.csv}")
    print(f"Raw passes: {len(df)}")

    # ============================================================
    # Curvature: area between the 3D ball path and the straight
    # start-to-end chord, divided by squared chord length.
    # (Canonical metric: src/curved_passes/curvature.py)
    # ============================================================
    print("Computing 3D curvature (this takes about 20 s on 66k passes)...")
    df["curvature_3d_normalized"] = df.apply(row_curvature, axis=1)
    print(f"Passes with measurable curvature: {df['curvature_3d_normalized'].notna().sum()}")

    # ============================================================
    # Starting position of each pass = first coordinate of ball_x / ball_y
    # ============================================================
    def first_coordinate(value):
        if value is None or pd.isna(value):
            return np.nan
        text = str(value).strip().strip("[]")
        if not text or text.lower() in {"nan", "none", "null"}:
            return np.nan
        try:
            return float(text.split(",")[0].strip())
        except ValueError:
            return np.nan

    grid_df = pd.DataFrame({
        "start_x": df["ball_x"].map(first_coordinate),
        "start_y": df["ball_y"].map(first_coordinate),
        "curvature": df["curvature_3d_normalized"],
    })
    grid_df = grid_df.dropna()
    grid_df = grid_df[
        (grid_df["start_x"] >= X_MIN) & (grid_df["start_x"] <= X_MAX)
        & (grid_df["start_y"] >= Y_MIN) & (grid_df["start_y"] <= Y_MAX)
    ]
    grid_df["is_straight"] = (grid_df["curvature"] <= STRAIGHT_THRESHOLD).astype(float)
    print(f"Passes used: {len(grid_df)}")

    # ============================================================
    # Aggregate per grid cell
    # ============================================================
    x_edges = np.linspace(X_MIN, X_MAX, X_BINS + 1)
    y_edges = np.linspace(Y_MIN, Y_MAX, Y_BINS + 1)
    counts, _, _ = np.histogram2d(grid_df["start_x"], grid_df["start_y"], bins=[x_edges, y_edges])
    straight_counts, _, _ = np.histogram2d(
        grid_df["start_x"], grid_df["start_y"],
        bins=[x_edges, y_edges], weights=grid_df["is_straight"],
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        prob_grid = straight_counts / counts
    print(f"Grid cells with data: {np.isfinite(prob_grid).sum()} / {prob_grid.size}")

    # ============================================================
    # Plot
    # ============================================================
    fig, ax = plt.subplots(figsize=(20, 12.5))
    mesh = ax.pcolormesh(
        x_edges, y_edges, prob_grid.T,
        cmap="RdYlGn", vmin=PROB_VMIN, vmax=PROB_VMAX,
        edgecolors="white", linewidth=0.8, alpha=0.85,
    )
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.8)
    cbar.set_label("P(pass is straight)")

    # ---- pitch markings ----
    line_color, line_width = "black", 1.5
    ax.plot([0, 0], [Y_MIN, Y_MAX], color=line_color, linewidth=line_width)
    ax.add_patch(Circle((0, 0), 9.15, fill=False, color=line_color, linewidth=line_width))
    ax.add_patch(Rectangle((X_MIN, -20.16), 16.5, 40.32, fill=False,
                           color=line_color, linewidth=line_width))
    ax.add_patch(Rectangle((X_MAX - 16.5, -20.16), 16.5, 40.32, fill=False,
                           color=line_color, linewidth=line_width))
    for pts in ([(X_MIN, X_MAX), (Y_MIN, Y_MIN)], [(X_MIN, X_MAX), (Y_MAX, Y_MAX)],
                [(X_MIN, X_MIN), (Y_MIN, Y_MAX)], [(X_MAX, X_MAX), (Y_MIN, Y_MAX)]):
        ax.plot(pts[0], pts[1], color=line_color, linewidth=line_width)

    # ---- cell text: n / Straight / Curved (percentages always sum to 100) ----
    for i in range(X_BINS):
        for j in range(Y_BINS):
            n = counts[i, j]
            s = straight_counts[i, j]
            if n <= 0:
                continue
            straight_pct = int(round(100 * s / n))
            curved_pct = 100 - straight_pct
            cell_text = (
                f"n = {int(round(n))}\n"
                f"Straight = {int(round(s))} ({straight_pct}%)\n"
                f"Curved = {int(round(n - s))} ({curved_pct}%)"
            )
            ax.text(
                (x_edges[i] + x_edges[i + 1]) / 2,
                (y_edges[j] + y_edges[j + 1]) / 2,
                cell_text, ha="center", va="center",
                fontsize=7, fontweight="bold", color="black", linespacing=1.4,
            )

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel("Ball x at pass start (m)")
    ax.set_ylabel("Ball y at pass start (m)")
    ax.set_title(
        "Probability a Pass Is Straight by Starting Position\n"
        f"(straight = normalized 3D curvature <= {STRAIGHT_THRESHOLD})"
    )

    # ---- legend note ----
    fig.text(
        0.5, 0.01,
        "Color = probability that a pass is straight (colorbar: P(pass is straight))    |    "
        "Cell text = n (number of observations) and the straight/curved outcome breakdown",
        ha="center", va="bottom", fontsize=10, fontstyle="italic", color="black",
    )
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
