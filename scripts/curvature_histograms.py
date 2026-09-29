"""Curvature histograms: how normalized 3D curvature is distributed across passes.

Reproduces the histogram/distribution workflow of the curvature study
(originally "histogram notebook.ipynb"):
  1. compute the canonical curvature metric for every pass in the table;
  2. histogram curved passes by outcome (successful vs failed):
     equal-width bins (threshold 0.02) and 20 quantile bins (threshold 0.01,
     with cumulative percentage line and probability density);
  3. curved-pass statistics at the reported threshold 0.06
     (overall / successful / failed curved percentages);
  4. N-quantile "trajectory classes" (default 5), the empirical curvature
     bins used as ``trajectory_classes`` in the curve-based pitch-control
     model (pitch_control/LaurieOnTracking/Metrica_PitchControl.py).

Input
-----
data/derived/worldcup2022_passes.csv (built by scripts/build_worldcup_passes_csv.py)

Outputs
-------
figures/curvature_histogram/histograms_equal_bins.png          (threshold 0.02)
figures/curvature_histogram/histograms_quantile_bins.png       (threshold 0.01)
figures/curvature_histogram/histograms_quantile_density.png    (threshold 0.01)
results/curvature/curved_pass_statistics.csv                   (threshold 0.06)
results/curvature/trajectory_classes.csv                       (5 quantile bins)

Requirements: numpy, pandas, matplotlib
Usage:
    python scripts/curvature_histograms.py
    python scripts/curvature_histograms.py --csv path/to/pass_table.csv
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from curved_passes.config import WORLDCUP_PASSES_CSV, DERIVED_DATA_DIR, FIGURES_DIR, RESULTS_DIR
from curved_passes.curvature import row_curvature

THRESHOLD_EQUAL_BINS = 0.02   # equal-width histogram threshold
THRESHOLD_QUANTILE = 0.01     # quantile histogram threshold
THRESHOLD_STATS = 0.06        # threshold for the reported curved-pass statistics
N_QUANTILE_BINS = 20
N_TRAJECTORY_CLASSES = 5      # quantile bins -> curve-based pitch-control classes


def compute_curvature(df):
    print("Computing 3D curvature for every pass ...")
    df["curvature_3d_normalized"] = df.apply(row_curvature, axis=1)
    print(f"Passes with measurable curvature: {df['curvature_3d_normalized'].notna().sum()}")
    return df


def curved_values(df, threshold, success):
    return df.loc[
        (df["pass_success"] == success)
        & (df["curvature_3d_normalized"] > threshold),
        "curvature_3d_normalized",
    ].dropna()


def plot_equal_width_histograms(df, threshold, out_path):
    """Cell: shared 31-bin equal-width histograms, successful vs failed."""
    successful = curved_values(df, threshold, True)
    failed = curved_values(df, threshold, False)
    all_values = pd.concat([successful, failed])
    bins = np.linspace(threshold, all_values.max(), 31)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    axes[0].hist(successful, bins=bins, alpha=0.7)
    axes[0].set_xlabel("Normalized 3D Curvature")
    axes[0].set_ylabel("Number of Passes")
    axes[0].set_title(f"Successful Curved Passes (n={len(successful)})")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].hist(failed, bins=bins, alpha=0.7)
    axes[1].set_xlabel("Normalized 3D Curvature")
    axes[1].set_title(f"Failed Curved Passes (n={len(failed)})")
    axes[1].grid(axis="y", alpha=0.2)

    fig.suptitle(
        f"Distribution of Curved Passes (Normalized 3D Curvature > {threshold})",
        fontsize=16,
    )
    axes[0].set_xlim(threshold, 5)
    axes[1].set_xlim(threshold, 5)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _quantile_hist_with_cumulative(ax, values, bins, title, density):
    counts, edges = np.histogram(values, bins=bins)
    ax.hist(values, bins=bins, density=density, alpha=0.7,
            label="Probability Density" if density else "Frequency")
    ax.set_xlabel("Normalized 3D Curvature")
    ax.set_ylabel("Probability Density" if density else "Frequency")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)

    cumulative_percentage = np.cumsum(counts) / counts.sum() * 100
    ax_cumulative = ax.twinx()
    ax_cumulative.plot(edges[1:], cumulative_percentage, marker="s",
                       linewidth=2, label="Cumulative %")
    ax_cumulative.set_ylabel("Cumulative Percentage")
    ax_cumulative.set_ylim(0, 100)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_cumulative.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")


def plot_quantile_histograms(df, threshold, n_bins, density, out_path):
    """Shared quantile bins from all curved passes; density flag per original cell."""
    successful = curved_values(df, threshold, True)
    failed = curved_values(df, threshold, False)
    all_values = pd.concat([successful, failed])

    quantile_edges = np.quantile(all_values, np.linspace(0, 1, n_bins + 1))
    quantile_edges = np.unique(quantile_edges)
    if len(quantile_edges) < 2:
        raise ValueError("Not enough unique curvature values to create quantile bins.")

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    _quantile_hist_with_cumulative(
        axes[0], successful, quantile_edges,
        f"Successful Curved Passes (n={len(successful)})", density)
    _quantile_hist_with_cumulative(
        axes[1], failed, quantile_edges,
        f"Failed Curved Passes (n={len(failed)})", density)
    for ax in axes:
        ax.set_xlim(threshold, 1)
    fig.suptitle(
        f"Quantile Histogram with Cumulative Percentage "
        f"(Normalized 3D Curvature > {threshold})",
        fontsize=16,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    return quantile_edges


def curved_pass_statistics(df, threshold, out_csv):
    """Threshold statistics (the source of the 41.12% / 55.70% prior figures)."""
    curvature = df["curvature_3d_normalized"]
    curved = curvature > threshold
    successful = df["pass_success"] == True  # noqa: E712 (matches original code)
    failed = df["pass_success"] == False  # noqa: E711

    rows = [
        ("overall", int(curvature.notna().sum()), int(curved.sum())),
        ("successful", int((successful & curvature.notna()).sum()),
         int((successful & curved).sum())),
        ("failed", int((failed & curvature.notna()).sum()),
         int((failed & curved).sum())),
    ]
    out = pd.DataFrame(rows, columns=["group", "total_passes", "curved_passes"])
    out["curved_percentage"] = 100 * out["curved_passes"] / out["total_passes"]

    print("=" * 60)
    print(f"CURVED PASS STATISTICS (threshold > {threshold})")
    print("=" * 60)
    print(out.to_string(index=False))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")
    return out


def trajectory_classes(df, n_bins, out_csv):
    """Quantile-based curvature classes (bins + empirical probability).

    This is the workflow that produced the 5 discretized curvature classes
    hard-coded as ``trajectory_classes`` in the curve-based pitch-control
    model (curvature_value 0.009089 / 0.025986 / 0.049209 / 0.094816 /
    0.240787, 0.20 probability each).
    """
    curvature = df["curvature_3d_normalized"].dropna()
    curvature = curvature[curvature >= 0]

    bin_edges = curvature.quantile(np.linspace(0, 1, n_bins + 1)).values
    bin_edges = np.unique(bin_edges)
    actual_bins = len(bin_edges) - 1

    bin_counts, _ = np.histogram(curvature, bins=bin_edges)
    probabilities = bin_counts / bin_counts.sum()

    classes = [
        {
            "class_index": i + 1,
            "curvature_min": bin_edges[i],
            "curvature_max": bin_edges[i + 1],
            "probability": probabilities[i],
        }
        for i in range(actual_bins)
    ]
    out = pd.DataFrame(classes)
    print("\nTrajectory classes (quantile bins):")
    print(out.to_string(index=False))
    print(f"Total probability: {out['probability'].sum():.4f}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=WORLDCUP_PASSES_CSV,
                        help="Pass table CSV (default: data/derived/worldcup2022_passes.csv)")
    parser.add_argument("--skip-curvature", action="store_true",
                        help="Load precomputed curvature from --curvature-csv instead "
                             "of recomputing (saves ~1 min on the full table)")
    parser.add_argument("--curvature-csv", type=Path,
                        default=DERIVED_DATA_DIR / "passes_with_curvature.csv",
                        help="Where the curvature-enriched pass table is written. "
                             "Keep it under data/derived/ (gitignored): it embeds "
                             "per-pass PFF trajectories and must not be committed.")
    args = parser.parse_args()

    if args.skip_curvature and args.curvature_csv.exists():
        df = pd.read_csv(args.curvature_csv)
        print(f"Loaded precomputed curvature table {args.curvature_csv}")
    else:
        df = pd.read_csv(args.csv)
        print(f"Loaded {args.csv}")
        compute_curvature(df)
        out = args.curvature_csv
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Saved curvature table {out}")

    fig_dir = FIGURES_DIR / "curvature_histogram"
    res_dir = RESULTS_DIR / "curvature"

    plot_equal_width_histograms(df, THRESHOLD_EQUAL_BINS,
                                fig_dir / "histograms_equal_bins.png")
    plot_quantile_histograms(df, THRESHOLD_QUANTILE, N_QUANTILE_BINS,
                             density=False, out_path=fig_dir / "histograms_quantile_bins.png")
    plot_quantile_histograms(df, THRESHOLD_QUANTILE, N_QUANTILE_BINS,
                             density=True, out_path=fig_dir / "histograms_quantile_density.png")
    curved_pass_statistics(df, THRESHOLD_STATS, res_dir / "curved_pass_statistics.csv")
    trajectory_classes(df, N_TRAJECTORY_CLASSES, res_dir / "trajectory_classes.csv")


if __name__ == "__main__":
    main()
