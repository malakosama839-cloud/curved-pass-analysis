import os
import time
import argparse
import numpy as np
import pandas as pd


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MATCH_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "pff_match_data")

from LaurieOnTracking import Metrica_IO as mio
from LaurieOnTracking import Metrica_Velocities as mvel
from LaurieOnTracking import Metrica_PitchControl as mpc


def to_metric_coordinates(data, field_dimen=(106.0, 68.0)):
    x_columns = [c for c in data.columns if c[-2:] == "_x" and c[:4] in ["Home", "Away"]]
    y_columns = [c for c in data.columns if c[-2:] == "_y" and c[:4] in ["Home", "Away"]]
    data[x_columns] = (data[x_columns] - 0.5) * field_dimen[0]
    data[y_columns] = (data[y_columns] - 0.5) * field_dimen[1]
    return data


def find_goalkeepers(tracking_home, tracking_away):
    home_players = sorted(
        set([c.split("_")[1] for c in tracking_home.columns if c.startswith("Home_") and c.endswith("_x")])
    )
    away_players = sorted(
        set([c.split("_")[1] for c in tracking_away.columns if c.startswith("Away_") and c.endswith("_x")])
    )

    home_avg_x = {p: tracking_home[f"Home_{p}_x"].mean() for p in home_players}
    away_avg_x = {p: tracking_away[f"Away_{p}_x"].mean() for p in away_players}

    home_gk = min(home_avg_x, key=home_avg_x.get)
    away_gk = max(away_avg_x, key=away_avg_x.get)

    return home_gk, away_gk


def compute_error_metrics(baseline, model, eps=1e-8, baseline_threshold=0.01):
    """
    Robust error metrics for probability surfaces.

    Notes:
    - Pointwise mean relative error can explode where baseline is near 0.
    - Weighted relative error (WAPE-style) is more stable:
        sum(|b - m|) / sum(|b|)
      computed only over significant baseline cells.
    """
    diff = baseline - model
    abs_diff = np.abs(diff)

    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    significant_mask = baseline > baseline_threshold
    sig_count = int(np.sum(significant_mask))

    if sig_count > 0:
        num = float(np.sum(abs_diff[significant_mask]))
        den = float(np.sum(np.abs(baseline[significant_mask])) + eps)
        weighted_rel_pct = (num / den) * 100.0

        # Keep a pointwise metric too, but with denominator clamped for stability.
        pointwise_rel = abs_diff[significant_mask] / np.maximum(np.abs(baseline[significant_mask]), baseline_threshold)
        mean_pointwise_rel_pct = float(np.mean(pointwise_rel) * 100.0)
    else:
        weighted_rel_pct = np.nan
        mean_pointwise_rel_pct = np.nan

    return {
        "mae": mae,
        "rmse": rmse,
        "weighted_rel_pct": weighted_rel_pct,
        "mean_pointwise_rel_pct": mean_pointwise_rel_pct,
        "sig_count": sig_count,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute average absolute/relative error for all pitch-control methods vs baseline."
    )
    parser.add_argument("--n-grid", type=int, default=25, help="Grid resolution in x direction (default: 25).")
    parser.add_argument("--max-events", type=int, default=0, help="Limit number of events (0 means all).")
    parser.add_argument("--event-type", type=str, default="", help="Optional event type filter, e.g. PASS.")
    parser.add_argument("--progress-every", type=int, default=10, help="Print progress every N processed events.")
    return parser.parse_args()


def main():
    args = parse_args()
    game_id = 3812
    field_dimen = (106.0, 68.0)

    print("=" * 70)
    print("ALL-EVENT ERROR SUMMARY FOR PITCH CONTROL MODELS")
    print("=" * 70)

    print("Loading tracking/event data...")
    tracking_home, tracking_away, events = mio.load_new_data(MATCH_DATA_PATH, game_id)

    print("Converting coordinates and calculating velocities...")
    tracking_home = to_metric_coordinates(tracking_home, field_dimen)
    tracking_away = to_metric_coordinates(tracking_away, field_dimen)
    events["Start X"] = (events["Start X"] - 0.5) * field_dimen[0]
    events["Start Y"] = (events["Start Y"] - 0.5) * field_dimen[1]

    tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True)
    tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)

    home_gk, away_gk = find_goalkeepers(tracking_home, tracking_away)
    GK_numbers = (home_gk, away_gk)
    params = mpc.default_model_params()

    model_specs = [
        ("Matrix Distance Based Probability", mpc.generate_pitch_control_for_event_linearized_matrixform, {"offsides": False}),
        ("Gaussian Distance Based Probability", mpc.generate_gaussian_pitch_control_for_event, {"offsides": False}),
        (
            "Distance Based Probability with exponential",
            mpc.generate_pitch_control_for_event_with_exponential,
            {"offsides": False, "sigma": 7},
        ),
    ]

    errors_by_model = {
        name: {
            "mae": [],
            "rmse": [],
            "weighted_rel_pct": [],
            "mean_pointwise_rel_pct": [],
            "sig_counts": [],
            "num_events": 0,
        }
        for name, _, _ in model_specs
    }
    timings = {
        "Baseline (Original)": {"durations": [], "num_events": 0},
        **{name: {"durations": [], "num_events": 0} for name, _, _ in model_specs},
    }

    selected_events = events
    if args.event_type:
        selected_events = events[events["Type"].astype(str).str.upper() == args.event_type.upper()]

    if args.max_events and args.max_events > 0:
        selected_events = selected_events.iloc[: args.max_events]

    total_events = len(selected_events.index)
    if total_events == 0:
        print("No events matched the selected filters.")
        return

    print(f"Using n_grid_cells_x={args.n_grid}")
    print(f"Events selected: {total_events}")

    processed_events = 0
    skipped_events = 0
    start_time = time.time()

    try:
        for event_id in selected_events.index:
            try:
                t0 = time.perf_counter()
                baseline, _, _ = mpc.generate_pitch_control_for_event(
                    event_id,
                    events,
                    tracking_home,
                    tracking_away,
                    params,
                    GK_numbers,
                    field_dimen=field_dimen,
                    n_grid_cells_x=args.n_grid,
                    offsides=False,
                )
                timings["Baseline (Original)"]["durations"].append(time.perf_counter() - t0)
                timings["Baseline (Original)"]["num_events"] += 1

                for model_name, model_fn, extra_kwargs in model_specs:
                    t1 = time.perf_counter()
                    comparison, _, _ = model_fn(
                        event_id,
                        events,
                        tracking_home,
                        tracking_away,
                        params,
                        GK_numbers,
                        field_dimen=field_dimen,
                        n_grid_cells_x=args.n_grid,
                        **extra_kwargs,
                    )
                    timings[model_name]["durations"].append(time.perf_counter() - t1)
                    timings[model_name]["num_events"] += 1

                    metrics = compute_error_metrics(baseline, comparison)
                    errors_by_model[model_name]["mae"].append(metrics["mae"])
                    errors_by_model[model_name]["rmse"].append(metrics["rmse"])
                    if not np.isnan(metrics["weighted_rel_pct"]):
                        errors_by_model[model_name]["weighted_rel_pct"].append(metrics["weighted_rel_pct"])
                    if not np.isnan(metrics["mean_pointwise_rel_pct"]):
                        errors_by_model[model_name]["mean_pointwise_rel_pct"].append(metrics["mean_pointwise_rel_pct"])
                    errors_by_model[model_name]["sig_counts"].append(metrics["sig_count"])
                    errors_by_model[model_name]["num_events"] += 1

                processed_events += 1

                if args.progress_every > 0 and processed_events % args.progress_every == 0:
                    elapsed = time.time() - start_time
                    avg_sec = elapsed / max(processed_events, 1)
                    remaining = total_events - processed_events
                    eta_sec = int(remaining * avg_sec)
                    print(
                        f"Processed {processed_events}/{total_events} events | "
                        f"elapsed={elapsed:.1f}s | eta~{eta_sec}s"
                    )

            except Exception as exc:
                skipped_events += 1
                print(f"Skipping event {event_id}: {exc}")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Computing summary from processed events...")

    rows = []
    for model_name, stats in errors_by_model.items():
        avg_mae = float(np.mean(stats["mae"])) if stats["mae"] else np.nan
        avg_rmse = float(np.mean(stats["rmse"])) if stats["rmse"] else np.nan
        avg_weighted_rel_pct = float(np.mean(stats["weighted_rel_pct"])) if stats["weighted_rel_pct"] else np.nan
        avg_mean_pointwise_rel_pct = (
            float(np.mean(stats["mean_pointwise_rel_pct"])) if stats["mean_pointwise_rel_pct"] else np.nan
        )
        avg_sig_cells = float(np.mean(stats["sig_counts"])) if stats["sig_counts"] else np.nan

        rows.append(
            {
                "model": model_name,
                "events_used": stats["num_events"],
                "avg_mae": avg_mae,
                "avg_rmse": avg_rmse,
                "avg_weighted_relative_error_percent": avg_weighted_rel_pct,
                "avg_pointwise_relative_error_percent": avg_mean_pointwise_rel_pct,
                "avg_significant_cells_per_event": avg_sig_cells,
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("model").reset_index(drop=True)

    timing_rows = []
    baseline_avg_sec = np.nan
    if timings["Baseline (Original)"]["durations"]:
        baseline_avg_sec = float(np.mean(timings["Baseline (Original)"]["durations"]))

    for method_name, stats in timings.items():
        avg_seconds = float(np.mean(stats["durations"])) if stats["durations"] else np.nan
        ratio_vs_baseline = (
            float(avg_seconds / baseline_avg_sec)
            if (not np.isnan(avg_seconds) and not np.isnan(baseline_avg_sec) and baseline_avg_sec > 0)
            else np.nan
        )

        timing_rows.append(
            {
                "method": method_name,
                "events_timed": stats["num_events"],
                "avg_seconds_per_event": avg_seconds,
                "ratio_vs_baseline": ratio_vs_baseline,
            }
        )

    timing_df = pd.DataFrame(timing_rows).sort_values("method").reset_index(drop=True)

    print("\n" + "-" * 70)
    print(f"Total events: {total_events}")
    print(f"Processed events: {processed_events}")
    print(f"Skipped events: {skipped_events}")
    print("-" * 70)
    print("\nAverage error summary (baseline vs each GUI method):")
    print("Relative metrics: weighted is the primary robust metric; pointwise is secondary.")
    print(summary_df.to_string(index=False))
    print("\nTiming summary:")
    print(timing_df.to_string(index=False))

    output_path = os.path.join(BASE_DIR, "all_events_error_summary.csv")
    summary_df.to_csv(output_path, index=False)
    print(f"\nSaved summary to: {output_path}")
    timing_output_path = os.path.join(BASE_DIR, "all_events_timing_summary.csv")
    timing_df.to_csv(timing_output_path, index=False)
    print(f"Saved timing summary to: {timing_output_path}")


if __name__ == "__main__":
    main()
