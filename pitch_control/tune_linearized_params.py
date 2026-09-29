import os
import time
import argparse
import numpy as np
import pandas as pd

from LaurieOnTracking import Metrica_IO as mio
from LaurieOnTracking import Metrica_Velocities as mvel
from LaurieOnTracking import Metrica_PitchControl as mpc


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MATCH_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "pff_match_data")


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


def parse_int_list(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Tune sigma and n_features for linearized pitch control.")
    parser.add_argument("--game-id", type=int, default=3812)
    parser.add_argument("--n-grid", type=int, default=25)
    parser.add_argument("--max-events", type=int, default=80)
    parser.add_argument("--event-type", type=str, default="")
    parser.add_argument("--sigmas", type=str, default="15,22,28,33,38")
    parser.add_argument("--features", type=str, default="20,30,40,60,80")
    parser.add_argument("--baseline-threshold", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    sigmas = parse_float_list(args.sigmas)
    features = parse_int_list(args.features)

    field_dimen = (106.0, 68.0)

    print("Loading tracking/event data...")
    tracking_home, tracking_away, events = mio.load_new_data(MATCH_DATA_PATH, args.game_id)

    print("Converting coordinates and velocities...")
    tracking_home = to_metric_coordinates(tracking_home, field_dimen)
    tracking_away = to_metric_coordinates(tracking_away, field_dimen)
    events["Start X"] = (events["Start X"] - 0.5) * field_dimen[0]
    events["Start Y"] = (events["Start Y"] - 0.5) * field_dimen[1]

    tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True)
    tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)

    home_gk, away_gk = find_goalkeepers(tracking_home, tracking_away)
    GK_numbers = (home_gk, away_gk)
    params = mpc.default_model_params()

    selected_events = events
    if args.event_type:
        selected_events = selected_events[selected_events["Type"].astype(str).str.upper() == args.event_type.upper()]

    if args.max_events > 0 and len(selected_events) > args.max_events:
        selected_events = selected_events.sample(n=args.max_events, random_state=args.seed)

    selected_events = selected_events.sort_index()
    event_ids = list(selected_events.index)

    print(f"Selected events: {len(event_ids)}")
    print(f"Sigma candidates: {sigmas}")
    print(f"Feature candidates: {features}")

    # Precompute baselines once for fair/fast parameter comparison.
    baseline_maps = {}
    print("Precomputing baseline pitch control maps...")
    for i, event_id in enumerate(event_ids, start=1):
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
        baseline_maps[event_id] = baseline
        if i % 20 == 0:
            print(f"  Baseline precompute: {i}/{len(event_ids)}")

    rows = []
    combos = [(s, f) for s in sigmas for f in features]
    print(f"Evaluating {len(combos)} parameter combinations...")

    for idx, (sigma, n_features) in enumerate(combos, start=1):
        combo_start = time.perf_counter()

        mae_vals = []
        rmse_vals = []
        weighted_num_total = 0.0
        weighted_den_total = 0.0
        event_times = []

        for event_id in event_ids:
            baseline = baseline_maps[event_id]

            t0 = time.perf_counter()
            pred, _, _ = mpc.generate_pitch_control_for_event_linearized_matrixform(
                event_id,
                events,
                tracking_home,
                tracking_away,
                params,
                GK_numbers,
                field_dimen=field_dimen,
                n_grid_cells_x=args.n_grid,
                offsides=False,
                sigma=sigma,
                n_features=n_features,
            )
            event_times.append(time.perf_counter() - t0)

            diff = baseline - pred
            abs_diff = np.abs(diff)
            mae_vals.append(float(np.mean(abs_diff)))
            rmse_vals.append(float(np.sqrt(np.mean(diff ** 2))))

            mask = baseline > args.baseline_threshold
            if np.any(mask):
                weighted_num_total += float(np.sum(abs_diff[mask]))
                weighted_den_total += float(np.sum(np.abs(baseline[mask])))

        mean_mae = float(np.mean(mae_vals))
        mean_rmse = float(np.mean(rmse_vals))
        mean_sec = float(np.mean(event_times))
        weighted_rel_pct = float((weighted_num_total / max(weighted_den_total, 1e-12)) * 100.0)

        rows.append(
            {
                "sigma": sigma,
                "n_features": n_features,
                "events": len(event_ids),
                "mean_mae": mean_mae,
                "mean_rmse": mean_rmse,
                "weighted_rel_pct": weighted_rel_pct,
                "mean_seconds_per_event": mean_sec,
                "combo_total_seconds": float(time.perf_counter() - combo_start),
            }
        )

        print(
            f"[{idx}/{len(combos)}] sigma={sigma}, n_features={n_features} | "
            f"WRel={weighted_rel_pct:.3f}% | MAE={mean_mae:.5f} | sec/ev={mean_sec:.4f}"
        )

    result_df = pd.DataFrame(rows)

    # Normalize and make a compromise score: lower is better.
    e = result_df["weighted_rel_pct"].to_numpy()
    t = result_df["mean_seconds_per_event"].to_numpy()
    e_norm = (e - e.min()) / max(e.max() - e.min(), 1e-12)
    t_norm = (t - t.min()) / max(t.max() - t.min(), 1e-12)
    result_df["compromise_score"] = e_norm + t_norm

    by_error = result_df.sort_values(["weighted_rel_pct", "mean_seconds_per_event"]).reset_index(drop=True)
    by_time = result_df.sort_values(["mean_seconds_per_event", "weighted_rel_pct"]).reset_index(drop=True)
    by_compromise = result_df.sort_values(["compromise_score", "weighted_rel_pct"]).reset_index(drop=True)

    print("\nBest by lowest weighted relative error:")
    print(by_error.head(1).to_string(index=False))

    print("\nBest by speed:")
    print(by_time.head(1).to_string(index=False))

    print("\nBest compromise (error + speed):")
    print(by_compromise.head(1).to_string(index=False))

    print("\nTop 10 compromise configs:")
    print(by_compromise.head(10).to_string(index=False))

    out_path = os.path.join(BASE_DIR, "linearized_sigma_features_tuning.csv")
    by_compromise.to_csv(out_path, index=False)
    print(f"\nSaved tuning table to: {out_path}")


if __name__ == "__main__":
    main()
