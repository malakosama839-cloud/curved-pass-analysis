"""
Batch Player Position Reconstruction Error

Runs a full-event batch pass and reports player-location reconstruction distance
errors for attacking and defending teams using the latest project modules.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "LaurieOnTracking"))

import Metrica_IO as mio
import Metrica_Velocities as mvel
import Metrica_PitchControl as mpc
from reverse_pitch_control_solver import ReversePitchControlSolver, assignment_distances


def to_metric_coordinates(data, field_dimen=(106.0, 68.0)):
    for suffix, size in (("_x", field_dimen[0]), ("_y", field_dimen[1])):
        cols = [c for c in data.columns if c.endswith(suffix) and c[:4] in ("Home", "Away")]
        data[cols] = (data[cols] - 0.5) * size
    return data


def find_goalkeepers(tracking_home, tracking_away):
    home_players = sorted(
        set(c.split("_")[1] for c in tracking_home.columns if c.startswith("Home_") and c.endswith("_x"))
    )
    away_players = sorted(
        set(c.split("_")[1] for c in tracking_away.columns if c.startswith("Away_") and c.endswith("_x"))
    )
    home_gk = min(home_players, key=lambda p: tracking_home[f"Home_{p}_x"].mean())
    away_gk = max(away_players, key=lambda p: tracking_away[f"Away_{p}_x"].mean())
    return home_gk, away_gk


def parse_args():
    parser = argparse.ArgumentParser(description="Batch reconstruction distance error runner")
    parser.add_argument("--game-id", type=int, default=3812)
    parser.add_argument("--n-grid", type=int, default=32)
    parser.add_argument("--sigma", type=float, default=38.0)
    parser.add_argument("--sharpness", type=float, default=40.0)
    parser.add_argument("--num-cands", type=int, default=64)
    parser.add_argument("--max-events", type=int, default=0, help="0 means all events")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def main():
    args = parse_args()
    field_dimen = (106.0, 68.0)

    t0 = time.time()
    print("Loading data...")
    tracking_home, tracking_away, events = mio.load_new_data(os.path.join(BASE_DIR, "..", "data", "pff_match_data"), args.game_id)

    tracking_home = to_metric_coordinates(tracking_home, field_dimen)
    tracking_away = to_metric_coordinates(tracking_away, field_dimen)
    events["Start X"] = (events["Start X"] - 0.5) * field_dimen[0]
    events["Start Y"] = (events["Start Y"] - 0.5) * field_dimen[1]

    tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True)
    tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)

    home_gk, away_gk = find_goalkeepers(tracking_home, tracking_away)
    GK_numbers = (home_gk, away_gk)
    params = mpc.default_model_params()

    n_grid_y = int(args.n_grid * field_dimen[1] / field_dimen[0])
    dx = field_dimen[0] / args.n_grid
    dy = field_dimen[1] / n_grid_y
    xgrid = np.arange(args.n_grid) * dx - field_dimen[0] / 2.0 + dx / 2.0
    ygrid = np.arange(n_grid_y) * dy - field_dimen[1] / 2.0 + dy / 2.0
    xx, yy = np.meshgrid(xgrid, ygrid)
    grid_coords = np.column_stack([xx.ravel(), yy.ravel()])

    fgg = mpc.FastGaussianGrid(sigma=args.sigma)
    solver = ReversePitchControlSolver(field_dimen=field_dimen, sigma=args.sigma)

    event_ids = list(events.index)
    if args.max_events > 0:
        event_ids = event_ids[: args.max_events]

    attack_distances = []
    defense_distances = []
    processed_events = 0

    print(f"Starting batch over {len(event_ids)} events...")
    for i, event_id in enumerate(event_ids, start=1):
        row = events.loc[event_id]
        pass_frame = row["Start Frame"]
        pass_team = row.Team

        try:
            h_row = tracking_home.loc[pass_frame]
            a_row = tracking_away.loc[pass_frame]
        except KeyError:
            continue

        if pass_team == "Home":
            att_players = mpc.initialise_players(h_row, "Home", params, GK_numbers[0])
            def_players = mpc.initialise_players(a_row, "Away", params, GK_numbers[1])
        elif pass_team == "Away":
            att_players = mpc.initialise_players(a_row, "Away", params, GK_numbers[1])
            def_players = mpc.initialise_players(h_row, "Home", params, GK_numbers[0])
        else:
            continue

        att_pos = np.array([[p.position[0], p.position[1]] for p in att_players], dtype=float)
        def_pos = np.array([[p.position[0], p.position[1]] for p in def_players], dtype=float)

        if len(att_pos) == 0 or len(def_pos) == 0:
            continue

        processed_events += 1

        att_heatmap = fgg.generate_heatmap(grid_coords, att_pos, sharpness=args.sharpness).reshape(n_grid_y, args.n_grid)
        def_heatmap = fgg.generate_heatmap(grid_coords, def_pos, sharpness=args.sharpness).reshape(n_grid_y, args.n_grid)

        att_result = solver.generate_infinite_player_sets(
            att_heatmap,
            n_players=len(att_pos),
            num_candidates=args.num_cands,
            top_k=1,
            seed=17 + int(event_id),
        )
        def_result = solver.generate_infinite_player_sets(
            def_heatmap,
            n_players=len(def_pos),
            num_candidates=args.num_cands,
            top_k=1,
            seed=7017 + int(event_id),
        )

        att_recon = att_result["top_solutions"][0].players_xy
        def_recon = def_result["top_solutions"][0].players_xy

        attack_distances.extend(assignment_distances(att_pos, att_recon).tolist())
        defense_distances.extend(assignment_distances(def_pos, def_recon).tolist())

        if i % args.progress_every == 0:
            total_so_far = len(attack_distances) + len(defense_distances)
            print(f"Processed {i}/{len(event_ids)} events | reconstructed players={total_so_far}")

    attack_arr = np.array(attack_distances, dtype=float)
    defense_arr = np.array(defense_distances, dtype=float)
    combined = np.concatenate([attack_arr, defense_arr]) if attack_arr.size and defense_arr.size else np.array([])

    elapsed = time.time() - t0
    print("\n=== SUMMARY ===")
    print(f"Total runtime: {elapsed:.2f} s")
    print(f"Events requested: {len(event_ids)}")
    print(f"Events processed: {processed_events}")
    print(f"Attack players evaluated: {len(attack_arr)}")
    print(f"Defense players evaluated: {len(defense_arr)}")
    print(f"Total players evaluated: {len(combined)}")

    if attack_arr.size:
        print(f"Attack mean/std error: {attack_arr.mean():.3f} / {attack_arr.std():.3f} m")
    if defense_arr.size:
        print(f"Defense mean/std error: {defense_arr.mean():.3f} / {defense_arr.std():.3f} m")
    if combined.size:
        print(f"Combined mean/std error: {combined.mean():.3f} / {combined.std():.3f} m")
        print(f"Combined min/max error: {combined.min():.3f} / {combined.max():.3f} m")


if __name__ == "__main__":
    main()
