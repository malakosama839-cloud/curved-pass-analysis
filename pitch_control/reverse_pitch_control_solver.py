import argparse
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from LaurieOnTracking import Metrica_IO as mio
from LaurieOnTracking import Metrica_Velocities as mvel
from LaurieOnTracking import Metrica_PitchControl as mpc


BASE_DIR = os.path.abspath(os.path.dirname(__file__))


@dataclass
class ReverseSolution:
    players_xy: np.ndarray
    mae: float
    mse: float
    norm_consistency_error: float


class ReversePitchControlSolver:
    """
    Reverse solver for a simplified matrix-only pitch-control flow.

    Forward approximation used here:
        h ~= G_tilde @ M @ p_sum
    where:
        - h is flattened heatmap
        - G_tilde is lifted grid matrix
        - M is the fixed 4x4 matrix from FastGaussianGrid
        - p_sum is the summed lifted player vector

    Reverse step requested by user is implemented as:
        p_sum ~= inv(M) @ pinv(G_tilde) @ h

    From p_sum, there are infinitely many player sets whose lifted vectors sum to p_sum.
    This solver samples many feasible sets and ranks them by reconstruction quality.
    """

    def __init__(self, field_dimen=(106.0, 68.0), sigma=38.0):
        self.field_dimen = field_dimen
        self.sigma = sigma
        self.M = np.array(
            [
                [-2.0, 0.0, 0.0, 0.0],
                [0.0, -2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=float,
        )

    def _build_grid(self, nx: int, ny: int):
        dx = self.field_dimen[0] / nx
        dy = self.field_dimen[1] / ny
        xgrid = np.arange(nx) * dx - self.field_dimen[0] / 2.0 + dx / 2.0
        ygrid = np.arange(ny) * dy - self.field_dimen[1] / 2.0 + dy / 2.0
        xx, yy = np.meshgrid(xgrid, ygrid)
        grid_coords = np.column_stack([xx.ravel(), yy.ravel()])
        return xgrid, ygrid, grid_coords

    def _lift_coords(self, coords: np.ndarray):
        coords = np.atleast_2d(coords)
        x = coords[:, 0] / self.sigma
        y = coords[:, 1] / self.sigma
        norm_sq = x ** 2 + y ** 2
        return np.column_stack((x, y, norm_sq, np.ones_like(x)))

    def _estimate_p_sum(self, heatmap: np.ndarray):
        ny, nx = heatmap.shape
        _, _, grid_coords = self._build_grid(nx, ny)
        G_tilde = self._lift_coords(grid_coords)
        GM = G_tilde @ self.M

        h = heatmap.reshape(-1)
        G_pinv = np.linalg.pinv(G_tilde)
        M_inv = np.linalg.inv(self.M)

        # Requested reverse direction: inv(M) @ pinv(G) @ h
        p_sum = M_inv @ (G_pinv @ h)
        return p_sum, G_tilde, GM, h

    def _reconstruct_heatmap(self, G_tilde: np.ndarray, p_sum: np.ndarray, ny: int, nx: int):
        h_hat = G_tilde @ self.M @ p_sum
        return h_hat.reshape(ny, nx)

    def generate_infinite_player_sets(
        self,
        heatmap: np.ndarray,
        n_players: int = 11,
        num_candidates: int = 500,
        top_k: int = 20,
        seed: int = 7,
    ) -> dict:
        if heatmap.ndim != 2:
            raise ValueError("heatmap must be a 2D array")

        ny, nx = heatmap.shape
        p_sum_est, G_tilde, GM, h = self._estimate_p_sum(heatmap)

        rng = np.random.default_rng(seed)
        x_min, x_max = -self.field_dimen[0] / 2.0, self.field_dimen[0] / 2.0
        y_min, y_max = -self.field_dimen[1] / 2.0, self.field_dimen[1] / 2.0

        tx, ty, tn, tw = p_sum_est
        solutions: List[ReverseSolution] = []

        for _ in range(num_candidates):
            if n_players == 1:
                x_last = tx * self.sigma
                y_last = ty * self.sigma
                players = np.array([[np.clip(x_last, x_min, x_max), np.clip(y_last, y_min, y_max)]], dtype=float)
            else:
                margin_x = 0.8
                margin_y = 0.8
                players = np.column_stack(
                    [
                        rng.uniform(x_min + margin_x, x_max - margin_x, n_players),
                        rng.uniform(y_min + margin_y, y_max - margin_y, n_players),
                    ]
                )

                # Enforce target first-moment constraints by distributing correction over all players.
                for _corr in range(3):
                    lifted_now = self._lift_coords(players)
                    sx = np.sum(lifted_now[:, 0])
                    sy = np.sum(lifted_now[:, 1])

                    dx_each = (tx - sx) * self.sigma / n_players
                    dy_each = (ty - sy) * self.sigma / n_players

                    players[:, 0] += dx_each
                    players[:, 1] += dy_each
                    players[:, 0] = np.clip(players[:, 0], x_min, x_max)
                    players[:, 1] = np.clip(players[:, 1], y_min, y_max)

            lifted = self._lift_coords(players)
            p_sum_candidate = lifted.sum(axis=0)

            h_hat = GM @ p_sum_candidate
            diff = h_hat - h
            mae = float(np.mean(np.abs(diff)))
            mse = float(np.mean(diff ** 2))
            norm_consistency_error = float(abs(tn - p_sum_candidate[2]) + abs(tw - p_sum_candidate[3]))

            solutions.append(
                ReverseSolution(
                    players_xy=players,
                    mae=mae,
                    mse=mse,
                    norm_consistency_error=norm_consistency_error,
                )
            )

        solutions.sort(key=lambda s: (s.mse, s.norm_consistency_error, s.mae))
        best = solutions[:top_k]

        return {
            "p_sum_est": p_sum_est,
            "n_players": n_players,
            "top_solutions": best,
            "reconstructed_heatmap_from_p_sum": self._reconstruct_heatmap(G_tilde, p_sum_est, ny, nx),
        }


def load_heatmap(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".csv":
        arr = pd.read_csv(path, header=None).values
    else:
        raise ValueError("Unsupported file format. Use .npy or .csv")

    if arr.ndim != 2:
        raise ValueError("Loaded heatmap is not 2D")
    return arr.astype(float)


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
    return min(home_avg_x, key=home_avg_x.get), max(away_avg_x, key=away_avg_x.get)


def assignment_distances(actual_xy: np.ndarray, recon_xy: np.ndarray) -> np.ndarray:
    cost = np.linalg.norm(actual_xy[:, None, :] - recon_xy[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    return cost[row_ind, col_ind]


def evaluate_reverse_constraint1(
    game_id: int = 3812,
    field_dimen=(106.0, 68.0),
    n_grid_cells_x: int = 25,
    sigma: float = 38.0,
    sharpness: float = 40.0,
    num_candidates: int = 400,
    max_events: int = 0,
    seed: int = 7,
    out_prefix: str = "reverse_constraint1",
):
    tracking_home, tracking_away, events = mio.load_new_data(os.path.join(BASE_DIR, "..", "data", "pff_match_data"), game_id)

    tracking_home = to_metric_coordinates(tracking_home, field_dimen)
    tracking_away = to_metric_coordinates(tracking_away, field_dimen)
    events["Start X"] = (events["Start X"] - 0.5) * field_dimen[0]
    events["Start Y"] = (events["Start Y"] - 0.5) * field_dimen[1]

    tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True)
    tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)

    home_gk, away_gk = find_goalkeepers(tracking_home, tracking_away)
    GK_numbers = (home_gk, away_gk)
    params = mpc.default_model_params()

    event_ids = list(events.index)
    if max_events and max_events > 0:
        event_ids = event_ids[:max_events]

    n_grid_cells_y = int(n_grid_cells_x * field_dimen[1] / field_dimen[0])
    dx = field_dimen[0] / n_grid_cells_x
    dy = field_dimen[1] / n_grid_cells_y
    xgrid = np.arange(n_grid_cells_x) * dx - field_dimen[0] / 2.0 + dx / 2.0
    ygrid = np.arange(n_grid_cells_y) * dy - field_dimen[1] / 2.0 + dy / 2.0
    xx, yy = np.meshgrid(xgrid, ygrid)
    grid_coords = np.column_stack([xx.ravel(), yy.ravel()])

    reverse_solver = ReversePitchControlSolver(field_dimen=field_dimen, sigma=sigma)
    fgg = mpc.FastGaussianGrid(sigma=sigma)

    attack_distances_all = []
    defense_distances_all = []
    per_event_rows = []

    for idx, event_id in enumerate(event_ids, start=1):
        pass_frame = events.loc[event_id]["Start Frame"]
        pass_team = events.loc[event_id].Team

        if pass_team == "Home":
            attacking_players = mpc.initialise_players(tracking_home.loc[pass_frame], "Home", params, GK_numbers[0])
            defending_players = mpc.initialise_players(tracking_away.loc[pass_frame], "Away", params, GK_numbers[1])
        elif pass_team == "Away":
            defending_players = mpc.initialise_players(tracking_home.loc[pass_frame], "Home", params, GK_numbers[0])
            attacking_players = mpc.initialise_players(tracking_away.loc[pass_frame], "Away", params, GK_numbers[1])
        else:
            continue

        att_pos = np.array([[p.position[0], p.position[1]] for p in attacking_players], dtype=float)
        def_pos = np.array([[p.position[0], p.position[1]] for p in defending_players], dtype=float)

        if len(att_pos) == 0 or len(def_pos) == 0:
            continue

        att_heatmap = fgg.generate_heatmap(grid_coords, att_pos, sharpness=sharpness).reshape(n_grid_cells_y, n_grid_cells_x)
        def_heatmap = fgg.generate_heatmap(grid_coords, def_pos, sharpness=sharpness).reshape(n_grid_cells_y, n_grid_cells_x)

        att_result = reverse_solver.generate_infinite_player_sets(
            att_heatmap,
            n_players=len(att_pos),
            num_candidates=num_candidates,
            top_k=1,
            seed=seed + idx,
        )
        def_result = reverse_solver.generate_infinite_player_sets(
            def_heatmap,
            n_players=len(def_pos),
            num_candidates=num_candidates,
            top_k=1,
            seed=seed + 10000 + idx,
        )

        att_recon = att_result["top_solutions"][0].players_xy
        def_recon = def_result["top_solutions"][0].players_xy

        att_dist = assignment_distances(att_pos, att_recon)
        def_dist = assignment_distances(def_pos, def_recon)

        attack_distances_all.extend(att_dist.tolist())
        defense_distances_all.extend(def_dist.tolist())

        per_event_rows.append(
            {
                "event_id": int(event_id),
                "attack_mean_error_m": float(np.mean(att_dist)),
                "attack_std_error_m": float(np.std(att_dist)),
                "defense_mean_error_m": float(np.mean(def_dist)),
                "defense_std_error_m": float(np.std(def_dist)),
                "attack_players": int(len(att_dist)),
                "defense_players": int(len(def_dist)),
            }
        )

        if idx % 20 == 0:
            print(f"Processed {idx}/{len(event_ids)} events...")

    attack_arr = np.array(attack_distances_all, dtype=float)
    defense_arr = np.array(defense_distances_all, dtype=float)
    combined_arr = np.concatenate([attack_arr, defense_arr]) if attack_arr.size and defense_arr.size else np.array([])

    summary = {
        "events_processed": len(per_event_rows),
        "attack_avg_error_m": float(np.mean(attack_arr)) if attack_arr.size else np.nan,
        "attack_std_error_m": float(np.std(attack_arr)) if attack_arr.size else np.nan,
        "defense_avg_error_m": float(np.mean(defense_arr)) if defense_arr.size else np.nan,
        "defense_std_error_m": float(np.std(defense_arr)) if defense_arr.size else np.nan,
        "combined_avg_error_m": float(np.mean(combined_arr)) if combined_arr.size else np.nan,
        "combined_std_error_m": float(np.std(combined_arr)) if combined_arr.size else np.nan,
    }

    per_event_df = pd.DataFrame(per_event_rows)
    summary_df = pd.DataFrame([summary])

    per_event_path = f"{out_prefix}_per_event.csv"
    summary_path = f"{out_prefix}_summary.csv"
    per_event_df.to_csv(per_event_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nConstraint #1 reverse-reconstruction summary")
    print(summary_df.to_string(index=False))
    print(f"Saved per-event errors: {per_event_path}")
    print(f"Saved summary: {summary_path}")

    return summary, per_event_df


def main():
    parser = argparse.ArgumentParser(description="Reverse matrix pitch-control solver from heatmap.")
    parser.add_argument("--heatmap", type=str, default="", help="Path to heatmap (.npy or .csv)")
    parser.add_argument("--evaluate-all-events", action="store_true", help="Run constraint #1 evaluation across all events.")
    parser.add_argument("--sigma", type=float, default=38.0)
    parser.add_argument("--field-length", type=float, default=106.0)
    parser.add_argument("--field-width", type=float, default=68.0)
    parser.add_argument("--n-players", type=int, default=11)
    parser.add_argument("--num-candidates", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--n-grid", type=int, default=25)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-prefix", type=str, default="reverse_solver")
    args = parser.parse_args()

    if args.evaluate_all_events:
        evaluate_reverse_constraint1(
            game_id=3812,
            field_dimen=(args.field_length, args.field_width),
            n_grid_cells_x=args.n_grid,
            sigma=args.sigma,
            num_candidates=args.num_candidates,
            max_events=args.max_events,
            seed=args.seed,
            out_prefix=args.out_prefix,
        )
        return

    if not args.heatmap:
        raise ValueError("--heatmap is required unless --evaluate-all-events is used")

    heatmap = load_heatmap(args.heatmap)

    solver = ReversePitchControlSolver(
        field_dimen=(args.field_length, args.field_width),
        sigma=args.sigma,
    )

    result = solver.generate_infinite_player_sets(
        heatmap,
        n_players=args.n_players,
        num_candidates=args.num_candidates,
        top_k=args.top_k,
        seed=args.seed,
    )

    p_sum = result["p_sum_est"]
    print("Estimated summed lifted player vector [x/sigma, y/sigma, norm, ones_sum]:")
    print(np.array2string(p_sum, precision=6))

    rows = []
    for i, sol in enumerate(result["top_solutions"], start=1):
        rows.append(
            {
                "rank": i,
                "mse": sol.mse,
                "mae": sol.mae,
                "norm_consistency_error": sol.norm_consistency_error,
                "players_xy": sol.players_xy.tolist(),
            }
        )

    out_csv = f"{args.out_prefix}_top_solutions.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved top solutions: {out_csv}")

    recon_csv = f"{args.out_prefix}_reconstructed_heatmap.csv"
    pd.DataFrame(result["reconstructed_heatmap_from_p_sum"]).to_csv(recon_csv, header=False, index=False)
    print(f"Saved reconstructed heatmap: {recon_csv}")


if __name__ == "__main__":
    main()
