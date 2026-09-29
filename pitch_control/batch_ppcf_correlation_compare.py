"""
Batch PPCF Correlation Comparison

Compares event-wise PPCF correlation between:
1) blind reconstruction targets (derived from PPCF only)
2) position-guided reconstruction targets (derived from true player positions)

This is intended to quantify the effect of strict blindness vs older leakage-style
team heatmap targets while keeping the reconstruction pipeline otherwise aligned.
"""

import argparse
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "LaurieOnTracking"))

import Metrica_IO as mio
import Metrica_Velocities as mvel
import Metrica_PitchControl as mpc
from reverse_pitch_control_solver import ReversePitchControlSolver


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


def build_team_heatmap_proxies(target_ppcf):
    p = np.clip(target_ppcf, 1e-4, 1.0 - 1e-4)
    confidence = np.abs(2.0 * p - 1.0)
    uncertainty = 1.0 - confidence
    center_band = np.exp(-((p - 0.5) / 0.20) ** 2)
    gy, gx = np.gradient(p)
    boundary = np.hypot(gx, gy)
    boundary_scale = np.percentile(boundary, 95) + 1e-8
    boundary_norm = np.clip(boundary / boundary_scale, 0.0, 1.0)
    total_proxy = (
        0.08
        + 0.24 * confidence
        + 0.34 * uncertainty
        + 0.22 * boundary_norm
        + 0.26 * center_band
    )

    padded = np.pad(total_proxy, ((1, 1), (1, 1)), mode="reflect")
    total_proxy = (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 9.0
    total_proxy = np.clip(total_proxy, 0.10, None)
    att_proxy = total_proxy * p
    def_proxy = total_proxy * (1.0 - p)
    return att_proxy, def_proxy


def build_grid(field_dimen, n_grid_cells_x):
    n_grid_cells_y = int(n_grid_cells_x * field_dimen[1] / field_dimen[0])
    dx = field_dimen[0] / n_grid_cells_x
    dy = field_dimen[1] / n_grid_cells_y
    xgrid = np.arange(n_grid_cells_x) * dx - field_dimen[0] / 2.0 + dx / 2.0
    ygrid = np.arange(n_grid_cells_y) * dy - field_dimen[1] / 2.0 + dy / 2.0
    xx, yy = np.meshgrid(xgrid, ygrid)
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    return coords, n_grid_cells_y


def compute_matrix_ppcf_from_positions(fgg, G_tilde, y_len, nx, att_pos, def_pos):
    att_inf = fgg.generate_heatmap_from_lifted_grid(G_tilde, att_pos, sharpness=40).reshape(y_len, nx)
    def_inf = fgg.generate_heatmap_from_lifted_grid(G_tilde, def_pos, sharpness=40).reshape(y_len, nx)
    total = att_inf + def_inf
    return np.where(total > 1e-12, att_inf / total, 0.5)


def team_spatial_regularization(team_pos, field_dimen):
    x_max = field_dimen[0] / 2.0
    y_max = field_dimen[1] / 2.0

    x = np.abs(team_pos[:, 0]) / max(x_max, 1e-6)
    y = np.abs(team_pos[:, 1]) / max(y_max, 1e-6)

    corner_mask = (x > 0.90) & (y > 0.88)
    corner_excess = max(0, int(np.sum(corner_mask)) - 1)
    corner_penalty = corner_excess / max(len(team_pos), 1)

    margin_x = x_max - np.abs(team_pos[:, 0])
    margin_y = y_max - np.abs(team_pos[:, 1])
    edge_penalty = np.mean(np.clip((1.5 - np.minimum(margin_x, margin_y)) / 1.5, 0.0, 1.0))

    return float(corner_penalty + 0.35 * edge_penalty)


def ppcf_objective(fgg, G_tilde, y_len, nx, field_dimen, target_ppcf, att_pos, def_pos):
    ppcf_hat = compute_matrix_ppcf_from_positions(fgg, G_tilde, y_len, nx, att_pos, def_pos)
    diff = ppcf_hat - target_ppcf
    mae = float(np.mean(np.abs(diff)))
    reg = 0.5 * (
        team_spatial_regularization(att_pos, field_dimen) + team_spatial_regularization(def_pos, field_dimen)
    )
    objective = mae + 0.010 * reg
    return objective, ppcf_hat


def refine_team_positions(fgg, G_tilde, field_dimen, target_heatmap, init_pos, iterations, step_xy, seed):
    rng = np.random.default_rng(seed)
    x_min, x_max = -field_dimen[0] / 2.0, field_dimen[0] / 2.0
    y_min, y_max = -field_dimen[1] / 2.0, field_dimen[1] / 2.0

    team = init_pos.copy()
    target_flat = target_heatmap.reshape(-1)

    P_tilde = fgg._lift_coords(team)
    r_sq = np.maximum(G_tilde @ fgg.M @ P_tilde.T, 0.0)
    contrib_matrix = np.maximum(fgg.linearizer.predict(r_sq).reshape(r_sq.shape), 0.0) ** 40
    current_flat = np.sum(contrib_matrix, axis=1)

    best_score = float(np.mean(np.abs(current_flat - target_flat))) + 0.018 * team_spatial_regularization(team, field_dimen)
    no_improve = 0
    improve_eps = 1e-5
    patience = max(22, iterations // 5)

    for k in range(iterations):
        idx = int(rng.integers(0, len(team)))
        trial = team.copy()
        trial[idx, 0] += rng.normal(0.0, step_xy)
        trial[idx, 1] += rng.normal(0.0, step_xy)
        trial[idx, 0] = np.clip(trial[idx, 0], x_min, x_max)
        trial[idx, 1] = np.clip(trial[idx, 1], y_min, y_max)

        trial_player = trial[idx : idx + 1]
        P_trial = fgg._lift_coords(trial_player)
        r_sq_trial = np.maximum(G_tilde @ fgg.M @ P_trial.T, 0.0)
        new_contrib = np.maximum(fgg.linearizer.predict(r_sq_trial).reshape(-1), 0.0) ** 40

        trial_flat = current_flat - contrib_matrix[:, idx] + new_contrib
        trial_score = float(np.mean(np.abs(trial_flat - target_flat))) + 0.018 * team_spatial_regularization(trial, field_dimen)

        if trial_score < best_score - improve_eps:
            team = trial
            current_flat = trial_flat
            contrib_matrix[:, idx] = new_contrib
            best_score = trial_score
            no_improve = 0
        else:
            no_improve += 1

        if (k + 1) % 55 == 0:
            step_xy = max(0.7, step_xy * 0.78)

        if no_improve >= patience:
            break

    return team


def joint_refine_positions(fgg, G_tilde, y_len, nx, field_dimen, target_ppcf, att_init, def_init, iterations, step_xy, seed):
    rng = np.random.default_rng(seed)
    att = att_init.copy()
    deff = def_init.copy()

    x_min, x_max = -field_dimen[0] / 2.0, field_dimen[0] / 2.0
    y_min, y_max = -field_dimen[1] / 2.0, field_dimen[1] / 2.0

    best_obj, best_ppcf = ppcf_objective(fgg, G_tilde, y_len, nx, field_dimen, target_ppcf, att, deff)
    no_improve = 0
    improve_eps = 1e-5
    patience = max(26, iterations // 4)

    for k in range(iterations):
        temperature = max(0.02, 1.0 - k / max(iterations, 1))
        proposals = []

        if rng.random() < 0.5:
            idx = int(rng.integers(0, len(att)))
            trial_att = att.copy()
            trial_def = deff.copy()
            trial_att[idx, 0] += rng.normal(0.0, step_xy)
            trial_att[idx, 1] += rng.normal(0.0, step_xy)
            trial_att[idx, 0] = np.clip(trial_att[idx, 0], x_min, x_max)
            trial_att[idx, 1] = np.clip(trial_att[idx, 1], y_min, y_max)
        else:
            idx = int(rng.integers(0, len(deff)))
            trial_att = att.copy()
            trial_def = deff.copy()
            trial_def[idx, 0] += rng.normal(0.0, step_xy)
            trial_def[idx, 1] += rng.normal(0.0, step_xy)
            trial_def[idx, 0] = np.clip(trial_def[idx, 0], x_min, x_max)
            trial_def[idx, 1] = np.clip(trial_def[idx, 1], y_min, y_max)
        proposals.append((trial_att, trial_def))

        a_idx = int(rng.integers(0, len(att)))
        d_idx = int(rng.integers(0, len(deff)))
        trial_att2 = att.copy()
        trial_def2 = deff.copy()
        trial_att2[a_idx, 0] += rng.normal(0.0, 0.8 * step_xy)
        trial_att2[a_idx, 1] += rng.normal(0.0, 0.8 * step_xy)
        trial_def2[d_idx, 0] += rng.normal(0.0, 0.8 * step_xy)
        trial_def2[d_idx, 1] += rng.normal(0.0, 0.8 * step_xy)
        trial_att2[:, 0] = np.clip(trial_att2[:, 0], x_min, x_max)
        trial_att2[:, 1] = np.clip(trial_att2[:, 1], y_min, y_max)
        trial_def2[:, 0] = np.clip(trial_def2[:, 0], x_min, x_max)
        trial_def2[:, 1] = np.clip(trial_def2[:, 1], y_min, y_max)
        proposals.append((trial_att2, trial_def2))

        if rng.random() < 0.16:
            trial_att3 = att + rng.normal(0.0, 0.35, size=att.shape)
            trial_def3 = deff + rng.normal(0.0, 0.35, size=deff.shape)
            trial_att3[:, 0] = np.clip(trial_att3[:, 0], x_min, x_max)
            trial_att3[:, 1] = np.clip(trial_att3[:, 1], y_min, y_max)
            trial_def3[:, 0] = np.clip(trial_def3[:, 0], x_min, x_max)
            trial_def3[:, 1] = np.clip(trial_def3[:, 1], y_min, y_max)
            proposals.append((trial_att3, trial_def3))

        local_best = None
        for p_att, p_def in proposals:
            obj, ppcf_hat = ppcf_objective(fgg, G_tilde, y_len, nx, field_dimen, target_ppcf, p_att, p_def)
            if (local_best is None) or (obj < local_best[0]):
                local_best = (obj, p_att, p_def, ppcf_hat)

        trial_obj, trial_att, trial_def, trial_ppcf = local_best
        delta = trial_obj - best_obj
        if trial_obj < best_obj - improve_eps or rng.random() < np.exp(-max(delta, 0.0) / max(temperature, 1e-6)) * 0.02:
            att = trial_att.copy()
            deff = trial_def.copy()
            if trial_obj < best_obj - improve_eps:
                best_obj = trial_obj
                best_ppcf = trial_ppcf
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        if (k + 1) % 60 == 0:
            step_xy = max(0.55, step_xy * 0.72)

        if no_improve >= patience:
            break

    return att, deff, best_ppcf


def final_joint_ppcf_polish(fgg, G_tilde, y_len, nx, field_dimen, target_ppcf, att_init, def_init, iterations):
    best_att = att_init.copy()
    best_def = def_init.copy()
    best_obj, best_ppcf = ppcf_objective(fgg, G_tilde, y_len, nx, field_dimen, target_ppcf, best_att, best_def)

    for k in range(2):
        if k == 0:
            att_seed = best_att.copy()
            def_seed = best_def.copy()
        else:
            rng = np.random.default_rng(1200 + k)
            att_seed = best_att + rng.normal(0.0, 0.5, size=best_att.shape)
            def_seed = best_def + rng.normal(0.0, 0.5, size=best_def.shape)
            att_seed[:, 0] = np.clip(att_seed[:, 0], -field_dimen[0] / 2.0, field_dimen[0] / 2.0)
            att_seed[:, 1] = np.clip(att_seed[:, 1], -field_dimen[1] / 2.0, field_dimen[1] / 2.0)
            def_seed[:, 0] = np.clip(def_seed[:, 0], -field_dimen[0] / 2.0, field_dimen[0] / 2.0)
            def_seed[:, 1] = np.clip(def_seed[:, 1], -field_dimen[1] / 2.0, field_dimen[1] / 2.0)

        att_try, def_try, ppcf_try = joint_refine_positions(
            fgg,
            G_tilde,
            y_len,
            nx,
            field_dimen,
            target_ppcf,
            att_seed,
            def_seed,
            iterations=iterations,
            step_xy=0.95,
            seed=1300 + k,
        )
        obj_try, _ = ppcf_objective(fgg, G_tilde, y_len, nx, field_dimen, target_ppcf, att_try, def_try)
        if obj_try < best_obj:
            best_obj = obj_try
            best_att = att_try
            best_def = def_try
            best_ppcf = ppcf_try

    return best_att, best_def, best_ppcf


def get_event_positions(events, tracking_home, tracking_away, params, GK_numbers, event_id):
    event_info = events.loc[event_id]
    pass_frame = event_info["Start Frame"]
    pass_team = event_info.Team

    if pass_frame in tracking_home.index:
        h_row = tracking_home.loc[pass_frame]
        a_row = tracking_away.loc[pass_frame]
    else:
        h_row = tracking_home.iloc[(tracking_home.index - pass_frame).abs().argmin()]
        a_row = tracking_away.iloc[(tracking_away.index - pass_frame).abs().argmin()]

    if pass_team == "Home":
        att_players = mpc.initialise_players(h_row, "Home", params, GK_numbers[0])
        def_players = mpc.initialise_players(a_row, "Away", params, GK_numbers[1])
    elif pass_team == "Away":
        def_players = mpc.initialise_players(h_row, "Home", params, GK_numbers[0])
        att_players = mpc.initialise_players(a_row, "Away", params, GK_numbers[1])
    else:
        return None, None

    att_pos = np.array([[p.position[0], p.position[1]] for p in att_players], dtype=float)
    def_pos = np.array([[p.position[0], p.position[1]] for p in def_players], dtype=float)
    return att_pos, def_pos


def evaluate_mode(mode, events, tracking_home, tracking_away, params, GK_numbers, cfg):
    field_dimen = cfg["field_dimen"]
    nx = cfg["n_grid_cells_x"]

    fgg = mpc.FastGaussianGrid(sigma=cfg["sigma"])
    solver = ReversePitchControlSolver(field_dimen=field_dimen, sigma=cfg["sigma"])

    grid_coords, y_len = build_grid(field_dimen, nx)
    G_tilde = fgg._lift_coords(grid_coords)

    event_ids = list(events.index)
    if cfg["max_events"] > 0:
        event_ids = event_ids[: cfg["max_events"]]

    correlations = []
    spearman_scores = []
    ssim_scores = []
    skipped = 0

    t0 = time.time()
    for i, event_id in enumerate(event_ids, start=1):
        try:
            ppcf_original, _, _ = mpc.generate_pitch_control_for_event_linearized_matrixform(
                event_id,
                events,
                tracking_home,
                tracking_away,
                params,
                GK_numbers,
                field_dimen=field_dimen,
                n_grid_cells_x=nx,
                offsides=False,
            )

            att_pos, def_pos = get_event_positions(events, tracking_home, tracking_away, params, GK_numbers, event_id)
            if att_pos is None or def_pos is None or len(att_pos) == 0 or len(def_pos) == 0:
                skipped += 1
                continue

            if mode == "blind":
                att_target, def_target = build_team_heatmap_proxies(ppcf_original)
            else:
                att_target = fgg.generate_heatmap_from_lifted_grid(G_tilde, att_pos, sharpness=40).reshape(y_len, nx)
                def_target = fgg.generate_heatmap_from_lifted_grid(G_tilde, def_pos, sharpness=40).reshape(y_len, nx)

            seed_base = int(event_id) * 9973
            att_result = solver.generate_infinite_player_sets(
                att_target,
                n_players=len(att_pos),
                num_candidates=cfg["num_candidates"],
                top_k=cfg["top_k"],
                seed=17 + seed_base,
            )
            def_result = solver.generate_infinite_player_sets(
                def_target,
                n_players=len(def_pos),
                num_candidates=cfg["num_candidates"],
                top_k=cfg["top_k"],
                seed=7017 + seed_base,
            )

            att_solutions = att_result["top_solutions"]
            def_solutions = def_result["top_solutions"]
            best_obj = float("inf")
            att_recon = att_solutions[0].players_xy.copy()
            def_recon = def_solutions[0].players_xy.copy()

            for a_sol in att_solutions:
                for d_sol in def_solutions:
                    obj, _ = ppcf_objective(fgg, G_tilde, y_len, nx, field_dimen, ppcf_original, a_sol.players_xy, d_sol.players_xy)
                    if obj < best_obj:
                        best_obj = obj
                        att_recon = a_sol.players_xy.copy()
                        def_recon = d_sol.players_xy.copy()

            att_recon = refine_team_positions(
                fgg,
                G_tilde,
                field_dimen,
                att_target,
                att_recon,
                iterations=cfg["refine_iterations"],
                step_xy=1.8,
                seed=101 + seed_base,
            )
            def_recon = refine_team_positions(
                fgg,
                G_tilde,
                field_dimen,
                def_target,
                def_recon,
                iterations=cfg["refine_iterations"],
                step_xy=1.8,
                seed=202 + seed_base,
            )

            ppcf_recon = compute_matrix_ppcf_from_positions(fgg, G_tilde, y_len, nx, att_recon, def_recon)

            if mode == "blind":
                mismatch = np.abs(ppcf_original - ppcf_recon)
                mm_scale = np.percentile(mismatch, 95) + 1e-8
                mismatch_norm = np.clip(mismatch / mm_scale, 0.0, 1.0)
                center_weight = np.exp(-((ppcf_original - 0.5) / 0.20) ** 2)
                boost = 1.0 + 0.26 * mismatch_norm * (0.55 + 0.45 * center_weight)

                att_target_boost = att_target * boost
                def_target_boost = def_target * boost
                quick_iters = max(36, cfg["refine_iterations"] // 3)

                att_recon = refine_team_positions(
                    fgg,
                    G_tilde,
                    field_dimen,
                    att_target_boost,
                    att_recon,
                    iterations=quick_iters,
                    step_xy=1.25,
                    seed=505 + seed_base,
                )
                def_recon = refine_team_positions(
                    fgg,
                    G_tilde,
                    field_dimen,
                    def_target_boost,
                    def_recon,
                    iterations=quick_iters,
                    step_xy=1.25,
                    seed=606 + seed_base,
                )
                ppcf_recon = compute_matrix_ppcf_from_positions(fgg, G_tilde, y_len, nx, att_recon, def_recon)

                mae_after_boost = float(np.mean(np.abs(ppcf_original - ppcf_recon)))
                if mae_after_boost > 0.020:
                    mismatch2 = np.abs(ppcf_original - ppcf_recon)
                    mm_scale2 = np.percentile(mismatch2, 95) + 1e-8
                    mismatch_norm2 = np.clip(mismatch2 / mm_scale2, 0.0, 1.0)
                    center_weight2 = np.exp(-((ppcf_original - 0.5) / 0.20) ** 2)
                    boost2 = 1.0 + 0.18 * mismatch_norm2 * (0.55 + 0.45 * center_weight2)

                    att_target_boost2 = att_target * boost2
                    def_target_boost2 = def_target * boost2
                    quick_iters2 = max(24, cfg["refine_iterations"] // 5)

                    att_recon = refine_team_positions(
                        fgg,
                        G_tilde,
                        field_dimen,
                        att_target_boost2,
                        att_recon,
                        iterations=quick_iters2,
                        step_xy=1.05,
                        seed=707 + seed_base,
                    )
                    def_recon = refine_team_positions(
                        fgg,
                        G_tilde,
                        field_dimen,
                        def_target_boost2,
                        def_recon,
                        iterations=quick_iters2,
                        step_xy=1.05,
                        seed=808 + seed_base,
                    )
                    ppcf_recon = compute_matrix_ppcf_from_positions(fgg, G_tilde, y_len, nx, att_recon, def_recon)

            att_recon, def_recon, ppcf_recon = final_joint_ppcf_polish(
                fgg,
                G_tilde,
                y_len,
                nx,
                field_dimen,
                ppcf_original,
                att_recon,
                def_recon,
                iterations=max(70, cfg["refine_iterations"] // 2),
            )

            corr = float(np.corrcoef(ppcf_original.ravel(), ppcf_recon.ravel())[0, 1])
            rho, _ = spearmanr(ppcf_original.ravel(), ppcf_recon.ravel())
            rho = float(rho)
            ssim_score = float(ssim(ppcf_original, ppcf_recon, data_range=1.0))

            if np.isfinite(corr) and np.isfinite(rho) and np.isfinite(ssim_score):
                correlations.append(corr)
                spearman_scores.append(rho)
                ssim_scores.append(ssim_score)
            else:
                skipped += 1

            if i % cfg["progress_every"] == 0:
                print(f"[{mode}] Processed {i}/{len(event_ids)} events | kept={len(correlations)} skipped={skipped}")

        except Exception as exc:
            skipped += 1
            if cfg["verbose"]:
                print(f"[{mode}] Skipping event {event_id}: {exc}")

    elapsed = time.time() - t0
    corr_arr = np.array(correlations, dtype=float)
    spearman_arr = np.array(spearman_scores, dtype=float)
    ssim_arr = np.array(ssim_scores, dtype=float)

    result = {
        "mode": mode,
        "runtime_s": elapsed,
        "events_requested": len(event_ids),
        "events_used": int(corr_arr.size),
        "events_skipped": skipped,
        "mean_corr": float(np.mean(corr_arr)) if corr_arr.size else float("nan"),
        "std_corr": float(np.std(corr_arr)) if corr_arr.size else float("nan"),
        "min_corr": float(np.min(corr_arr)) if corr_arr.size else float("nan"),
        "max_corr": float(np.max(corr_arr)) if corr_arr.size else float("nan"),
        "mean_spearman": float(np.mean(spearman_arr)) if spearman_arr.size else float("nan"),
        "std_spearman": float(np.std(spearman_arr)) if spearman_arr.size else float("nan"),
        "min_spearman": float(np.min(spearman_arr)) if spearman_arr.size else float("nan"),
        "max_spearman": float(np.max(spearman_arr)) if spearman_arr.size else float("nan"),
        "mean_ssim": float(np.mean(ssim_arr)) if ssim_arr.size else float("nan"),
        "std_ssim": float(np.std(ssim_arr)) if ssim_arr.size else float("nan"),
        "min_ssim": float(np.min(ssim_arr)) if ssim_arr.size else float("nan"),
        "max_ssim": float(np.max(ssim_arr)) if ssim_arr.size else float("nan"),
    }
    return result


def parse_args():
    p = argparse.ArgumentParser(description="Compare PPCF correlation for blind vs position-guided reconstruction")
    p.add_argument("--mode", choices=["blind", "guided", "both"], default="both")
    p.add_argument("--game-id", type=int, default=3812)
    p.add_argument("--n-grid-cells-x", type=int, default=50)
    p.add_argument("--sigma", type=float, default=38.0)
    p.add_argument("--num-candidates", type=int, default=128)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--refine-iterations", type=int, default=150)
    p.add_argument("--max-events", type=int, default=0, help="0 means all events")
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def print_summary(result):
    print("\n=== MODE SUMMARY ===")
    print(f"Mode: {result['mode']}")
    print(f"Runtime: {result['runtime_s']:.2f} s")
    print(f"Events requested: {result['events_requested']}")
    print(f"Events used: {result['events_used']}")
    print(f"Events skipped: {result['events_skipped']}")
    print(f"Mean correlation: {result['mean_corr']:.5f}")
    print(f"Std correlation:  {result['std_corr']:.5f}")
    print(f"Min correlation:  {result['min_corr']:.5f}")
    print(f"Max correlation:  {result['max_corr']:.5f}")
    print(f"Mean Spearman:    {result['mean_spearman']:.5f}")
    print(f"Std Spearman:     {result['std_spearman']:.5f}")
    print(f"Min Spearman:     {result['min_spearman']:.5f}")
    print(f"Max Spearman:     {result['max_spearman']:.5f}")
    print(f"Mean SSIM:        {result['mean_ssim']:.5f}")
    print(f"Std SSIM:         {result['std_ssim']:.5f}")
    print(f"Min SSIM:         {result['min_ssim']:.5f}")
    print(f"Max SSIM:         {result['max_ssim']:.5f}")


def main():
    args = parse_args()
    field_dimen = (106.0, 68.0)

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

    cfg = {
        "field_dimen": field_dimen,
        "n_grid_cells_x": args.n_grid_cells_x,
        "sigma": args.sigma,
        "num_candidates": args.num_candidates,
        "top_k": args.top_k,
        "refine_iterations": args.refine_iterations,
        "max_events": args.max_events,
        "progress_every": args.progress_every,
        "verbose": args.verbose,
    }

    modes = [args.mode] if args.mode in ("blind", "guided") else ["blind", "guided"]
    results = []
    for mode in modes:
        print(f"\nRunning mode: {mode}")
        result = evaluate_mode(mode, events, tracking_home, tracking_away, params, GK_numbers, cfg)
        results.append(result)
        print_summary(result)

    if len(results) == 2:
        blind = next(r for r in results if r["mode"] == "blind")
        guided = next(r for r in results if r["mode"] == "guided")
        print("\n=== COMPARISON (guided - blind) ===")
        print(f"Mean correlation delta: {guided['mean_corr'] - blind['mean_corr']:+.5f}")
        print(f"Runtime delta (s):       {guided['runtime_s'] - blind['runtime_s']:+.2f}")


if __name__ == "__main__":
    main()
