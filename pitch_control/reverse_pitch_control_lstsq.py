"""
Reverse Pitch Control via Optimized (Iterative) Linear Least Squares
====================================================================
Optimized for performance: static matrices are precomputed, duplicate tensor 
reductions are eliminated, and smart-density initialization is utilized.
"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from LaurieOnTracking import Metrica_PitchControl as mpc

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def _approx_cos_deriv(x):
    x = np.asarray(x)
    x_wrapped = np.mod(x, 2 * np.pi)
    return np.where(x_wrapped <= np.pi, -2.0 / 2.0, 2.0 / 2.0)  # derivative of piece-wise slopes (-2/pi and 2/pi modified to fit scaled chain rules)
    # Keeping original derivative scaling for matching consistency:
    return np.where(x_wrapped <= np.pi, -2.0 / np.pi, 2.0 / np.pi)


@dataclass
class LSTSQSolution:
    players_xy: np.ndarray
    mae: float
    mse: float
    rmse: float
    iterations_used: int
    converged: bool
    history_mse: list


class LeastSquaresPitchControlSolver:
    def __init__(self, field_dimen=(106.0, 68.0), sigma=38.0, sharpness=3.0,
                 fgg: Optional["mpc.FastGaussianGrid"] = None):
        self.field_dimen = field_dimen
        self.sigma = sigma
        self.sharpness = sharpness
        self.fgg = fgg if fgg is not None else mpc.FastGaussianGrid(sigma=sigma)
        self.M = self.fgg.M

    def _build_grid(self, nx: int, ny: int):
        dx = self.field_dimen[0] / nx
        dy = self.field_dimen[1] / ny
        xgrid = np.arange(nx) * dx - self.field_dimen[0] / 2.0 + dx / 2.0
        ygrid = np.arange(ny) * dy - self.field_dimen[1] / 2.0 + dy / 2.0
        xx, yy = np.meshgrid(xgrid, ygrid)
        return np.column_stack([xx.ravel(), yy.ravel()])

    def _forward_and_jacobian(self, GM, players_xy):
        """
        OPTIMIZED: Receives precomputed static matrix GM instead of G_tilde.
        """
        n_grid = GM.shape[0]
        n_players = players_xy.shape[0]
        sigma = self.sigma
        s = self.sharpness

        P_tilde = self.fgg._lift_coords(players_xy)          # (n_players, 4)
        r_sq = GM @ P_tilde.T                                 # (n_grid, n_players)
        r_sq_clipped = np.maximum(r_sq, 0.0)

        t = self.fgg.linearizer.t            # (n_features,)
        c = self.fgg.linearizer.c            # (n_features,)

        arg = 2.0 * r_sq_clipped[:, :, None] * t[None, None, :]
        cos_vals = mpc.approx_cos_numpy(arg)                  # (n_grid, n_players, n_features)
        dcos_vals = _approx_cos_deriv(arg)                    # same shape

        # OPTIMIZATION: Compute tensordot ONLY ONCE instead of twice
        phi_raw = np.tensordot(cos_vals, c, axes=([2], [0]))  # (n_grid, n_players)
        phi = np.maximum(phi_raw, 0.0)

        dphi_dr = np.tensordot(dcos_vals * (2.0 * t)[None, None, :], c, axes=([2], [0]))
        
        # Merge mask checks to avoid redundant overhead allocation
        dphi_dr = np.where((r_sq > 0) & (phi_raw > 0), dphi_dr, 0.0)

        contrib = phi ** s                                    # (n_grid, n_players)
        h_hat = contrib.sum(axis=1)                           # (n_grid,)

        with np.errstate(invalid="ignore"):
            dcontrib_dr = s * np.where(phi > 0, phi ** (s - 1.0), 0.0) * dphi_dr  # (n_grid, n_players)

        # Pre-calculated scalar inversion constants
        inv_sigma = 1.0 / sigma
        inv_sigma_sq2 = 2.0 / (sigma ** 2)

        dr_dx = GM[:, [0]] * inv_sigma + GM[:, [2]] * (players_xy[:, 0][None, :] * inv_sigma_sq2)
        dr_dy = GM[:, [1]] * inv_sigma + GM[:, [2]] * (players_xy[:, 1][None, :] * inv_sigma_sq2)

        J = np.zeros((n_grid, 2 * n_players), dtype=float)
        J[:, 0::2] = dcontrib_dr * dr_dx
        J[:, 1::2] = dcontrib_dr * dr_dy

        return h_hat, J

    def solve(
        self,
        heatmap: np.ndarray,
        n_players: int,
        init_positions: Optional[np.ndarray] = None,
        max_iterations: int = 30,
        step: float = 1.0,
        damping: float = 1e-3,
        tol: float = 1e-8,
        seed: int = 0,
        verbose: bool = False,
    ) -> LSTSQSolution:
        if heatmap.ndim != 2:
            raise ValueError("heatmap must be 2D")

        ny, nx = heatmap.shape
        grid_coords = self._build_grid(nx, ny)
        
        # OPTIMIZATION: Precompute static matrices outside the iterative loops
        G_tilde = self.fgg._lift_coords(grid_coords)
        GM = G_tilde @ self.M
        h_target = heatmap.reshape(-1).astype(float)

        x_min, x_max = -self.field_dimen[0] / 2.0, self.field_dimen[0] / 2.0
        y_min, y_max = -self.field_dimen[1] / 2.0, self.field_dimen[1] / 2.0

        if init_positions is not None:
            players = np.asarray(init_positions, dtype=float).copy()
        else:
            # OPTIMIZATION / ACCURACY: Smart Heatmap-Informed Initialization
            # Instead of complete random blind placements, sample locations where heatmap has density
            rng = np.random.default_rng(seed)
            flat_h = np.clip(h_target, 0.0, None)
            h_sum = flat_h.sum()
            
            if h_sum > 1e-6:
                probs = flat_h / h_sum
                chosen_indices = rng.choice(len(grid_coords), size=n_players, p=probs, replace=True)
                players = grid_coords[chosen_indices] + rng.uniform(-2.0, 2.0, (n_players, 2))
            else:
                players = np.column_stack([
                    rng.uniform(x_min * 0.5, x_max * 0.5, n_players),
                    rng.uniform(y_min * 0.5, y_max * 0.5, n_players)
                ])

        history_mse = []
        prev_mse = None
        converged = False
        iters_used = 0

        # Cache Identity Matrix for Ridge Regularization
        I_reg = np.eye(2 * n_players)

        for it in range(max_iterations):
            h_hat, J = self._forward_and_jacobian(GM, players)
            residual = h_target - h_hat
            mse = float(np.mean(residual ** 2))
            history_mse.append(mse)
            iters_used = it + 1

            if verbose:
                mae = float(np.mean(np.abs(residual)))
                print(f"iter {it:3d}  mse={mse:.6e}  mae={mae:.6e}")

            if prev_mse is not None:
                if (prev_mse - mse) / max(prev_mse, 1e-12) < tol:
                    converged = True
                    break
            prev_mse = mse

            JTJ = J.T @ J
            JTr = J.T @ residual
            
            try:
                dP = np.linalg.solve(JTJ + damping * I_reg, JTr)
            except np.linalg.LinAlgError:
                dP, *_ = np.linalg.lstsq(J, residual, rcond=None)

            players += step * dP.reshape(n_players, 2)
            players[:, 0] = np.clip(players[:, 0], x_min, x_max)
            players[:, 1] = np.clip(players[:, 1], y_min, y_max)

        h_hat, _ = self._forward_and_jacobian(GM, players)
        residual = h_target - h_hat
        mae = float(np.mean(np.abs(residual)))
        mse = float(np.mean(residual ** 2))

        return LSTSQSolution(
            players_xy=players, mae=mae, mse=mse, rmse=float(np.sqrt(mse)),
            iterations_used=iters_used, converged=converged, history_mse=history_mse
        )


    def solve_collective_alternating(
            self,
            collective_heatmap: np.ndarray,
            n_att: int,
            n_def: int,
            init_att: Optional[np.ndarray] = None,
            init_def: Optional[np.ndarray] = None,
            outer_max_iterations: int = 10,
            inner_max_iterations: int = 15,
            step: float = 1.0,
            damping: float = 1e-3,
            tol: float = 1e-6,
            epsilon: float = 1e-6,  # EXPOSED: Numerical stabilization parameter
            seed: int = 0,
            verbose: bool = False,
        ):
            if collective_heatmap.ndim != 2:
                raise ValueError("collective_heatmap must be 2D")

            ny, nx = collective_heatmap.shape
            grid_coords = self._build_grid(nx, ny)
            
            # Precompute static matrices once for the entire alternating process
            G_tilde = self.fgg._lift_coords(grid_coords)
            GM = G_tilde @ self.M
            P = collective_heatmap.reshape(-1).astype(float)

            x_min, x_max = -self.field_dimen[0] / 2.0, self.field_dimen[0] / 2.0
            y_min, y_max = -self.field_dimen[1] / 2.0, self.field_dimen[1] / 2.0
            rng = np.random.default_rng(seed)

            # Smart Heatmap-Informed Initialization for Attackers (High P)
            if init_att is not None:
                att_pos = np.asarray(init_att, dtype=float).copy()
            else:
                probs_att = np.clip(P, 0.0, None)
                s_att = probs_att.sum()
                if s_att > 1e-6:
                    idx = rng.choice(len(grid_coords), size=n_att, p=probs_att/s_att, replace=True)
                    att_pos = grid_coords[idx] + rng.uniform(-2.0, 2.0, (n_att, 2))
                else:
                    att_pos = np.column_stack([rng.uniform(x_min, x_max, n_att), rng.uniform(y_min, y_max, n_att)])

            # Smart Heatmap-Informed Initialization for Defenders (Low P)
            if init_def is not None:
                def_pos = np.asarray(init_def, dtype=float).copy()
            else:
                probs_def = np.clip(1.0 - P, 0.0, None)
                s_def = probs_def.sum()
                if s_def > 1e-6:
                    idx = rng.choice(len(grid_coords), size=n_def, p=probs_def/s_def, replace=True)
                    def_pos = grid_coords[idx] + rng.uniform(-2.0, 2.0, (n_def, 2))
                else:
                    def_pos = np.column_stack([rng.uniform(x_min, x_max, n_def), rng.uniform(y_min, y_max, n_def)])

            history_mse = []
            prev_mse = None
            converged = False
            iters_used = 0

            for outer_it in range(outer_max_iterations):
                # =========================================================
                # STEP A: Optimize Attackers (Fix Defenders)
                # =========================================================
                # 1. Compute current Defender Influence
                I_def_raw, _ = self._forward_and_jacobian(GM, def_pos)

                safe_def = np.clip(I_def_raw, 1e-4, None)
                
                # 2. Extract Target Attacker Influence
                # Cap at 100.0 to prevent gradient explosions in zones where P=1
                I_att_target = safe_def * (P / (1.0 - P + epsilon))
                I_att_target = np.clip(I_att_target, 0.0, float(n_att))
                
                # 3. Inner solve for Attackers
                sol_att = self.solve(
                    heatmap=I_att_target.reshape(ny, nx),
                    n_players=n_att,
                    init_positions=att_pos,
                    max_iterations=inner_max_iterations,
                    step=step, damping=damping, tol=tol, seed=seed, verbose=False
                )
                att_pos = sol_att.players_xy

                # =========================================================
                # STEP B: Optimize Defenders (Fix Attackers)
                # =========================================================
                # 1. Compute newly updated Attacker Influence
                I_att_raw, _ = self._forward_and_jacobian(GM, att_pos)
                safe_att = np.clip(I_att_raw, 1e-4, None)

                # 2. Extract Target Defender Influence
                I_def_target = safe_att * ((1.0 - P) / (P + epsilon))
                I_def_target = np.clip(I_def_target, 0.0, float(n_def))
                
                # 3. Inner solve for Defenders
                sol_def = self.solve(
                    heatmap=I_def_target.reshape(ny, nx),
                    n_players=n_def,
                    init_positions=def_pos,
                    max_iterations=inner_max_iterations,
                    step=step, damping=damping, tol=tol, seed=seed+1, verbose=False
                )
                def_pos = sol_def.players_xy

                # =========================================================
                # STEP C: Convergence Check
                # =========================================================
                # Recompute defender influence using updated positions to get final P_hat
                I_def_raw, _ = self._forward_and_jacobian(GM, def_pos)
                P_hat = I_att_raw / (I_att_raw + I_def_raw + epsilon)
                
                mse = float(np.mean((P - P_hat) ** 2))
                history_mse.append(mse)
                iters_used = outer_it + 1

                if verbose:
                    print(f"Outer Iter {outer_it:3d} | Collective MSE: {mse:.6e}")

                if prev_mse is not None:
                    if (prev_mse - mse) / max(prev_mse, 1e-12) < tol:
                        converged = True
                        break
                prev_mse = mse

            return {
                "att_xy": att_pos,
                "def_xy": def_pos,
                "mse": prev_mse,
                "iterations": iters_used,
                "converged": converged,
                "history_mse": history_mse
            }
        
    def reconstruct_heatmap(self, players_xy: np.ndarray, ny: int, nx: int):
        grid_coords = self._build_grid(nx, ny)
        G_tilde = self.fgg._lift_coords(grid_coords)
        GM = G_tilde @ self.M
        h_hat, _ = self._forward_and_jacobian(GM, players_xy)
        return h_hat.reshape(ny, nx)