"""
Pitch Control Reconstruction Comparison - Interactive GUI
Pipeline:
1. Generates original Left Map via Metrica matrix formula.
2. Generates True Influence Targets (No proxies!).
3. Reverses player X,Y locations using pure Gauss-Newton Least Squares.
4. PURGES mock tracking frame (all old X, Y -> NaN, all velocities -> 0.0)
5. Injects solved X,Y coords and generates reconstructed Right Map.
"""
import sys
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# Import the optimized iterative linear least squares solver
from reverse_pitch_control_lstsq import LeastSquaresPitchControlSolver

# --- PATH SETUP ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
LAURIE_PATH = os.path.join(BASE_DIR, "LaurieOnTracking")
MATCH_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "pff_match_data")

sys.path.insert(0, LAURIE_PATH)

import Metrica_IO as mio
import Metrica_Velocities as mvel
import Metrica_PitchControl as mpc


# --- HELPER FUNCTIONS ---
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
    return home_gk, away_gk, home_players, away_players


def assignment_distances(pos1, pos2):
    """Computes tracking error distance via linear sum assignment matching."""
    cost = np.linalg.norm(pos1[:, None, :] - pos2[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    return cost[row_ind, col_ind]


class PitchControlReconstructionGUI:
    def __init__(self, events, tracking_home, tracking_away, params, GK_numbers, home_players, away_players, field_dimen):
        self.n_grid_cells_x = 64
        self.refine_iterations = 30
        self.damping = 0.1
        self.step = 1.0
        self.n_starts = 30
        
        # Solver Settings - 3.0 restores the natural fading!
        self.sharpness = 3.0
        self.sigma = 38.0

        self.events = events
        self.tracking_home = tracking_home
        self.tracking_away = tracking_away
        self.params = params
        self.GK_numbers = GK_numbers
        self.home_players = home_players
        self.away_players = away_players
        self.field_dimen = field_dimen

        # Instantiate analytical solver
        self.lstsq_solver = LeastSquaresPitchControlSolver(
            field_dimen=self.field_dimen, 
            sigma=self.sigma, 
            sharpness=self.sharpness
        )

        self.root = tk.Tk()
        self.root.title("Pure Pipeline Pitch Control Reconstruction")
        self.root.state("zoomed")

        self.control_frame = tk.Frame(self.root, bg="#e1e1e1", height=50)
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Label(self.control_frame, text="Enter Event ID:", bg="#e1e1e1", font=("Arial", 12)).pack(side=tk.LEFT, padx=10)

        self.entry_var = tk.StringVar(value="1")
        self.quality_var = tk.IntVar(value=150)
        self.entry = tk.Entry(self.control_frame, textvariable=self.entry_var, font=("Arial", 12), width=10)
        self.entry.pack(side=tk.LEFT, padx=10)

        tk.Label(self.control_frame, text="Quality / Search Depth:", bg="#e1e1e1", font=("Arial", 11)).pack(side=tk.LEFT, padx=(8, 4))
        self.quality_scale = tk.Scale(
            self.control_frame,
            from_=25,
            to=1000,
            orient=tk.HORIZONTAL,
            length=260,
            resolution=1,
            variable=self.quality_var,
            command=self._on_quality_change,
            bg="#e1e1e1",
            highlightthickness=0,
        )
        self.quality_scale.pack(side=tk.LEFT, padx=(0, 8))

        self.quality_info_label = tk.Label(self.control_frame, text="", bg="#e1e1e1", fg="#555", font=("Arial", 9))
        self.quality_info_label.pack(side=tk.LEFT, padx=(0, 12))

        self.btn_load = tk.Button(
            self.control_frame,
            text="Analyze Event",
            command=self.load_event,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 11, "bold"),
        )
        self.btn_load.pack(side=tk.LEFT, padx=10)

        self.status_label = tk.Label(self.control_frame, text="Ready", bg="#e1e1e1", fg="blue", font=("Arial", 10, "italic"))
        self.status_label.pack(side=tk.LEFT, padx=20)

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=1)

        self.canvas = tk.Canvas(self.main_frame, bg="#1a1a1a")
        self.v_scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)

        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)

        self.content_frame = tk.Frame(self.canvas, bg="#1a1a1a")
        self.canvas_window = self.canvas.create_window((0, 0), window=self.content_frame, anchor="nw")

        self.content_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self.current_figures = []
        self.analysis_running = False
        self._grid_cache = {}
        pool_workers = max(2, min(8, (os.cpu_count() or 4) - 1))
        self._pair_pool = ThreadPoolExecutor(max_workers=pool_workers)
        self._event_static_cache = {}
        self._event_result_cache = {}
        self._event_warm_start = {}
        self._apply_quality_profile(self.quality_var.get())

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        try:
            self._pair_pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        self.root.destroy()

    def _apply_quality_profile(self, q):
        q = int(np.clip(q, 25, 1000))
        self.refine_iterations = int(15 + q // 15)
        self.damping = 0.5
        self.step = 1.0
        # Multi-starts safeguard against player overlapping
        self.n_starts = 50

        self.quality_info_label.config(
            text=(
                f"max_iters={self.refine_iterations} "
                f"damping={self.damping:.1e} "
                f"starts={self.n_starts}"
            )
        )

    def _on_quality_change(self, _value):
        self._apply_quality_profile(self.quality_var.get())

    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=max(event.width, self.content_frame.winfo_reqwidth()))

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def update_status(self, text):
        self.status_label.config(text=text)
        self.root.update()

    def _get_grid_cache(self, n_grid_cells_x):
        key = int(n_grid_cells_x)
        cached = self._grid_cache.get(key)
        if cached is not None:
            return cached

        n_grid_cells_y = int(key * self.field_dimen[1] / self.field_dimen[0])
        dx = self.field_dimen[0] / key
        dy = self.field_dimen[1] / n_grid_cells_y
        xgrid = np.arange(key) * dx - self.field_dimen[0] / 2.0 + dx / 2.0
        ygrid = np.arange(n_grid_cells_y) * dy - self.field_dimen[1] / 2.0 + dy / 2.0
        xx, yy = np.meshgrid(xgrid, ygrid)
        grid_coords = np.column_stack([xx.ravel(), yy.ravel()])

        G_tilde = self.lstsq_solver.fgg._lift_coords(grid_coords)
        cached = {
            "y_len": len(ygrid),
            "grid_coords": grid_coords,
            "G_tilde": G_tilde,
        }
        self._grid_cache[key] = cached
        return cached

    def _compute_error_metrics(self, ppcf_original, ppcf_reconstructed):
        abs_err = np.abs(ppcf_original - ppcf_reconstructed)
        mae = float(np.mean(abs_err))
        rmse = float(np.sqrt(np.mean((ppcf_original - ppcf_reconstructed) ** 2)))
        rel_weighted = float((np.sum(abs_err) / (np.sum(np.abs(ppcf_original)) + 1e-8)) * 100.0)
        rel_pointwise = float(np.mean(abs_err / (np.abs(ppcf_original) + 0.05)) * 100.0)

        return {
            "mae_pct_points": mae * 100.0,
            "rmse_pct_points": rmse * 100.0,
            "avg_relative_error_pct": rel_weighted,
            "avg_relative_error_stabilized_pct": rel_pointwise,
            "abs_error_map_pct_points": abs_err * 100.0,
            "avg_absolute_error_raw": mae,
            "avg_relative_error_raw": rel_weighted / 100.0,
        }

    def _compute_event_result(self, event_id):
        quality_key = (
            int(self.n_grid_cells_x),
            int(self.refine_iterations),
            float(self.damping),
            int(self.n_starts),
        )
        result_key = (int(event_id), quality_key)
        cached_result = self._event_result_cache.get(result_key)
        if cached_result is not None:
            return cached_result

        start_time = time.time()
        
        # =========================================================================
        # 1. GENERATE TRUE ORIGINAL HEATMAP (LEFT MAP)
        # =========================================================================
        event_info = self.events.loc[event_id]
        pass_frame = event_info["Start Frame"]
        pass_team = event_info.Team

        static_key = (int(event_id), int(self.n_grid_cells_x))
        static_cached = self._event_static_cache.get(static_key)
        
        if static_cached is None:
            # Generate the true Metrica map!
            ppcf_original, _, _ = mpc.generate_pitch_control_for_event_linearized_matrixform(
                event_id, self.events, self.tracking_home, self.tracking_away, self.params, self.GK_numbers,
                field_dimen=self.field_dimen, n_grid_cells_x=self.n_grid_cells_x, offsides=False,
            )
            
            # Find closest tracking rows to extract ground truth locations
            if pass_frame in self.tracking_home.index:
                h_row_orig = self.tracking_home.loc[pass_frame]
                a_row_orig = self.tracking_away.loc[pass_frame]
            else:
                h_row_orig = self.tracking_home.iloc[(self.tracking_home.index - pass_frame).abs().argmin()]
                a_row_orig = self.tracking_away.iloc[(self.tracking_away.index - pass_frame).abs().argmin()]
                
            att_team_name = "Home" if pass_team == "Home" else "Away"
            def_team_name = "Away" if pass_team == "Home" else "Home"
            
            att_players = mpc.initialise_players(h_row_orig if pass_team == "Home" else a_row_orig, att_team_name, self.params, self.GK_numbers[0] if pass_team == "Home" else self.GK_numbers[1])
            def_players = mpc.initialise_players(a_row_orig if pass_team == "Home" else h_row_orig, def_team_name, self.params, self.GK_numbers[1] if pass_team == "Home" else self.GK_numbers[0])

            att_pos_orig = np.array([[p.position[0], p.position[1]] for p in att_players], dtype=float)
            def_pos_orig = np.array([[p.position[0], p.position[1]] for p in def_players], dtype=float)
            
            static_cached = {
                "ppcf_original": ppcf_original,
                "pass_team": pass_team,
                "att_pos_orig": att_pos_orig,
                "def_pos_orig": def_pos_orig,
                "att_players": att_players,
                "def_players": def_players,
                "h_row_orig": h_row_orig.copy(),
                "a_row_orig": a_row_orig.copy(),
                "att_team_name": att_team_name,
                "def_team_name": def_team_name
            }
            self._event_static_cache[static_key] = static_cached
        else:
            ppcf_original = static_cached["ppcf_original"]
            pass_team = static_cached["pass_team"]
            att_pos_orig = static_cached["att_pos_orig"]
            def_pos_orig = static_cached["def_pos_orig"]
            att_players = static_cached["att_players"]
            def_players = static_cached["def_players"]
            h_row_orig = static_cached["h_row_orig"]
            a_row_orig = static_cached["a_row_orig"]
            att_team_name = static_cached["att_team_name"]
            def_team_name = static_cached["def_team_name"]

        n_att = len(att_pos_orig)
        n_def = len(def_pos_orig)


        seed_base = int(event_id) * 9973
        warm_key = (int(event_id), int(self.n_grid_cells_x), int(n_att), int(n_def))
        warm_start = self._event_warm_start.get(warm_key)

        best_ppcf_mae = float('inf')
        best_att_recon = None
        best_def_recon = None
        best_ppcf_recon = None

        ppcf_clean = np.nan_to_num(ppcf_original, nan=0.5)
        ppcf_clean = np.clip(ppcf_clean, 0.0, 1.0)

        futures_list = []
        for run_idx in range(self.n_starts):
            if run_idx == 0 and warm_start is not None:
                init_att, init_def = warm_start["att"], warm_start["def"]
            else:
                init_att, init_def = None, None

            # Submit the unified outer loop to the thread pool
            # Note: ppcf_original is the raw unified probability heatmap [0.0 - 1.0]
            fut = self._pair_pool.submit(
                self.lstsq_solver.solve_collective_alternating, 
                collective_heatmap=ppcf_clean, 
                n_att=n_att, 
                n_def=n_def, 
                init_att=init_att, 
                init_def=init_def,
                outer_max_iterations=self.refine_iterations,                  # Adjust as needed
                inner_max_iterations=self.refine_iterations,
                damping=self.damping, 
                step=self.step,
                epsilon=1e-6,                            # Exposed parameter
                seed=17 + seed_base + run_idx
            )
            futures_list.append(fut)

        # =========================================================================
        # 4. INJECT PREDICTED X, Y INTO A COMPLETELY PURGED MOCK TRACKING FRAME
        # =========================================================================
        for fut in futures_list:
            sol = fut.result()
            att_pos_predicted = sol["att_xy"]
            def_pos_predicted = sol["def_xy"]

            h_row_mock = h_row_orig.copy()
            a_row_mock = a_row_orig.copy()
            
            # ---> THE PURGE: Destroy ALL original locations and velocities.
            # Wipe X and Y to NaN so missing players don't accidentally get calculated.
            # Wipe Velocities to 0.0 to guarantee pure linear isotropy.
            for df_mock in (h_row_mock, a_row_mock):
                for col in df_mock.index:
                    if col.endswith('_x') or col.endswith('_y'):
                        df_mock[col] = np.nan
                    elif col.endswith('_vx') or col.endswith('_vy') or col.endswith('_speed'):
                        df_mock[col] = 0.0

            # ---> SECURE INJECTION:
            # Inject only the mathematically predicted coordinates. 
            cost_att = np.linalg.norm(att_pos_orig[:, None, :] - att_pos_predicted[None, :, :], axis=2)
            row_a, col_a = linear_sum_assignment(cost_att)
            for r_idx, c_idx in zip(row_a, col_a):
                p_id = att_players[r_idx].id
                x, y = att_pos_predicted[c_idx]
                prefix = "Home" if att_team_name == "Home" else "Away"
                if prefix == "Home":
                    h_row_mock[f"Home_{p_id}_x"], h_row_mock[f"Home_{p_id}_y"] = x, y
                else:
                    a_row_mock[f"Away_{p_id}_x"], a_row_mock[f"Away_{p_id}_y"] = x, y

            cost_def = np.linalg.norm(def_pos_orig[:, None, :] - def_pos_predicted[None, :, :], axis=2)
            row_d, col_d = linear_sum_assignment(cost_def)
            for r_idx, c_idx in zip(row_d, col_d):
                p_id = def_players[r_idx].id
                x, y = def_pos_predicted[c_idx]
                prefix = "Home" if def_team_name == "Home" else "Away"
                if prefix == "Home":
                    h_row_mock[f"Home_{p_id}_x"], h_row_mock[f"Home_{p_id}_y"] = x, y
                else:
                    a_row_mock[f"Away_{p_id}_x"], a_row_mock[f"Away_{p_id}_y"] = x, y

            trk_home_mock = self.tracking_home.copy()
            trk_away_mock = self.tracking_away.copy()
            trk_home_mock.loc[pass_frame] = h_row_mock
            trk_away_mock.loc[pass_frame] = a_row_mock

            # THE PURE PIPELINE: Pass the clean, purged mock data back into Metrica
            ppcf_cand, _, _ = mpc.generate_pitch_control_for_event_linearized_matrixform(
                event_id, self.events, trk_home_mock, trk_away_mock, self.params, self.GK_numbers,
                field_dimen=self.field_dimen, n_grid_cells_x=self.n_grid_cells_x, offsides=False,
            )

            metrics_candidate = self._compute_error_metrics(ppcf_original, ppcf_cand)

            if metrics_candidate["avg_absolute_error_raw"] < best_ppcf_mae:
                best_ppcf_mae = metrics_candidate["avg_absolute_error_raw"]
                best_att_recon = att_pos_predicted.copy()
                best_def_recon = def_pos_predicted.copy()
                best_ppcf_recon = ppcf_cand

        metrics = self._compute_error_metrics(ppcf_original, best_ppcf_recon)
        ppcf_err_pct = metrics["abs_error_map_pct_points"]
        att_dist = assignment_distances(att_pos_orig, best_att_recon)
        def_dist = assignment_distances(def_pos_orig, best_def_recon)
        elapsed = time.time() - start_time

        result = {
            "event_id": event_id,
            "pass_team": pass_team,
            "ppcf_original": ppcf_original,
            "ppcf_reconstructed": best_ppcf_recon,
            "ppcf_err_pct": ppcf_err_pct,
            "metrics": metrics,
            "att_orig_pos": att_pos_orig,
            "def_orig_pos": def_pos_orig,
            "att_recon_pos": best_att_recon,
            "def_recon_pos": best_def_recon,
            "att_dist": att_dist,
            "def_dist": def_dist,
            "elapsed": elapsed,
        }
        self._event_result_cache[result_key] = result
        self._event_warm_start[warm_key] = {"att": best_att_recon.copy(), "def": best_def_recon.copy()}
        return result

    def _on_analysis_done(self, result):
        self.analysis_running = False
        self.btn_load.config(state=tk.NORMAL)

        if "error" in result:
            self.update_status(f"Error: {result['error']}")
            print(f"\nCRITICAL PIPELINE ERROR:\n{result['traceback']}")
            return

        self.render_plots(
            result["event_id"], result["pass_team"], result["ppcf_original"], result["ppcf_reconstructed"],
            result["ppcf_err_pct"], result["metrics"], result["att_orig_pos"], result["def_orig_pos"],
            result["att_recon_pos"], result["def_recon_pos"], result["att_dist"], result["def_dist"],
        )
        self.update_status(f"Comparison complete for Event {result['event_id']} in {result['elapsed']:.2f}s")

    def _analyze_event_worker(self, event_id):
        try:
            self.root.after(0, lambda: self.update_status("Computing Gauss-Newton Matrix updates..."))
            result = self._compute_event_result(event_id)
            self.root.after(0, lambda: self._on_analysis_done(result))
        except Exception as exc:
            err_trace = traceback.format_exc()
            self.root.after(0, lambda: self._on_analysis_done({"error": str(exc), "traceback": err_trace}))

    def load_event(self):
        if self.analysis_running:
            self.update_status("Analysis already running. Please wait...")
            return

        try:
            event_id = int(self.entry_var.get())
            if event_id not in self.events.index:
                self.update_status(f"Error: Event {event_id} not found!")
                return
        except ValueError:
            self.update_status("Error: Please enter a numeric Event ID")
            return

        for widget in self.content_frame.winfo_children():
            widget.destroy()
        for fig in self.current_figures:
            plt.close(fig)
        self.current_figures = []

        self.analysis_running = True
        self.btn_load.config(state=tk.DISABLED)
        self.update_status(f"Analyzing Event {event_id} via Gauss-Newton LSTSQ...")

        worker = threading.Thread(target=self._analyze_event_worker, args=(event_id,), daemon=True)
        worker.start()

    def render_plots(
        self, event_id, pass_team, ppcf_original, ppcf_reconstructed, ppcf_err_pct, metrics,
        att_orig_pos, def_orig_pos, att_recon_pos, def_recon_pos, att_dist, def_dist,
    ):
        screen_width_px = self.root.winfo_screenwidth()
        screen_height_px = self.root.winfo_screenheight()
        dpi = 96
        fig_width = (screen_width_px * 0.95) / dpi
        fig_height = (screen_height_px * 0.9) / dpi

        fig, axes = plt.subplots(2, 3, figsize=(fig_width, fig_height), dpi=dpi)
        fig.patch.set_facecolor("#2d5016")
        self.current_figures.append(fig)

        x_min, x_max = -self.field_dimen[0] / 2.0, self.field_dimen[0] / 2.0
        y_min, y_max = -self.field_dimen[1] / 2.0, self.field_dimen[1] / 2.0

        err_cap = np.percentile(ppcf_err_pct, 99)
        if not np.isfinite(err_cap) or err_cap <= 0:
            err_cap = 1.0

        ppcf_original_vis = np.clip(ppcf_original, 0.0, 1.0)
        ppcf_reconstructed_vis = np.clip(ppcf_reconstructed, 0.0, 1.0)
        
        # Display smoothing to avoid blocky matrix rendering edges
        for vis_arr in [ppcf_original_vis, ppcf_reconstructed_vis]:
            vis_arr[:] = (
                np.pad(vis_arr, ((1, 1), (1, 1)), mode="reflect")[:-2, :-2]
                + np.pad(vis_arr, ((1, 1), (1, 1)), mode="reflect")[:-2, 1:-1]
                + np.pad(vis_arr, ((1, 1), (1, 1)), mode="reflect")[:-2, 2:]
                + np.pad(vis_arr, ((1, 1), (1, 1)), mode="reflect")[1:-1, :-2]
                + np.pad(vis_arr, ((1, 1), (1, 1)), mode="reflect")[1:-1, 1:-1]
                + np.pad(vis_arr, ((1, 1), (1, 1)), mode="reflect")[1:-1, 2:]
                + np.pad(vis_arr, ((1, 1), (1, 1)), mode="reflect")[2:, :-2]
                + np.pad(vis_arr, ((1, 1), (1, 1)), mode="reflect")[2:, 1:-1]
                + np.pad(vis_arr, ((1, 1), (1, 1)), mode="reflect")[2:, 2:]
            ) / 9.0

        plots = [
            (ppcf_original_vis, "Original Matrix PPCF", "RdBu_r", 0.0, 1.0),
            (ppcf_reconstructed_vis, "Reconstructed Pure Matrix PPCF", "RdBu_r", 0.0, 1.0),
            (np.clip(ppcf_err_pct, 0.0, err_cap), "PPCF Abs Error (pp)", "hot", 0.0, err_cap),
        ]

        for idx, (ax, (data, title, cmap, vmin, vmax)) in enumerate(zip(axes[0], plots)):
            ax.set_facecolor("#3d7d21")
            ax.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], color="white", linewidth=2)
            ax.plot([0, 0], [y_min, y_max], color="white", linewidth=2)
            ax.add_patch(plt.Circle((0, 0), 9.15, fill=False, color="white", linewidth=2))

            im = ax.imshow(
                np.flipud(data), extent=[x_min, x_max, y_min, y_max], cmap=cmap, alpha=0.78,
                interpolation="bicubic", aspect="auto", vmin=vmin, vmax=vmax,
            )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title, color="white", fontsize=11, fontweight="bold")
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.axis("off")

            if idx == 2:
                txt = (
                    f"Event: {event_id} ({pass_team} in possession)\n"
                    f"Solver: Gauss-Newton Pure Matrix Injection\n"
                    f"Heatmap Avg Abs Error (MAE): {metrics['mae_pct_points']:.2f} pp\n"
                    f"Heatmap Avg Rel Error: {metrics['avg_relative_error_pct']:.2f} %\n"
                    f"Heatmap RMSE: {metrics['rmse_pct_points']:.2f} pp"
                )
                ax.text(
                    0.03, 0.97, txt, transform=ax.transAxes, fontsize=9, fontfamily="monospace",
                    va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
                )

        ax_attack_recon = axes[1, 0]
        ax_defense_recon = axes[1, 1]
        ax_text = axes[1, 2]

        for ax in (ax_attack_recon, ax_defense_recon):
            ax.set_facecolor("#3d7d21")
            ax.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], color="white", linewidth=2)
            ax.plot([0, 0], [y_min, y_max], color="white", linewidth=2)
            ax.add_patch(plt.Circle((0, 0), 9.15, fill=False, color="white", linewidth=2))
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.axis("off")

        ax_attack_recon.scatter(att_orig_pos[:, 0], att_orig_pos[:, 1], c="#ffb3b3", s=58, marker="o", label="Original")
        ax_attack_recon.scatter(att_recon_pos[:, 0], att_recon_pos[:, 1], c="#ff4d4d", s=65, marker="^", label="Reconstructed")
        ax_attack_recon.legend(loc="upper right", fontsize=8, framealpha=0.85)
        ax_attack_recon.set_title("Attack Locations: Original vs Least Squares", color="white", fontsize=11, fontweight="bold")

        ax_defense_recon.scatter(def_orig_pos[:, 0], def_orig_pos[:, 1], c="#c6dbef", s=58, marker="o", label="Original")
        ax_defense_recon.scatter(def_recon_pos[:, 0], def_recon_pos[:, 1], c="#2171b5", s=65, marker="^", label="Reconstructed")
        ax_defense_recon.legend(loc="upper right", fontsize=8, framealpha=0.85)
        ax_defense_recon.set_title("Defense Locations: Original vs Least Squares", color="white", fontsize=11, fontweight="bold")

        ax_text.set_facecolor("#2d5016")
        ax_text.axis("off")
        lines = [
            "PLAYER LOCATION SUMMARY",
            "(Inversion solved purely from influence matrices)",
            "",
            f"Attack tracking alignment: {np.mean(att_dist):.2f} +/- {np.std(att_dist):.2f} m",
            f"Defense tracking alignment: {np.mean(def_dist):.2f} +/- {np.std(def_dist):.2f} m",
            "",
            "Attack reconstructed coordinates:",
        ]
        for i in range(min(8, len(att_recon_pos))):
            lines.append(f"A{i+1:02d}: ({att_recon_pos[i,0]:6.1f}, {att_recon_pos[i,1]:6.1f})")
        lines.append("")
        lines.append("Defense reconstructed coordinates:")
        for i in range(min(8, len(def_recon_pos))):
            lines.append(f"D{i+1:02d}: ({def_recon_pos[i,0]:6.1f}, {def_recon_pos[i,1]:6.1f})")
        lines.append("")
        lines.append(f"Avg Absolute Error: {metrics['avg_absolute_error_raw']:.5f}")
        lines.append(f"Avg Relative Error: {metrics['avg_relative_error_raw']:.5f}")

        ax_text.text(0.02, 0.98, "\n".join(lines), color="white", fontsize=8.5, va="top", fontfamily="monospace")

        plt.tight_layout(pad=1.2)

        canvas = FigureCanvasTkAgg(fig, master=self.content_frame)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def run(self):
        self.root.mainloop()


def main():
    print("=" * 60)
    print("PITCH CONTROL RECONSTRUCTION COMPARISON - PURE MATRICES GUI")
    print("=" * 60)

    game_id = 3812
    data_path = MATCH_DATA_PATH
    field_dimen = (106.0, 68.0)

    print("\nLoading tracking and event data...")
    tracking_home, tracking_away, events = mio.load_new_data(data_path, game_id)

    print("Converting coordinates and calculating velocities...")
    tracking_home = to_metric_coordinates(tracking_home, field_dimen)
    tracking_away = to_metric_coordinates(tracking_away, field_dimen)

    events["Start X"] = (events["Start X"] - 0.5) * field_dimen[0]
    events["Start Y"] = (events["Start Y"] - 0.5) * field_dimen[1]

    tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True)
    tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)

    home_gk, away_gk, home_players, away_players = find_goalkeepers(tracking_home, tracking_away)
    GK_numbers = (home_gk, away_gk)
    params = mpc.default_model_params()

    print("Data ready. Launching Least Squares reconstruction comparison GUI...")

    app = PitchControlReconstructionGUI(
        events, tracking_home, tracking_away, params, GK_numbers, home_players, away_players, field_dimen,
    )

    pass_events = events[events["Type"] == "PASS"]
    if not pass_events.empty:
        app.entry_var.set(str(pass_events.index[0]))

    app.run()


if __name__ == "__main__":
    main()