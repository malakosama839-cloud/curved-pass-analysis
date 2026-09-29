"""
Pitch Control Formation Matcher GUI
Pipeline:
1. Takes a Target Event (The tactical destination / goal heatmap).
2. Generates standard 11-man formation templates (4-3-3, 4-2-3-1, etc).
3. Uses Gauss-Newton to bend and stretch the selected formation to match the target heatmap.
4. Generates reconstructed Right Map using pure structural (no velocity) tracking data.
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

from reverse_pitch_control_lstsq import LeastSquaresPitchControlSolver

# --- PATH SETUP ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
LAURIE_PATH = os.path.join(BASE_DIR, "LaurieOnTracking")
MATCH_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "pff_match_data")

sys.path.insert(0, LAURIE_PATH)

import Metrica_IO as mio
import Metrica_Velocities as mvel  # <--- FIX: Brought velocities back to satisfy the player object
import Metrica_PitchControl as mpc


# --- FORMATION TEMPLATES ---
# Normalized coordinates: X (0.0=Goal Line, 1.0=Opponent Goal), Y (0.0=Left, 1.0=Right)
FORMATIONS = {
    "4-4-2": [[0.05, 0.5], [0.25, 0.2], [0.25, 0.4], [0.25, 0.6], [0.25, 0.8], [0.5, 0.2], [0.5, 0.4], [0.5, 0.6], [0.5, 0.8], [0.75, 0.35], [0.75, 0.65]],
    "4-3-3": [[0.05, 0.5], [0.25, 0.2], [0.25, 0.4], [0.25, 0.6], [0.25, 0.8], [0.5, 0.3], [0.5, 0.5], [0.5, 0.7], [0.75, 0.2], [0.75, 0.5], [0.75, 0.8]],
    "4-2-3-1": [[0.05, 0.5], [0.2, 0.2], [0.2, 0.4], [0.2, 0.6], [0.2, 0.8], [0.4, 0.35], [0.4, 0.65], [0.6, 0.2], [0.6, 0.5], [0.6, 0.8], [0.8, 0.5]],
    "3-5-2": [[0.05, 0.5], [0.25, 0.3], [0.25, 0.5], [0.25, 0.7], [0.5, 0.1], [0.5, 0.3], [0.5, 0.5], [0.5, 0.7], [0.5, 0.9], [0.75, 0.35], [0.75, 0.65]],
    "5-3-2": [[0.05, 0.5], [0.2, 0.1], [0.2, 0.3], [0.2, 0.5], [0.2, 0.7], [0.2, 0.9], [0.5, 0.3], [0.5, 0.5], [0.5, 0.7], [0.75, 0.35], [0.75, 0.65]]
}

def get_formation_positions(formation_name, team_size, defend_left=True):
    coords = np.array(FORMATIONS[formation_name])
    # Scale to pitch limits (-45 to +35 in X, -30 to +30 in Y)
    x_actual = (coords[:, 0] * 80) - 45 
    y_actual = (coords[:, 1] - 0.5) * 60

    if not defend_left:
        x_actual = -x_actual
        y_actual = -y_actual

    res = np.column_stack((x_actual, y_actual))
    return res[:team_size] # Gracefully handle red cards (team size < 11)


# --- HELPER FUNCTIONS ---
def to_metric_coordinates(data, field_dimen=(106.0, 68.0)):
    x_columns = [c for c in data.columns if c[-2:] == "_x" and c[:4] in ["Home", "Away"]]
    y_columns = [c for c in data.columns if c[-2:] == "_y" and c[:4] in ["Home", "Away"]]
    data[x_columns] = (data[x_columns] - 0.5) * field_dimen[0]
    data[y_columns] = (data[y_columns] - 0.5) * field_dimen[1]
    return data

def find_goalkeepers(tracking_home, tracking_away):
    home_players = sorted(set([c.split("_")[1] for c in tracking_home.columns if c.startswith("Home_") and c.endswith("_x")]))
    away_players = sorted(set([c.split("_")[1] for c in tracking_away.columns if c.startswith("Away_") and c.endswith("_x")]))
    home_avg_x = {p: tracking_home[f"Home_{p}_x"].mean() for p in home_players}
    away_avg_x = {p: tracking_away[f"Away_{p}_x"].mean() for p in away_players}
    return min(home_avg_x, key=home_avg_x.get), max(away_avg_x, key=away_avg_x.get), home_players, away_players


class PitchControlFormationGUI:
    def __init__(self, events, tracking_home, tracking_away, params, GK_numbers, home_players, away_players, field_dimen):
        self.n_grid_cells_x = 64
        self.damping = 1e-2
        self.step = 0.4
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

        self.lstsq_solver = LeastSquaresPitchControlSolver(
            field_dimen=self.field_dimen, sigma=self.sigma, sharpness=self.sharpness
        )

        self.root = tk.Tk()
        self.root.title("Gauss-Newton Formation Matcher")
        self.root.state("zoomed")

        self.control_frame = tk.Frame(self.root, bg="#e1e1e1", height=50)
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # TARGET EVENT
        tk.Label(self.control_frame, text="Target Event:", bg="#e1e1e1", font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=(10, 2))
        self.entry_target_var = tk.StringVar(value="2")
        tk.Entry(self.control_frame, textvariable=self.entry_target_var, font=("Arial", 11), width=5).pack(side=tk.LEFT, padx=(0, 15))

        # ATTACK FORMATION
        tk.Label(self.control_frame, text="Attack Formation:", bg="#e1e1e1", font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=(5, 2))
        self.att_form_var = tk.StringVar(value="4-3-3")
        self.att_form_combo = ttk.Combobox(self.control_frame, textvariable=self.att_form_var, values=["Original", "4-4-2", "4-3-3", "4-2-3-1", "3-5-2", "5-3-2"], state="readonly", width=10)
        self.att_form_combo.pack(side=tk.LEFT, padx=(0, 15))

        # DEFEND FORMATION
        tk.Label(self.control_frame, text="Defend Formation:", bg="#e1e1e1", font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=(5, 2))
        self.def_form_var = tk.StringVar(value="4-4-2")
        self.def_form_combo = ttk.Combobox(self.control_frame, textvariable=self.def_form_var, values=["Original", "4-4-2", "4-3-3", "4-2-3-1", "3-5-2", "5-3-2"], state="readonly", width=10)
        self.def_form_combo.pack(side=tk.LEFT, padx=(0, 15))

        # SHAPE FLEXIBILITY
        tk.Label(self.control_frame, text="Shape Flexibility:", bg="#e1e1e1", font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=(5, 2))
        self.adapt_var = tk.StringVar(value="Flexible")
        self.adapt_combo = ttk.Combobox(self.control_frame, textvariable=self.adapt_var, values=["Strict", "Flexible", "Full Snap"], state="readonly", width=10)
        self.adapt_combo.pack(side=tk.LEFT, padx=(0, 15))

        self.btn_load = tk.Button(self.control_frame, text="Run Matcher", command=self.load_event, bg="#4CAF50", fg="white", font=("Arial", 11, "bold"))
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
        self.content_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(self.canvas_window, width=max(e.width, self.content_frame.winfo_reqwidth())))
        self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        self.current_figures = []
        self.analysis_running = False
        self._grid_cache = {}
        self._pair_pool = ThreadPoolExecutor(max_workers=2)
        self._event_static_cache = {}

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        try:
            self._pair_pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        self.root.destroy()

    def update_status(self, text):
        self.status_label.config(text=text)
        self.root.update()

    def _get_grid_cache(self, n_grid_cells_x):
        key = int(n_grid_cells_x)
        if key in self._grid_cache: return self._grid_cache[key]
        n_grid_cells_y = int(key * self.field_dimen[1] / self.field_dimen[0])
        dx = self.field_dimen[0] / key
        dy = self.field_dimen[1] / n_grid_cells_y
        xgrid = np.arange(key) * dx - self.field_dimen[0] / 2.0 + dx / 2.0
        ygrid = np.arange(n_grid_cells_y) * dy - self.field_dimen[1] / 2.0 + dy / 2.0
        xx, yy = np.meshgrid(xgrid, ygrid)
        grid_coords = np.column_stack([xx.ravel(), yy.ravel()])
        G_tilde = self.lstsq_solver.fgg._lift_coords(grid_coords)
        self._grid_cache[key] = {"y_len": len(ygrid), "grid_coords": grid_coords, "G_tilde": G_tilde}
        return self._grid_cache[key]

    def _safe_extract_row(self, df, frame):
        """Safely extracts a single frame, perfectly handling Metrica duplicate-frame bugs!"""
        if frame in df.index:
            row = df.loc[frame]
        else:
            row = df.iloc[(df.index - frame).abs().argmin()]
        
        # If dataset glitched and returned a DataFrame, force it to be a Series
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row

    def _extract_team_positions(self, event_id, target_possession_team):
        event_info = self.events.loc[event_id]
        pass_frame = event_info["Start Frame"]
        
        h_row = self._safe_extract_row(self.tracking_home, pass_frame)
        a_row = self._safe_extract_row(self.tracking_away, pass_frame)

        att_team_name = "Home" if target_possession_team == "Home" else "Away"
        def_team_name = "Away" if target_possession_team == "Home" else "Home"

        att_players = mpc.initialise_players(h_row if att_team_name == "Home" else a_row, att_team_name, self.params, self.GK_numbers[0] if att_team_name == "Home" else self.GK_numbers[1])
        def_players = mpc.initialise_players(a_row if def_team_name == "Away" else h_row, def_team_name, self.params, self.GK_numbers[1] if def_team_name == "Away" else self.GK_numbers[0])

        att_pos = np.array([[p.position[0], p.position[1]] for p in att_players], dtype=float)
        def_pos = np.array([[p.position[0], p.position[1]] for p in def_players], dtype=float)
        return att_pos, def_pos, att_players, def_players, h_row, a_row, att_team_name, def_team_name

    def _compute_event_result(self, target_event_id):
        start_time = time.time()
        
        target_info = self.events.loc[target_event_id]
        target_pass_team = target_info.Team
        target_frame = target_info["Start Frame"]

        if target_event_id not in self._event_static_cache:
            ppcf_target, _, _ = mpc.generate_pitch_control_for_event_linearized_matrixform(
                target_event_id, self.events, self.tracking_home, self.tracking_away, self.params, self.GK_numbers,
                field_dimen=self.field_dimen, n_grid_cells_x=self.n_grid_cells_x, offsides=False,
            )
            t_att_pos, t_def_pos, t_att_p, t_def_p, t_h_row, t_a_row, t_att_name, t_def_name = self._extract_team_positions(target_event_id, target_pass_team)
            
            grid = self._get_grid_cache(self.n_grid_cells_x)
            G_tilde = grid["G_tilde"]
            y_len = grid["y_len"]

            att_target_map = self.lstsq_solver.fgg.generate_heatmap_from_lifted_grid(G_tilde, t_att_pos, sharpness=self.sharpness).reshape(y_len, self.n_grid_cells_x)
            def_target_map = self.lstsq_solver.fgg.generate_heatmap_from_lifted_grid(G_tilde, t_def_pos, sharpness=self.sharpness).reshape(y_len, self.n_grid_cells_x)

            self._event_static_cache[target_event_id] = {
                "ppcf": ppcf_target, "att_pos": t_att_pos, "def_pos": t_def_pos, "att_p": t_att_p, "def_p": t_def_p,
                "h_row": t_h_row, "a_row": t_a_row, "att_name": t_att_name, "def_name": t_def_name,
                "att_map": att_target_map, "def_map": def_target_map
            }

        t_data = self._event_static_cache[target_event_id]

        # Securely determine which direction the teams are defending
        att_gk_id = self.GK_numbers[0] if t_data["att_name"] == "Home" else self.GK_numbers[1]
        att_gks = [p for p in t_data["att_p"] if str(p.id) == str(att_gk_id)]
        att_defend_left = att_gks[0].position[0] < 0 if att_gks else True
        
        def_gk_id = self.GK_numbers[0] if t_data["def_name"] == "Home" else self.GK_numbers[1]
        def_gks = [p for p in t_data["def_p"] if str(p.id) == str(def_gk_id)]
        def_defend_left = def_gks[0].position[0] < 0 if def_gks else False

        # Generate Formation Initial Coordinates
        att_form = self.att_form_var.get()
        if att_form == "Original":
            init_att = t_data["att_pos"]
        else:
            init_att = get_formation_positions(att_form, len(t_data["att_pos"]), att_defend_left)

        def_form = self.def_form_var.get()
        if def_form == "Original":
            init_def = t_data["def_pos"]
        else:
            init_def = get_formation_positions(def_form, len(t_data["def_pos"]), def_defend_left)

        # Flexibility limits iterations
        adapt_mode = self.adapt_var.get()
        if adapt_mode == "Strict": iters = 3
        elif adapt_mode == "Flexible": iters = 18
        else: iters = 60 # Full Snap

        # --- RUN THE SOLVER ---
        fut_att = self._pair_pool.submit(
            self.lstsq_solver.solve, t_data["att_map"], n_players=len(init_att), init_positions=init_att,
            max_iterations=iters, damping=self.damping, step=self.step, seed=42
        )
        fut_def = self._pair_pool.submit(
            self.lstsq_solver.solve, t_data["def_map"], n_players=len(init_def), init_positions=init_def,
            max_iterations=iters, damping=self.damping, step=self.step, seed=42
        )
        
        sol_att = fut_att.result()
        sol_def = fut_def.result()

        # --- PURE STRUCTURAL PURGE & INJECT ---
        h_row_mock = t_data["h_row"].copy()
        a_row_mock = t_data["a_row"].copy()
        
        for df_mock in (h_row_mock, a_row_mock):
            for col in df_mock.index:
                if (col.startswith('Home_') or col.startswith('Away_')) and (col.endswith('_x') or col.endswith('_y')):
                    df_mock[col] = np.nan

        for true_pos, sol_pos, players, team_name in [
            (t_data["att_pos"], sol_att.players_xy, t_data["att_p"], t_data["att_name"]),
            (t_data["def_pos"], sol_def.players_xy, t_data["def_p"], t_data["def_name"])
        ]:
            cost = np.linalg.norm(true_pos[:, None, :] - sol_pos[None, :, :], axis=2)
            row_idx, col_idx = linear_sum_assignment(cost)
            for r, c in zip(row_idx, col_idx):
                p_id = players[r].id
                x, y = sol_pos[c]
                if team_name == "Home": 
                    h_row_mock[f"Home_{p_id}_x"] = x
                    h_row_mock[f"Home_{p_id}_y"] = y
                else: 
                    a_row_mock[f"Away_{p_id}_x"] = x
                    a_row_mock[f"Away_{p_id}_y"] = y

        trk_home_mock = pd.DataFrame([h_row_mock])
        trk_home_mock.index = [target_frame]
        
        trk_away_mock = pd.DataFrame([a_row_mock])
        trk_away_mock.index = [target_frame]

        ppcf_nudged, _, _ = mpc.generate_pitch_control_for_event_linearized_matrixform(
            target_event_id, self.events, trk_home_mock, trk_away_mock, self.params, self.GK_numbers,
            field_dimen=self.field_dimen, n_grid_cells_x=self.n_grid_cells_x, offsides=False,
        )

        abs_err = np.abs(t_data["ppcf"] - ppcf_nudged)
        mae = float(np.mean(abs_err))
        
        return {
            "target_id": target_event_id, "att_form": att_form, "def_form": def_form,
            "ppcf_target": t_data["ppcf"], "ppcf_nudged": ppcf_nudged, "abs_err_map": abs_err, "mae": mae,
            "t_att_pos": t_data["att_pos"], "t_def_pos": t_data["def_pos"],
            "s_att_pos": sol_att.players_xy, "s_def_pos": sol_def.players_xy,
            "elapsed": time.time() - start_time
        }

    def _on_analysis_done(self, result):
        self.analysis_running = False
        self.btn_load.config(state=tk.NORMAL)

        if "error" in result:
            self.update_status(f"Error: {result['error']}")
            print(f"\nCRITICAL PIPELINE ERROR:\n{result['traceback']}")
            return

        self.render_plots(result)
        self.update_status(f"Matcher Complete | Att: {result['att_form']} vs Def: {result['def_form']} | {result['elapsed']:.2f}s")

    def _analyze_event_worker(self, t_id):
        try:
            res = self._compute_event_result(t_id)
            self.root.after(0, lambda: self._on_analysis_done(res))
        except Exception as exc:
            err_trace = traceback.format_exc()
            err_msg = str(exc)
            self.root.after(0, lambda e=err_msg, t=err_trace: self._on_analysis_done({"error": e, "traceback": t}))

    def load_event(self):
        try:
            t_id = int(self.entry_target_var.get())
            if t_id not in self.events.index:
                self.update_status("Error: Event ID not found!")
                return
        except ValueError:
            self.update_status("Error: Please enter a valid Target Event ID")
            return

        self.analysis_running = True
        self.btn_load.config(state=tk.DISABLED)
        self.update_status("Bending formation to target heatmap...")
        threading.Thread(target=self._analyze_event_worker, args=(t_id,), daemon=True).start()

    def render_plots(self, res):
        for widget in self.content_frame.winfo_children(): widget.destroy()
        for fig in self.current_figures: plt.close(fig)
        self.current_figures.clear()

        fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=96)
        fig.patch.set_facecolor("#2d5016")
        self.current_figures.append(fig)

        x_min, x_max, y_min, y_max = -self.field_dimen[0]/2, self.field_dimen[0]/2, -self.field_dimen[1]/2, self.field_dimen[1]/2
        
        vis_target = res["ppcf_target"].copy()
        vis_nudged = res["ppcf_nudged"].copy()

        for vis_arr in [vis_target, vis_nudged]:
            vis_arr[:] = (
                np.pad(vis_arr, ((1, 1), (1, 1)), mode="reflect")[:-2, :-2] + np.pad(vis_arr, ((1, 1), (1, 1)), mode="reflect")[:-2, 1:-1] + np.pad(vis_arr, ((1, 1), (1, 1)), mode="reflect")[:-2, 2:] +
                np.pad(vis_arr, ((1, 1), (1, 1)), mode="reflect")[1:-1, :-2] + np.pad(vis_arr, ((1, 1), (1, 1)), mode="reflect")[1:-1, 1:-1] + np.pad(vis_arr, ((1, 1), (1, 1)), mode="reflect")[1:-1, 2:] +
                np.pad(vis_arr, ((1, 1), (1, 1)), mode="reflect")[2:, :-2] + np.pad(vis_arr, ((1, 1), (1, 1)), mode="reflect")[2:, 1:-1] + np.pad(vis_arr, ((1, 1), (1, 1)), mode="reflect")[2:, 2:]
            ) / 9.0

        plots = [
            (vis_target, f"Original Goal Map (Event {res['target_id']})", "RdBu_r", 0.0, 1.0),
            (vis_nudged, f"Formation Adjusted Map", "RdBu_r", 0.0, 1.0),
            (res["abs_err_map"], "Formation Compromise Error", "hot", 0.0, 0.5),
        ]

        for idx, (ax, (data, title, cmap, vmin, vmax)) in enumerate(zip(axes[0], plots)):
            ax.set_facecolor("#3d7d21")
            ax.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], color="white", linewidth=2)
            ax.plot([0, 0], [y_min, y_max], color="white", linewidth=2)
            ax.add_patch(plt.Circle((0, 0), 9.15, fill=False, color="white", linewidth=2))
            im = ax.imshow(np.flipud(data), extent=[x_min, x_max, y_min, y_max], cmap=cmap, alpha=0.78, interpolation="bicubic", aspect="auto", vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title, color="white", fontsize=11, fontweight="bold")
            ax.set_xlim([x_min, x_max]); ax.set_ylim([y_min, y_max]); ax.axis("off")

        for ax in axes[1, :2]:
            ax.set_facecolor("#3d7d21")
            ax.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], color="white", linewidth=2)
            ax.plot([0, 0], [y_min, y_max], color="white", linewidth=2)
            ax.add_patch(plt.Circle((0, 0), 9.15, fill=False, color="white", linewidth=2))
            ax.set_xlim([x_min, x_max]); ax.set_ylim([y_min, y_max]); ax.axis("off")

        axes[1,0].scatter(res["t_att_pos"][:, 0], res["t_att_pos"][:, 1], c="#ffb3b3", s=58, marker="o", label="Original Event")
        axes[1,0].scatter(res["s_att_pos"][:, 0], res["s_att_pos"][:, 1], c="#ff4d4d", s=65, marker="^", label=f"Formation ({res['att_form']})")
        axes[1,0].legend(); axes[1,0].set_title("Attack Positions", color="white")

        axes[1,1].scatter(res["t_def_pos"][:, 0], res["t_def_pos"][:, 1], c="#c6dbef", s=58, marker="o", label="Original Event")
        axes[1,1].scatter(res["s_def_pos"][:, 0], res["s_def_pos"][:, 1], c="#2171b5", s=65, marker="^", label=f"Formation ({res['def_form']})")
        axes[1,1].legend(); axes[1,1].set_title("Defense Positions", color="white")

        axes[1,2].set_facecolor("#2d5016"); axes[1,2].axis("off")
        txt = (f"FORMATION MATCHER METRICS\n\nMap Mean Abs Error: {res['mae'] * 100:.2f} %\n\n"
               "The solver initialized players in the requested\nformations and mathematically 'bent' their shape\ntowards the target tactical heatmap influence.")
        axes[1,2].text(0.1, 0.8, txt, color="white", fontsize=11, fontfamily="monospace", va="top")

        plt.tight_layout(pad=1.2)
        canvas = FigureCanvasTkAgg(fig, master=self.content_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def run(self):
        self.root.mainloop()

def main():
    print("Launching Formation Matcher GUI...")
    tracking_home, tracking_away, events = mio.load_new_data(MATCH_DATA_PATH, 3812)
    tracking_home = to_metric_coordinates(tracking_home)
    tracking_away = to_metric_coordinates(tracking_away)
    
    # FIX: We calculate velocities purely so the player objects don't crash when initializing.
    tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True)
    tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)
    
    home_gk, away_gk, home_players, away_players = find_goalkeepers(tracking_home, tracking_away)
    app = PitchControlFormationGUI(events, tracking_home, tracking_away, mpc.default_model_params(), (home_gk, away_gk), home_players, away_players, (106.0, 68.0))
    app.run()

if __name__ == "__main__":
    main()