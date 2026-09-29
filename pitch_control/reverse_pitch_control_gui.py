import os
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from LaurieOnTracking import Metrica_IO as mio
from LaurieOnTracking import Metrica_Velocities as mvel
from LaurieOnTracking import Metrica_PitchControl as mpc
from reverse_pitch_control_solver import (
    ReversePitchControlSolver,
    assignment_distances,
    find_goalkeepers,
    to_metric_coordinates,
)


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MATCH_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "pff_match_data")


class ReversePitchControlGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Reverse Matrix Pitch Control - Event Analyzer")
        self.root.geometry("1400x900")

        self.game_id_var = tk.StringVar(value="3812")
        self.event_id_var = tk.StringVar()
        self.sigma_var = tk.StringVar(value="38")
        self.n_grid_var = tk.StringVar(value="50")
        self.num_candidates_var = tk.StringVar(value="200")
        self.seed_var = tk.StringVar(value="7")

        self.field_dimen = (106.0, 68.0)
        self.tracking_home = None
        self.tracking_away = None
        self.events = None
        self.params = None
        self.GK_numbers = None

        self.current_canvas = None
        self.current_fig = None

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        controls = ttk.LabelFrame(main, text="Controls", padding=8)
        controls.pack(fill=tk.X)

        ttk.Label(controls, text="Game ID:").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.game_id_var, width=10).grid(row=0, column=1, padx=4, sticky="w")
        ttk.Button(controls, text="Load Match", command=self._load_match).grid(row=0, column=2, padx=6, sticky="w")

        ttk.Label(controls, text="Event ID:").grid(row=0, column=3, sticky="w")
        self.event_combo = ttk.Combobox(controls, textvariable=self.event_id_var, width=16, state="readonly")
        self.event_combo.grid(row=0, column=4, padx=4, sticky="w")

        ttk.Label(controls, text="Sigma:").grid(row=0, column=5, sticky="w")
        ttk.Entry(controls, textvariable=self.sigma_var, width=8).grid(row=0, column=6, padx=4, sticky="w")

        ttk.Label(controls, text="Grid X cells:").grid(row=0, column=7, sticky="w")
        ttk.Entry(controls, textvariable=self.n_grid_var, width=8).grid(row=0, column=8, padx=4, sticky="w")

        ttk.Label(controls, text="Reverse candidates:").grid(row=0, column=9, sticky="w")
        ttk.Entry(controls, textvariable=self.num_candidates_var, width=8).grid(row=0, column=10, padx=4, sticky="w")

        ttk.Label(controls, text="Seed:").grid(row=0, column=11, sticky="w")
        ttk.Entry(controls, textvariable=self.seed_var, width=8).grid(row=0, column=12, padx=4, sticky="w")

        ttk.Button(controls, text="Analyze Event", command=self._analyze_event).grid(row=0, column=13, padx=8, sticky="w")

        self.status_var = tk.StringVar(value="Load a match to begin.")
        ttk.Label(controls, textvariable=self.status_var, foreground="blue").grid(
            row=1, column=0, columnspan=14, sticky="w", pady=(6, 0)
        )

        self.output = tk.Text(main, height=10, wrap="word")
        self.output.pack(fill=tk.X, pady=8)

        self.plot_frame = ttk.Frame(main)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

    def _update_status(self, text):
        self.status_var.set(text)
        self.root.update_idletasks()

    def _load_match(self):
        try:
            game_id = int(self.game_id_var.get())
            self._update_status("Loading tracking/event data...")
            tracking_home, tracking_away, events = mio.load_new_data(MATCH_DATA_PATH, game_id)

            self._update_status("Converting coordinates and velocities...")
            tracking_home = to_metric_coordinates(tracking_home, self.field_dimen)
            tracking_away = to_metric_coordinates(tracking_away, self.field_dimen)
            events["Start X"] = (events["Start X"] - 0.5) * self.field_dimen[0]
            events["Start Y"] = (events["Start Y"] - 0.5) * self.field_dimen[1]

            tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True)
            tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)

            home_gk, away_gk = find_goalkeepers(tracking_home, tracking_away)
            self.GK_numbers = (home_gk, away_gk)
            self.params = mpc.default_model_params()

            self.tracking_home = tracking_home
            self.tracking_away = tracking_away
            self.events = events

            event_ids = [str(eid) for eid in events.index.tolist()]
            self.event_combo["values"] = event_ids
            if event_ids:
                self.event_id_var.set(event_ids[0])

            self._update_status(f"Loaded match {game_id}. Events available: {len(event_ids)}")
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            self._update_status("Failed to load match.")

    def _get_event_players(self, event_id):
        pass_frame = self.events.loc[event_id]["Start Frame"]
        pass_team = self.events.loc[event_id].Team
        ball_start_pos = np.array([self.events.loc[event_id]["Start X"], self.events.loc[event_id]["Start Y"]])

        if pass_frame in self.tracking_home.index:
            h_row = self.tracking_home.loc[pass_frame]
            a_row = self.tracking_away.loc[pass_frame]
        else:
            h_row = self.tracking_home.iloc[(self.tracking_home.index - pass_frame).abs().argmin()]
            a_row = self.tracking_away.iloc[(self.tracking_away.index - pass_frame).abs().argmin()]

        if pass_team == "Home":
            attacking_players = mpc.initialise_players(h_row, "Home", self.params, self.GK_numbers[0])
            defending_players = mpc.initialise_players(a_row, "Away", self.params, self.GK_numbers[1])
        elif pass_team == "Away":
            defending_players = mpc.initialise_players(h_row, "Home", self.params, self.GK_numbers[0])
            attacking_players = mpc.initialise_players(a_row, "Away", self.params, self.GK_numbers[1])
        else:
            raise ValueError("Team in possession must be Home or Away")

        attacking_players = mpc.check_offsides(attacking_players, defending_players, ball_start_pos, self.GK_numbers)

        att_pos = np.array([[p.position[0], p.position[1]] for p in attacking_players], dtype=float)
        def_pos = np.array([[p.position[0], p.position[1]] for p in defending_players], dtype=float)

        return pass_team, att_pos, def_pos

    def _build_grid(self, n_grid_cells_x):
        n_grid_cells_y = int(n_grid_cells_x * self.field_dimen[1] / self.field_dimen[0])
        dx = self.field_dimen[0] / n_grid_cells_x
        dy = self.field_dimen[1] / n_grid_cells_y
        xgrid = np.arange(n_grid_cells_x) * dx - self.field_dimen[0] / 2.0 + dx / 2.0
        ygrid = np.arange(n_grid_cells_y) * dy - self.field_dimen[1] / 2.0 + dy / 2.0
        xx, yy = np.meshgrid(xgrid, ygrid)
        grid_coords = np.column_stack([xx.ravel(), yy.ravel()])
        return xgrid, ygrid, grid_coords

    def _render_plots(
        self,
        event_id,
        pass_team,
        ppcf_orig,
        ppcf_recon,
        ppcf_err_pct,
    ):
        if self.current_canvas is not None:
            self.current_canvas.get_tk_widget().destroy()
        if self.current_fig is not None:
            plt.close(self.current_fig)

        fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=100)
        self.current_fig = fig

        x_min, x_max = -self.field_dimen[0] / 2.0, self.field_dimen[0] / 2.0
        y_min, y_max = -self.field_dimen[1] / 2.0, self.field_dimen[1] / 2.0

        def draw_pitch(ax):
            ax.set_facecolor("#2f5f26")
            ax.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], color="white", lw=1.5)
            ax.plot([0, 0], [y_min, y_max], color="white", lw=1.2)
            ax.add_patch(plt.Circle((0, 0), 9.15, fill=False, color="white", lw=1.2))
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.axis("off")

        ppcf_orig_disp = np.clip(ppcf_orig, 0.0, 1.0)
        ppcf_recon_disp = np.clip(ppcf_recon, 0.0, 1.0)
        err_cap = np.percentile(ppcf_err_pct, 99)
        if not np.isfinite(err_cap) or err_cap <= 0:
            err_cap = 1.0
        ppcf_err_disp = np.clip(ppcf_err_pct, 0, err_cap)

        heat_cfg = [
            (ppcf_orig_disp, "Original Matrix PPCF", "RdBu_r", (0, 1)),
            (ppcf_recon_disp, "Reconstructed Matrix PPCF", "RdBu_r", (0, 1)),
            (ppcf_err_disp, "PPCF Error %", "hot", (0, err_cap)),
        ]

        for ax, (data, title, cmap, limits) in zip(axes, heat_cfg):
            draw_pitch(ax)
            vmin, vmax = limits
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = vmin + 1.0
            im = ax.imshow(
                np.flipud(data),
                extent=[x_min, x_max, y_min, y_max],
                cmap=cmap,
                interpolation="gaussian",
                alpha=0.80,
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(title, fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(f"Event {event_id} | Team in possession: {pass_team}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        self.current_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.current_canvas.draw()
        self.current_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _compute_matrix_ppcf_from_positions(self, grid_coords, n_grid_y, n_grid_x, att_pos, def_pos):
        fgg = mpc.FastGaussianGrid(sigma=38.0)
        att_inf = fgg.generate_heatmap(grid_coords, att_pos, sharpness=40).reshape(n_grid_y, n_grid_x)
        def_inf = fgg.generate_heatmap(grid_coords, def_pos, sharpness=40).reshape(n_grid_y, n_grid_x)
        total = att_inf + def_inf
        return np.where(total > 1e-12, att_inf / total, 0.5)

    def _analyze_event(self):
        try:
            if self.events is None:
                raise ValueError("Load match data first.")

            event_id = int(self.event_id_var.get())
            n_grid = int(self.n_grid_var.get())
            num_candidates = int(self.num_candidates_var.get())
            seed = int(self.seed_var.get())

            pass_team, att_pos, def_pos = self._get_event_players(event_id)
            if len(att_pos) == 0 or len(def_pos) == 0:
                raise ValueError("Event has empty attacker or defender set after filtering.")

            self._update_status("Building original matrix PPCF...")
            _, _, grid_coords = self._build_grid(n_grid)
            n_grid_y = int(n_grid * self.field_dimen[1] / self.field_dimen[0])

            # Exact original matrix-based pitch-control output (attacking control probability).
            ppcf_orig, _, _ = mpc.generate_pitch_control_for_event_linearized_matrixform(
                event_id,
                self.events,
                self.tracking_home,
                self.tracking_away,
                self.params,
                self.GK_numbers,
                field_dimen=self.field_dimen,
                n_grid_cells_x=n_grid,
                offsides=True,
            )

            # Influence maps are used as reverse-problem inputs.
            fgg = mpc.FastGaussianGrid(sigma=38.0)
            att_heatmap = fgg.generate_heatmap(grid_coords, att_pos, sharpness=40).reshape(n_grid_y, n_grid)
            def_heatmap = fgg.generate_heatmap(grid_coords, def_pos, sharpness=40).reshape(n_grid_y, n_grid)

            self._update_status("Solving reverse problem for attack and defense...")
            solver = ReversePitchControlSolver(field_dimen=self.field_dimen, sigma=38.0)

            att_result = solver.generate_infinite_player_sets(
                att_heatmap,
                n_players=len(att_pos),
                num_candidates=num_candidates,
                top_k=1,
                seed=seed,
            )
            def_result = solver.generate_infinite_player_sets(
                def_heatmap,
                n_players=len(def_pos),
                num_candidates=num_candidates,
                top_k=1,
                seed=seed + 999,
            )

            att_recon_pos = att_result["top_solutions"][0].players_xy
            def_recon_pos = def_result["top_solutions"][0].players_xy

            att_dists = assignment_distances(att_pos, att_recon_pos)
            def_dists = assignment_distances(def_pos, def_recon_pos)

            ppcf_recon = self._compute_matrix_ppcf_from_positions(
                grid_coords, n_grid_y, n_grid, att_recon_pos, def_recon_pos
            )
            ppcf_err_pct = (np.abs(ppcf_orig - ppcf_recon) / (np.abs(ppcf_orig) + 1e-6)) * 100.0

            self.output.delete("1.0", tk.END)
            self.output.insert(tk.END, f"Event {event_id} | possession={pass_team}\n\n")
            self.output.insert(
                tk.END,
                f"Attack reverse position error (m): mean={np.mean(att_dists):.3f}, std={np.std(att_dists):.3f}\n",
            )
            self.output.insert(
                tk.END,
                f"Defense reverse position error (m): mean={np.mean(def_dists):.3f}, std={np.std(def_dists):.3f}\n\n",
            )
            self.output.insert(
                tk.END,
                f"PPCF error % (original vs reconstructed): mean={np.mean(ppcf_err_pct):.2f}, std={np.std(ppcf_err_pct):.2f}\n",
            )
            self.output.insert(tk.END, "\n")
            self.output.insert(tk.END, "Top reconstructed attacker positions:\n")
            self.output.insert(tk.END, f"{att_recon_pos.tolist()}\n\n")
            self.output.insert(tk.END, "Top reconstructed defender positions:\n")
            self.output.insert(tk.END, f"{def_recon_pos.tolist()}\n")

            self._render_plots(
                event_id,
                pass_team,
                ppcf_orig,
                ppcf_recon,
                ppcf_err_pct,
            )

            self._update_status("Analysis complete.")

        except Exception as exc:
            messagebox.showerror("Analyze error", str(exc))
            self._update_status("Analysis failed.")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    ReversePitchControlGUI().run()
