#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pitch Control Comparison GUI
Pipeline:
1. Takes an Event ID.
2. Computes the "Normal" (straight-line ball travel) PPCF surface using
   generate_pitch_control_for_event().
3. Computes the "Curved" (Bezier trajectory-weighted) PPCF surface using
   generate_pitch_control_for_event_curve_based().
4. Displays both heatmaps side by side, plus a heatmap of the squared error
   between them (and the resulting Mean Squared Error).
"""
import sys
import os
import threading
import time
import traceback
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# --- PATH SETUP ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
LAURIE_PATH = os.path.join(BASE_DIR, "LaurieOnTracking")
MATCH_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "pff_match_data")

sys.path.insert(0, LAURIE_PATH)

import LaurieOnTracking.Metrica_IO as mio
import LaurieOnTracking.Metrica_Velocities as mvel
import LaurieOnTracking.Metrica_PitchControl as mpc


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


class PitchControlComparisonGUI:
    def __init__(self, events, tracking_home, tracking_away, params, GK_numbers, field_dimen, n_grid_cells_x=50):
        self.events = events
        self.tracking_home = tracking_home
        self.tracking_away = tracking_away
        self.params = params
        self.GK_numbers = GK_numbers
        self.field_dimen = field_dimen
        self.n_grid_cells_x = n_grid_cells_x

        self.root = tk.Tk()
        self.root.title("Pitch Control: Normal vs. Curved Trajectory Comparison")
        self.root.state("zoomed")

        # --- CONTROLS ---
        self.control_frame = tk.Frame(self.root, bg="#e1e1e1", height=50)
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Label(self.control_frame, text="Event ID:", bg="#e1e1e1", font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=(10, 2))
        self.entry_event_var = tk.StringVar(value="2")
        self.entry_event = tk.Entry(self.control_frame, textvariable=self.entry_event_var, font=("Arial", 11), width=6)
        self.entry_event.pack(side=tk.LEFT, padx=(0, 15))
        self.entry_event.bind("<Return>", lambda e: self.load_event())

        tk.Label(self.control_frame, text="Offsides:", bg="#e1e1e1", font=("Arial", 11)).pack(side=tk.LEFT, padx=(5, 2))
        self.offsides_var = tk.BooleanVar(value=True)
        self.offsides_chk = tk.Checkbutton(self.control_frame, variable=self.offsides_var, bg="#e1e1e1")
        self.offsides_chk.pack(side=tk.LEFT, padx=(0, 15))

        self.btn_load = tk.Button(
            self.control_frame, text="Run/Refresh", command=self.load_event,
            bg="#4CAF50", fg="white", font=("Arial", 11, "bold")
        )
        self.btn_load.pack(side=tk.LEFT, padx=10)

        self.status_label = tk.Label(self.control_frame, text="Ready", bg="#e1e1e1", fg="blue", font=("Arial", 10, "italic"))
        self.status_label.pack(side=tk.LEFT, padx=20)

        # --- SCROLLABLE PLOT AREA ---
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

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        self.root.destroy()

    def update_status(self, text):
        self.status_label.config(text=text)
        self.root.update()

    # --- COMPUTATION ---
    def _compute_comparison(self, event_id, offsides):
        start_time = time.time()

        PPCF_normal, xgrid, ygrid = mpc.generate_pitch_control_for_event(
            event_id, self.events, self.tracking_home, self.tracking_away, self.params, self.GK_numbers,
            field_dimen=self.field_dimen, n_grid_cells_x=self.n_grid_cells_x, offsides=offsides,
        )

        PPCF_curved, _, _ = mpc.generate_pitch_control_for_event_curve_based(
            event_id, self.events, self.tracking_home, self.tracking_away, self.params, self.GK_numbers,
            field_dimen=self.field_dimen, n_grid_cells_x=self.n_grid_cells_x, offsides=offsides,
        )

        sq_err_map = (PPCF_normal - PPCF_curved) ** 2
        mse = float(np.mean(sq_err_map))
        mae = float(np.mean(np.abs(PPCF_normal - PPCF_curved)))
        max_abs_err = float(np.max(np.abs(PPCF_normal - PPCF_curved)))

        return {
            "event_id": event_id,
            "PPCF_normal": PPCF_normal,
            "PPCF_curved": PPCF_curved,
            "sq_err_map": sq_err_map,
            "mse": mse,
            "mae": mae,
            "max_abs_err": max_abs_err,
            "xgrid": xgrid,
            "ygrid": ygrid,
            "elapsed": time.time() - start_time,
        }

    def _on_analysis_done(self, result):
        self.analysis_running = False
        if "error" in result:
            self.update_status(f"Error: {result['error']}")
            print(f"\nCRITICAL PIPELINE ERROR:\n{result['traceback']}")
            return
        self.render_plots(result)
        self.update_status(
            f"Event {result['event_id']} | MSE={result['mse']:.5f} | MAE={result['mae']:.5f} | "
            f"computed in {result['elapsed']:.2f}s"
        )

    def _analyze_event_worker(self, event_id, offsides):
        try:
            res = self._compute_comparison(event_id, offsides)
            self.root.after(0, lambda: self._on_analysis_done(res))
        except Exception as exc:
            err_trace = traceback.format_exc()
            self.root.after(0, lambda: self._on_analysis_done({"error": str(exc), "traceback": err_trace}))

    def load_event(self):
        if self.analysis_running:
            return
        try:
            event_id = int(self.entry_event_var.get())
            if event_id not in self.events.index:
                self.update_status("Error: Event ID not found!")
                return
        except ValueError:
            self.update_status("Error: Please enter a valid event ID")
            return

        offsides = self.offsides_var.get()
        self.analysis_running = True
        self.update_status(f"Computing PPCF surfaces for event {event_id}...")
        threading.Thread(target=self._analyze_event_worker, args=(event_id, offsides), daemon=True).start()

    # --- RENDERING ---
    def _draw_pitch(self, ax, x_min, x_max, y_min, y_max):
        ax.set_facecolor("#3d7d21")
        ax.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], color="white", linewidth=2)
        ax.plot([0, 0], [y_min, y_max], color="white", linewidth=2)
        ax.add_patch(plt.Circle((0, 0), 9.15, fill=False, color="white", linewidth=2))
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.axis("off")

    def render_plots(self, res):
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        for fig in self.current_figures:
            plt.close(fig)
        self.current_figures.clear()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), dpi=96)
        fig.patch.set_facecolor("#2d5016")
        self.current_figures.append(fig)

        x_min, x_max = -self.field_dimen[0] / 2, self.field_dimen[0] / 2
        y_min, y_max = -self.field_dimen[1] / 2, self.field_dimen[1] / 2

        plots = [
            (res["PPCF_normal"], f"Normal PPCF (straight-line)\nEvent {res['event_id']}", "RdBu_r", 0.0, 1.0),
            (res["PPCF_curved"], f"Curved PPCF (Bezier-weighted)\nEvent {res['event_id']}", "RdBu_r", 0.0, 1.0),
            (res["sq_err_map"], f"Squared Error Heatmap\nMSE = {res['mse']:.5f}", "hot", 0.0, max(res["sq_err_map"].max(), 1e-6)),
        ]

        for ax, (data, title, cmap, vmin, vmax) in zip(axes, plots):
            self._draw_pitch(ax, x_min, x_max, y_min, y_max)
            im = ax.imshow(
                np.flipud(data), extent=[x_min, x_max, y_min, y_max], cmap=cmap,
                alpha=0.8, interpolation="bicubic", aspect="auto", vmin=vmin, vmax=vmax
            )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title, color="white", fontsize=11, fontweight="bold")

        fig.suptitle(
            f"MAE = {res['mae']:.5f}   |   Max Abs Error = {res['max_abs_err']:.5f}",
            color="white", fontsize=11
        )

        plt.tight_layout(pad=1.4, rect=[0, 0, 1, 0.94])
        canvas = FigureCanvasTkAgg(fig, master=self.content_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def run(self):
        self.root.mainloop()


def main():
    print("Launching Pitch Control Comparison GUI...")
    tracking_home, tracking_away, events = mio.load_new_data(MATCH_DATA_PATH, 3812)
    tracking_home = to_metric_coordinates(tracking_home)
    tracking_away = to_metric_coordinates(tracking_away)
    tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True)
    tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)
    home_gk, away_gk, home_players, away_players = find_goalkeepers(tracking_home, tracking_away)
    app = PitchControlComparisonGUI(
        events, tracking_home, tracking_away, mpc.default_model_params(),
        (home_gk, away_gk), (106.0, 68.0), n_grid_cells_x=50
    )
    app.run()


if __name__ == "__main__":
    main()