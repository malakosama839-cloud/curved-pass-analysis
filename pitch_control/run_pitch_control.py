"""
Pitch Control Analysis Script - Interactive GUI Version
Fixed: proper scrollbars (horizontal + vertical), figure sized to fit screen
"""
import sys
import os
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd

# --- PATH SETUP ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
LAURIE_PATH = os.path.join(BASE_DIR, 'LaurieOnTracking')
MATCH_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "pff_match_data")

sys.path.insert(0, LAURIE_PATH)

import Metrica_IO as mio
import Metrica_Velocities as mvel
import Metrica_PitchControl as mpc

# --- HELPER FUNCTIONS ---

def to_metric_coordinates(data, field_dimen=(106., 68.)):
    x_columns = [c for c in data.columns if c[-2:] == '_x' and c[:4] in ['Home', 'Away']]
    y_columns = [c for c in data.columns if c[-2:] == '_y' and c[:4] in ['Home', 'Away']]
    data[x_columns] = (data[x_columns] - 0.5) * field_dimen[0]
    data[y_columns] = (data[y_columns] - 0.5) * field_dimen[1]
    return data

def find_goalkeepers(tracking_home, tracking_away):
    home_players = sorted(set([c.split('_')[1] for c in tracking_home.columns
                               if c.startswith('Home_') and c.endswith('_x')]))
    away_players = sorted(set([c.split('_')[1] for c in tracking_away.columns
                               if c.startswith('Away_') and c.endswith('_x')]))
    home_avg_x = {p: tracking_home[f'Home_{p}_x'].mean() for p in home_players}
    away_avg_x = {p: tracking_away[f'Away_{p}_x'].mean() for p in away_players}
    home_gk = min(home_avg_x, key=home_avg_x.get)
    away_gk = max(away_avg_x, key=away_avg_x.get)
    return home_gk, away_gk, home_players, away_players

# --- GUI CLASS ---

class PitchControlGUI:
    def __init__(self, events, tracking_home, tracking_away, params, GK_numbers, home_players, away_players, field_dimen):
        self.events = events
        self.tracking_home = tracking_home
        self.tracking_away = tracking_away
        self.params = params
        self.GK_numbers = GK_numbers
        self.home_players = home_players
        self.away_players = away_players
        self.field_dimen = field_dimen

        # Initialize Window - maximized
        self.root = tk.Tk()
        self.root.title("Metrica Pitch Control Analyzer")
        self.root.state('zoomed')  # Start maximized on Windows

        # --- CONTROL FRAME (Top Bar) ---
        self.control_frame = tk.Frame(self.root, bg="#e1e1e1", height=50)
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Label(self.control_frame, text="Enter Event ID:", bg="#e1e1e1", font=("Arial", 12)).pack(side=tk.LEFT, padx=10)

        self.entry_var = tk.StringVar(value="1")
        self.entry = tk.Entry(self.control_frame, textvariable=self.entry_var, font=("Arial", 12), width=10)
        self.entry.pack(side=tk.LEFT, padx=10)

        self.btn_load = tk.Button(self.control_frame, text="Analyze Event", command=self.load_event,
                                  bg="#4CAF50", fg="white", font=("Arial", 11, "bold"))
        self.btn_load.pack(side=tk.LEFT, padx=10)

        self.status_label = tk.Label(self.control_frame, text="Ready", bg="#e1e1e1", fg="blue", font=("Arial", 10, "italic"))
        self.status_label.pack(side=tk.LEFT, padx=20)

        # --- SCROLLABLE CONTENT AREA (both directions) ---
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=1)

        # Canvas with both scrollbars
        self.canvas = tk.Canvas(self.main_frame, bg="#1a1a1a")

        self.v_scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)

        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        self.canvas.configure(
            yscrollcommand=self.v_scrollbar.set,
            xscrollcommand=self.h_scrollbar.set
        )

        self.content_frame = tk.Frame(self.canvas, bg="#1a1a1a")
        self.canvas_window = self.canvas.create_window((0, 0), window=self.content_frame, anchor="nw")

        self.content_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Mouse wheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self.current_figures = []

    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        # Make content frame at least as wide as canvas
        self.canvas.itemconfig(self.canvas_window, width=max(event.width, self.content_frame.winfo_reqwidth()))

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def update_status(self, text):
        self.status_label.config(text=text)
        self.root.update()

    def load_event(self):
        try:
            event_id = int(self.entry_var.get())
            if event_id not in self.events.index:
                self.update_status(f"Error: Event {event_id} not found!")
                return
        except ValueError:
            self.update_status("Error: Please enter a numeric Event ID")
            return

        # Clear previous plots
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        for fig in self.current_figures:
            plt.close(fig)
        self.current_figures = []

        # Perform Calculations
        self.update_status(f"Calculating Baseline Model for Event {event_id}...")
        PPCFa, xgrid, ygrid = mpc.generate_pitch_control_for_event(
            event_id, self.events, self.tracking_home, self.tracking_away,
            self.params, self.GK_numbers, field_dimen=self.field_dimen,
            n_grid_cells_x=50, offsides=False
        )

        self.update_status("Calculating Matrix Distance Based Probability Model...")
        PPCF_linearized_matrixform, _, _ = mpc.generate_pitch_control_for_event_linearized_matrixform(
            event_id, self.events, self.tracking_home, self.tracking_away,
            self.params, self.GK_numbers, field_dimen=self.field_dimen,
            n_grid_cells_x=50, offsides=False
        )

        self.update_status("Calculating Gaussian Distance Based Probability Model...")
        PPCF_gaussian_matrixform, _, _ = mpc.generate_gaussian_pitch_control_for_event(
            event_id, self.events, self.tracking_home, self.tracking_away,
            self.params, self.GK_numbers, field_dimen=self.field_dimen,
            n_grid_cells_x=50, offsides=False
        )

        self.update_status("Calculating Distance Based Probability Model with exponential...")
        PPCF_with_exponential_matrixform, _, _ = mpc.generate_pitch_control_for_event_with_exponential(
            event_id, self.events, self.tracking_home, self.tracking_away,
            self.params, self.GK_numbers, field_dimen=self.field_dimen,
            n_grid_cells_x=50, offsides=False, sigma=7
        )


        modified_models = [
            (PPCF_linearized_matrixform, "Matrix Distance Based Probability"),
            (PPCF_gaussian_matrixform, "Gaussian Distance Based Probability"),
            (PPCF_with_exponential_matrixform, "Distance Based Probability with exponential")
        ]

        self.update_status("Rendering Visualizations...")
        self.render_plots(event_id, PPCFa, modified_models)
        self.update_status(f"Analysis complete for Event {event_id}")

    def render_plots(self, event_id, PPCFa, modified_models):
        event_info = self.events.loc[event_id]
        pass_frame = event_info['Start Frame']

        if pass_frame in self.tracking_home.index:
            h_frame = self.tracking_home.loc[pass_frame]
            a_frame = self.tracking_away.loc[pass_frame]
        else:
            h_frame = self.tracking_home.iloc[(self.tracking_home.index - pass_frame).abs().argmin()]
            a_frame = self.tracking_away.iloc[(self.tracking_away.index - pass_frame).abs().argmin()]

        significant_mask = PPCFa > 0.01
        num_models = len(modified_models)

        # Get screen width in inches for figure sizing
        screen_width_px = self.root.winfo_screenwidth()
        screen_height_px = self.root.winfo_screenheight()
        dpi = 96
        # Use 95% of screen width, leave room for scrollbar
        fig_width = (screen_width_px * 0.95) / dpi
        row_height = (screen_height_px * 0.42) / dpi
        fig_height = row_height * num_models

        fig, axes = plt.subplots(num_models, 3, figsize=(fig_width, fig_height), dpi=dpi)
        fig.patch.set_facecolor('#2d5016')
        self.current_figures.append(fig)

        if num_models == 1:
            axes = np.expand_dims(axes, axis=0)

        for row_idx, (PPCFb, model_name) in enumerate(modified_models):
            diff_map = PPCFa - PPCFb
            abs_diff = np.abs(diff_map)

            mean_rel = 0.0
            if np.any(significant_mask):
                rel_err_map = (abs_diff / (PPCFa + 1e-6)) * 100
                mean_rel = np.mean(rel_err_map[significant_mask])

            max_abs = np.max(abs_diff)

            configs = [
                {'data': PPCFa,    'cmap': 'RdBu',    'title': f"Baseline (Original)\nMean: {PPCFa.mean():.1%}"},
                {'data': PPCFb,    'cmap': 'RdBu',    'title': f"Model: {model_name}\nMean: {PPCFb.mean():.1%}"},
                {'data': diff_map, 'cmap': 'seismic', 'title': f"Diff (Original - {model_name})", 'vmin': -0.5, 'vmax': 0.5}
            ]

            for col_idx in range(3):
                ax = axes[row_idx, col_idx]
                cfg = configs[col_idx]
                ax.set_facecolor('#3d7d21')

                # Pitch lines
                ax.plot([-53, 53, 53, -53, -53], [-34, -34, 34, 34, -34], color='white', linewidth=2)
                ax.plot([0, 0], [-34, 34], color='white', linewidth=2)
                ax.add_patch(plt.Circle((0, 0), 9.15, fill=False, color='white', linewidth=2))
                ax.plot([-53, -36.5, -36.5, -53], [-20.16, -20.16, 20.16, 20.16], color='white', linewidth=1.5, alpha=0.8)
                ax.plot([53, 36.5, 36.5, 53], [-20.16, -20.16, 20.16, 20.16], color='white', linewidth=1.5, alpha=0.8)

                # Heatmap
                ax.imshow(np.flipud(cfg['data']), extent=[-53, 53, -34, 34], cmap=cfg['cmap'],
                          alpha=0.7, interpolation='gaussian', aspect='auto',
                          vmin=cfg.get('vmin', 0), vmax=cfg.get('vmax', 1))

                # Players (first 2 cols only)
                if col_idx < 2:
                    for p in self.home_players:
                        x, y = h_frame[f'Home_{p}_x'], h_frame[f'Home_{p}_y']
                        if not np.isnan(x):
                            c = 'darkred' if p == self.GK_numbers[0] else '#FF4444'
                            ax.plot(x, y, 'o', color=c, ms=8, mec='white')
                            ax.text(x, y, p, color='white', fontsize=6, ha='center', va='center', fontweight='bold')
                    for p in self.away_players:
                        x, y = a_frame[f'Away_{p}_x'], a_frame[f'Away_{p}_y']
                        if not np.isnan(x):
                            c = 'darkblue' if p == self.GK_numbers[1] else '#4444FF'
                            ax.plot(x, y, 'o', color=c, ms=8, mec='white')
                            ax.text(x, y, p, color='white', fontsize=6, ha='center', va='center', fontweight='bold')
                    ax.plot(event_info['Start X'], event_info['Start Y'], 'o', color='yellow', ms=7, mec='black')

                # Stats box (3rd col only)
                if col_idx == 2:
                    txt = f"Max Abs Diff: {max_abs:.3f}\nMean Rel Err: {mean_rel:.1f}%"
                    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=9, fontfamily='monospace', va='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                ax.set_title(cfg['title'], color='white', fontsize=10, fontweight='bold')
                ax.set_xlim([-53, 53])
                ax.set_ylim([-34, 34])
                ax.axis('off')

        plt.tight_layout(pad=1.5)

        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.content_frame)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def run(self):
        self.root.mainloop()

# --- MAIN EXECUTION ---

def main():
    print("=" * 60)
    print("PITCH CONTROL ANALYSIS - INTERACTIVE GUI")
    print("=" * 60)

    game_id = 3812
    data_path = MATCH_DATA_PATH
    field_dimen = (106., 68.)

    print("\nLoading tracking and event data...")
    tracking_home, tracking_away, events = mio.load_new_data(data_path, game_id)
    print(events)

    print("Converting coordinates and calculating velocities...")
    tracking_home = to_metric_coordinates(tracking_home, field_dimen)
    tracking_away = to_metric_coordinates(tracking_away, field_dimen)

    events['Start X'] = (events['Start X'] - 0.5) * field_dimen[0]
    events['Start Y'] = (events['Start Y'] - 0.5) * field_dimen[1]

    tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True)
    tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)

    home_gk, away_gk, home_players, away_players = find_goalkeepers(tracking_home, tracking_away)
    GK_numbers = (home_gk, away_gk)
    params = mpc.default_model_params()

    print("✓ Data Ready. Launching GUI...")

    app = PitchControlGUI(
        events,
        tracking_home,
        tracking_away,
        params,
        GK_numbers,
        home_players,
        away_players,
        field_dimen
    )

    pass_events = events[events['Type'] == 'PASS']
    if not pass_events.empty:
        first_event_id = pass_events.index[0]
        app.entry_var.set(str(first_event_id))

    app.run()

if __name__ == "__main__":
    main()