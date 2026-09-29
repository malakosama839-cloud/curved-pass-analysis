#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OBSO + Pitch Control Comparison GUI
Pipeline:
1. Pick an event from a dropdown of all available events.
2. Compute the "Normal" (straight-line ball travel) PPCF surface via
   generate_pitch_control_for_event().
3. Compute the "Curved" (Bezier trajectory-weighted) PPCF surface via
   generate_pitch_control_for_event_curve_based().
4. Feed both PPCF surfaces into calc_obso() (same Transition/Score grids)
   to get the corresponding "Normal" and "Curved" OBSO surfaces.
5. Display a 2x3 grid:
     Row 1 (OBSO): Normal | Curved | Squared error
     Row 2 (PC)  : Normal | Curved | Squared error
   plus MSE/MAE/max-abs-error summaries for both rows.
"""
import sys
import os
import re
import json
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
import LaurieOnTracking.Metrica_EPV as mepv

# Module containing calc_obso() / calc_obso_curve_based() (adjust import to
# match wherever those functions actually live in your project).
import obso_player as obso


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


def discover_available_matches(data_dir=MATCH_DATA_PATH):
    """Match ids in data_dir that have players CSV + ball CSV + an events JSON
    (the three files mio.load_new_data requires)."""
    matches = []
    for fname in os.listdir(data_dir):
        m = re.match(r"^match_(\d+)_players\.csv$", fname)
        if not m:
            continue
        match_id = m.group(1)
        ball_csv = os.path.join(data_dir, f"match_{match_id}_ball.csv")
        event_json = os.path.join(data_dir, f"{match_id}.json")
        if os.path.exists(ball_csv) and os.path.exists(event_json):
            matches.append(match_id)
    return sorted(matches, key=int)


def normalize_event_teams(events, match_id):
    """The event converter leaves real team names (e.g. 'Argentina') in the Team
    column for most matches; the pitch control code expects 'Home'/'Away'
    consistent with the tracking split, so map them via the match meta file."""
    team_values = set(events["Team"].dropna().unique())
    if team_values <= {"Home", "Away"}:
        return events
    home_name = away_name = None
    meta_path = os.path.join(MATCH_DATA_PATH, f"{match_id}_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            entry = meta[0] if isinstance(meta, list) and meta else meta
            home_name = (entry.get("homeTeam") or {}).get("name")
            away_name = (entry.get("awayTeam") or {}).get("name")
        except Exception as exc:
            print(f"  WARNING: could not read {match_id}_meta.json ({exc})")
    if home_name is None or away_name is None:
        ordered = sorted(team_values)
        home_name, away_name = ordered[0], ordered[-1]
        print(f"  WARNING: no team mapping for match {match_id}; assuming '{home_name}'=Home, '{away_name}'=Away")
    print(f"  Mapping event teams for match {match_id}: '{home_name}' -> Home, '{away_name}' -> Away")
    events = events.copy()
    events["Team"] = events["Team"].replace({home_name: "Home", away_name: "Away"})
    return events


_GK_CACHE = {}


def find_goalkeepers_from_json(match_id):
    """Goalkeeper jersey numbers for (Home, Away) taken from the positionGroupType
    field of the match's events JSON. The avg-position heuristic is unreliable for
    matches with substitutions and half-time end swaps, so the annotated data wins."""
    if match_id in _GK_CACHE:
        return _GK_CACHE[match_id]
    result = (None, None)

    def gk_jersey(players):
        jerseys = [p.get("jerseyNum") for p in (players or [])
                   if p.get("positionGroupType") == "GK" and p.get("jerseyNum") is not None]
        if not jerseys:
            return None
        counts = {}
        for j in jerseys:
            counts[j] = counts.get(j, 0) + 1
        return str(max(counts, key=counts.get))

    try:
        with open(os.path.join(MATCH_DATA_PATH, f"{match_id}.json"), encoding="utf-8") as f:
            data = json.load(f)
        for frame in (data if isinstance(data, list) else [data]):
            home_gk = gk_jersey(frame.get("homePlayers"))
            away_gk = gk_jersey(frame.get("awayPlayers"))
            if home_gk and away_gk:
                result = (home_gk, away_gk)
                break
    except Exception as exc:
        print(f"  WARNING: could not read goalkeepers from {match_id}.json ({exc})")
    _GK_CACHE[match_id] = result
    return result


def load_match_data(match_id):
    """Load tracking + events for one match and prepare them for the GUI."""
    tracking_home, tracking_away, events = mio.load_new_data(MATCH_DATA_PATH, match_id)
    events = normalize_event_teams(events, match_id)
    tracking_home = to_metric_coordinates(tracking_home)
    tracking_away = to_metric_coordinates(tracking_away)
    tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True)
    tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)
    home_gk, away_gk = find_goalkeepers_from_json(match_id)
    if home_gk is None or away_gk is None:
        home_gk, away_gk, _, _ = find_goalkeepers(tracking_home, tracking_away)
    else:
        print(f"  Goalkeepers from event data: Home_{home_gk}, Away_{away_gk}")
    return {
        "match_id": match_id,
        "tracking_home": tracking_home,
        "tracking_away": tracking_away,
        "events": events,
        "GK_numbers": (home_gk, away_gk),
    }


def load_transition_grid(fname="Transition_gauss.csv"):
    """
    Load a pregenerated Transition surface from file (comma-delimited grid,
    same format as Metrica_EPV.load_EPV_grid but for the transition matrix
    used by calc_obso). Metrica_EPV does not ship a loader for this file,
    so it's loaded directly here -- adjust fname/path to wherever your
    Transition grid actually lives.
    """
    return np.loadtxt(fname, delimiter=",")


def format_event_label(event_id, row):
    team = row.get("Team", "?")
    etype = row.get("Type", "?")
    subtype = row.get("Subtype", "")
    subtype_str = f" ({subtype})" if isinstance(subtype, str) and subtype and subtype.lower() != "nan" else ""
    return f"{event_id}: {team} - {etype}{subtype_str}"


class OBSOPitchControlComparisonGUI:
    def __init__(self, events, tracking_home, tracking_away, params, GK_numbers,
                 field_dimen, Transition, Score, n_grid_cells_x=50,
                 initial_match_id="3812", available_matches=None):
        self.events = events
        self.tracking_home = tracking_home
        self.tracking_away = tracking_away
        self.params = params
        self.GK_numbers = GK_numbers
        self.field_dimen = field_dimen
        self.Transition = Transition
        self.Score = Score
        self.n_grid_cells_x = n_grid_cells_x
        self.current_match_id = str(initial_match_id)
        self.available_matches = [str(m) for m in (available_matches or [initial_match_id])]

        # only events with a valid Team ('Home'/'Away') can have PPCF/OBSO computed
        valid_events = self.events[self.events["Team"].isin(["Home", "Away"])]
        self.event_ids = list(valid_events.index)
        self.event_labels = [format_event_label(eid, valid_events.loc[eid]) for eid in self.event_ids]
        self.label_to_id = dict(zip(self.event_labels, self.event_ids))

        self.root = tk.Tk()
        self.root.title(f"OBSO & Pitch Control: Normal vs. Curved Trajectory Comparison - Match {self.current_match_id}")
        self.root.state("zoomed")

        # --- CONTROLS ---
        self.control_frame = tk.Frame(self.root, bg="#e1e1e1", height=50)
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Label(self.control_frame, text="Match:", bg="#e1e1e1", font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=(10, 2))
        self.match_var = tk.StringVar(value=self.current_match_id)
        self.match_combo = ttk.Combobox(
            self.control_frame, textvariable=self.match_var, values=self.available_matches,
            state="readonly", width=8, font=("Arial", 10)
        )
        self.match_combo.pack(side=tk.LEFT, padx=(0, 15))
        self.match_combo.bind("<<ComboboxSelected>>", lambda e: self.on_match_selected())

        tk.Label(self.control_frame, text="Event:", bg="#e1e1e1", font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=(10, 2))
        self.event_var = tk.StringVar(value=self.event_labels[0] if self.event_labels else "")
        self.event_combo = ttk.Combobox(
            self.control_frame, textvariable=self.event_var, values=self.event_labels,
            state="readonly", width=40, font=("Arial", 10)
        )
        self.event_combo.pack(side=tk.LEFT, padx=(0, 15))
        self.event_combo.bind("<<ComboboxSelected>>", lambda e: self.load_event())

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

    # --- MATCH SWITCHING ---
    def _set_busy(self, busy):
        self.match_combo.config(state="disabled" if busy else "readonly")
        self.event_combo.config(state="disabled" if busy else "readonly")
        self.btn_load.config(state="disabled" if busy else "normal")

    def _rebuild_event_list(self):
        valid_events = self.events[self.events["Team"].isin(["Home", "Away"])]
        self.event_ids = list(valid_events.index)
        self.event_labels = [format_event_label(eid, valid_events.loc[eid]) for eid in self.event_ids]
        self.label_to_id = dict(zip(self.event_labels, self.event_ids))
        self.event_combo["values"] = self.event_labels
        self.event_var.set(self.event_labels[0] if self.event_labels else "")

    def on_match_selected(self):
        if self.analysis_running:
            self.match_var.set(self.current_match_id)
            self.update_status("Busy - wait for the current computation to finish before switching matches")
            return
        match_id = self.match_var.get()
        if match_id == self.current_match_id:
            return
        self.analysis_running = True
        self._set_busy(True)
        self.update_status(f"Loading match {match_id} (tracking data + velocities, may take a minute)...")
        threading.Thread(target=self._match_load_worker, args=(match_id,), daemon=True).start()

    def _match_load_worker(self, match_id):
        try:
            data = load_match_data(match_id)
            self.root.after(0, lambda: self._on_match_loaded(data))
        except Exception as exc:
            err_msg = str(exc)
            err_trace = traceback.format_exc()
            self.root.after(0, lambda msg=err_msg, tb=err_trace: self._on_match_load_error(msg, tb))

    def _on_match_loaded(self, data):
        self.current_match_id = data["match_id"]
        self.tracking_home = data["tracking_home"]
        self.tracking_away = data["tracking_away"]
        self.events = data["events"]
        self.GK_numbers = data["GK_numbers"]
        self._rebuild_event_list()
        self.root.title(f"OBSO & Pitch Control: Normal vs. Curved Trajectory Comparison - Match {self.current_match_id}")
        self.analysis_running = False
        self._set_busy(False)
        self.update_status(f"Match {self.current_match_id} loaded - {len(self.event_ids)} events ready")

    def _on_match_load_error(self, msg, tb):
        self.match_var.set(self.current_match_id)
        self.analysis_running = False
        self._set_busy(False)
        self.update_status(f"Error loading match: {msg}")
        print(f"\nMATCH LOAD ERROR:\n{tb}")

    def _get_attack_direction(self, event_id):
        row = self.events.loc[event_id]
        team = row["Team"]
        period = row["Period"]
        if team == "Home":
            return mio.find_playing_direction(self.tracking_home[self.tracking_home["Period"] == period], "Home")
        else:
            return mio.find_playing_direction(self.tracking_away[self.tracking_away["Period"] == period], "Away")

    # --- COMPUTATION ---
    def _compute_comparison(self, event_id, offsides):
        start_time = time.time()

        pass_frame = self.events.loc[event_id]["Start Frame"]
        pass_team = self.events.loc[event_id]["Team"]
        tracking_row = self.tracking_home.loc[pass_frame] if pass_team == "Home" else self.tracking_away.loc[pass_frame]
        attack_direction = self._get_attack_direction(event_id)

        # --- Pitch control: normal (straight-line) vs curved (Bezier) ---
        PPCF_normal, xgrid, ygrid = mpc.generate_pitch_control_for_event(
            event_id, self.events, self.tracking_home, self.tracking_away, self.params, self.GK_numbers,
            field_dimen=self.field_dimen, n_grid_cells_x=self.n_grid_cells_x, offsides=offsides,
        )

        PPCF_curved, _, _ = mpc.generate_pitch_control_for_event_curve_based(
            event_id, self.events, self.tracking_home, self.tracking_away, self.params, self.GK_numbers,
            field_dimen=self.field_dimen, n_grid_cells_x=self.n_grid_cells_x, offsides=offsides,
        )

        pc_sq_err_map = (PPCF_normal - PPCF_curved) ** 2
        pc_mse = float(np.mean(pc_sq_err_map))
        pc_mae = float(np.mean(np.abs(PPCF_normal - PPCF_curved)))
        pc_max_abs_err = float(np.max(np.abs(PPCF_normal - PPCF_curved)))

        # --- OBSO: same Transition/Score grids, fed by each PPCF surface ---
        OBSO_normal, _ = obso.calc_obso(
            PPCF_normal, self.Transition, self.Score, tracking_row, attack_direction=attack_direction
        )
        OBSO_curved, _ = obso.calc_obso(
            PPCF_curved, self.Transition, self.Score, tracking_row, attack_direction=attack_direction
        )

        obso_sq_err_map = (OBSO_normal - OBSO_curved) ** 2
        obso_mse = float(np.mean(obso_sq_err_map))
        obso_mae = float(np.mean(np.abs(OBSO_normal - OBSO_curved)))
        obso_max_abs_err = float(np.max(np.abs(OBSO_normal - OBSO_curved)))

        return {
            "event_id": event_id,
            "PPCF_normal": PPCF_normal,
            "PPCF_curved": PPCF_curved,
            "pc_sq_err_map": pc_sq_err_map,
            "pc_mse": pc_mse,
            "pc_mae": pc_mae,
            "pc_max_abs_err": pc_max_abs_err,
            "OBSO_normal": OBSO_normal,
            "OBSO_curved": OBSO_curved,
            "obso_sq_err_map": obso_sq_err_map,
            "obso_mse": obso_mse,
            "obso_mae": obso_mae,
            "obso_max_abs_err": obso_max_abs_err,
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
            f"Event {result['event_id']} | "
            f"OBSO MSE={result['obso_mse']:.3e} MAE={result['obso_mae']:.3e} | "
            f"PC MSE={result['pc_mse']:.5f} MAE={result['pc_mae']:.5f} | "
            f"computed in {result['elapsed']:.2f}s"
        )

    def _analyze_event_worker(self, event_id, offsides):
        try:
            res = self._compute_comparison(event_id, offsides)
            self.root.after(0, lambda: self._on_analysis_done(res))
        except Exception as exc:
            err_msg = str(exc)
            err_trace = traceback.format_exc()
            self.root.after(0, lambda msg=err_msg, tb=err_trace: self._on_analysis_done({"error": msg, "traceback": tb}))

    def load_event(self):
        if self.analysis_running:
            return
        label = self.event_var.get()
        if label not in self.label_to_id:
            self.update_status("Error: please select a valid event from the dropdown")
            return
        event_id = self.label_to_id[label]

        offsides = self.offsides_var.get()
        self.analysis_running = True
        self.update_status(f"Computing OBSO / PPCF surfaces for event {event_id}...")
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

        fig, axes = plt.subplots(2, 3, figsize=(18, 12.5), dpi=96)
        fig.patch.set_facecolor("#2d5016")
        self.current_figures.append(fig)

        x_min, x_max = -self.field_dimen[0] / 2, self.field_dimen[0] / 2
        y_min, y_max = -self.field_dimen[1] / 2, self.field_dimen[1] / 2

        obso_vmax = max(res["OBSO_normal"].max(), res["OBSO_curved"].max(), 1e-6)

        row_plots = [
            # Row 1: OBSO
            [
                (res["OBSO_normal"], f"Normal OBSO (straight-line)\nEvent {res['event_id']}", "viridis", 0.0, obso_vmax),
                (res["OBSO_curved"], f"Curved OBSO (Bezier-weighted)\nEvent {res['event_id']}", "viridis", 0.0, obso_vmax),
                (res["obso_sq_err_map"], f"OBSO Squared Error\nMSE = {res['obso_mse']:.3e}", "hot", 0.0, max(res["obso_sq_err_map"].max(), 1e-30)),
            ],
            # Row 2: Pitch control
            [
                (res["PPCF_normal"], f"Normal PPCF (straight-line)\nEvent {res['event_id']}", "RdBu_r", 0.0, 1.0),
                (res["PPCF_curved"], f"Curved PPCF (Bezier-weighted)\nEvent {res['event_id']}", "RdBu_r", 0.0, 1.0),
                (res["pc_sq_err_map"], f"PPCF Squared Error\nMSE = {res['pc_mse']:.5f}", "hot", 0.0, max(res["pc_sq_err_map"].max(), 1e-6)),
            ],
        ]

        for row_idx, row in enumerate(row_plots):
            for col_idx, (data, title, cmap, vmin, vmax) in enumerate(row):
                ax = axes[row_idx, col_idx]
                self._draw_pitch(ax, x_min, x_max, y_min, y_max)
                im = ax.imshow(
                    np.flipud(data), extent=[x_min, x_max, y_min, y_max], cmap=cmap,
                    alpha=0.8, interpolation="bicubic", aspect="auto", vmin=vmin, vmax=vmax
                )
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(title, color="white", fontsize=11, fontweight="bold")

        fig.suptitle(
            f"OBSO: MAE={res['obso_mae']:.3e}  MaxAbsErr={res['obso_max_abs_err']:.3e}   |   "
            f"PC: MAE={res['pc_mae']:.5f}  MaxAbsErr={res['pc_max_abs_err']:.5f}",
            color="white", fontsize=11
        )

        plt.tight_layout(pad=1.4, rect=[0, 0, 1, 0.96])
        canvas = FigureCanvasTkAgg(fig, master=self.content_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def run(self):
        self.root.mainloop()


def main():
    available_matches = discover_available_matches()
    if not available_matches:
        print(f"No loadable matches found in {MATCH_DATA_PATH} "
              f"(need match_<id>_players.csv, match_<id>_ball.csv and <id>.json for each match).")
        return

    match_id = sys.argv[1] if len(sys.argv) > 1 else ("3812" if "3812" in available_matches else available_matches[0])
    if match_id not in available_matches:
        print(f"Match {match_id} is missing required data files. Available matches: {', '.join(available_matches)}")
        return

    print("Launching OBSO & Pitch Control Comparison GUI...")
    print(f"Available matches: {', '.join(available_matches)} (switchable in the GUI)")
    data = load_match_data(match_id)

    # Load Transition and Score (EPV) grids used by calc_obso().
    # Adjust file names/paths to match your project.
    Transition = load_transition_grid("LaurieOnTracking/Transition_gauss.csv")  # e.g. a 100 x 64 grid
    Score = mepv.load_EPV_grid("LaurieOnTracking/EPV_grid.csv")  # e.g. a 32 x 50 grid

    app = OBSOPitchControlComparisonGUI(
        data["events"], data["tracking_home"], data["tracking_away"], mpc.default_model_params(),
        data["GK_numbers"], (106.0, 68.0), Transition, Score, n_grid_cells_x=50,
        initial_match_id=match_id, available_matches=available_matches
    )
    app.run()


if __name__ == "__main__":
    main()