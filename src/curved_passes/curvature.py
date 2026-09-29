"""Canonical pass-curvature metric for the curved-pass project.

This module is the single source of truth for the metric used across the
project (World Cup curvature heat map, curvature histograms, and the
empirical priors that feed the pitch-control study). The implementation is
verbatim the one developed during the research; the notebooks in
``notebooks/`` carry inline copies of the same code.

Definition (normalized 3D deviation-area curvature)
---------------------------------------------------
Given a ball trajectory sampled at n points

    r_0, r_1, ..., r_{n-1}      (each r_i = (x_i, y_i, z_i), in meters)

1.  Chord (straight-line passer-to-receiver distance):

        D = || r_{n-1} - r_0 ||

2.  Perpendicular distance of each trajectory point from the chord:

        d_i = || (r_i - r_0) x (r_{n-1} - r_0) || / D

3.  Deviation area between the trajectory and the chord, obtained by
    trapezoidal integration of d along the actual 3D arc length:

        A_dev = sum_i  0.5 * (d_i + d_{i+1}) * || r_{i+1} - r_i ||

4.  Normalized curvature (dimensionless):

        C = A_dev / D^2

C = 0 for a perfectly straight pass and grows with the amount of bend.
Typical completed WC-2022 passes have C < 0.3; the project's working
thresholds are "straight" for C <= 0.02 and "curved" for C > 0.06.

Inputs required
---------------
Per pass: a list of timestamps in **milliseconds** and the ball's (x, y, z)
coordinates per frame, in meters (PFF FC tracking convention). Timestamps
are only used for validation (they must be strictly increasing after
converting ms -> s); the metric itself is purely geometric.
"""
from __future__ import annotations

import ast

import numpy as np
import pandas as pd

__all__ = ["parse_list", "curvature_3d_normalized", "row_curvature"]


def parse_list(value):
    """Parse a stringified Python list from a CSV cell into a Python list.

    Returns an empty list for missing/NaN values and for cells that do not
    parse. Handles the conventions used when the pass table was written
    (``json.dumps``-style stringified lists of floats).
    """
    if value is None or pd.isna(value):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    value = str(value).strip()
    if not value or value.lower() in {"nan", "none", "null", "na", "n/a"}:
        return []
    try:
        parsed = ast.literal_eval(value)
        return list(parsed) if isinstance(parsed, (list, tuple, np.ndarray)) else []
    except (ValueError, SyntaxError):
        return []


def curvature_3d_normalized(timestamps_ms, x, y, z):
    """Normalized 3D deviation-area curvature C = A_dev / D^2.

    Parameters
    ----------
    timestamps_ms : sequence of float
        Frame timestamps in milliseconds (strictly increasing after
        conversion to seconds; otherwise the pass is rejected as NaN).
    x, y, z : sequences of float
        Ball coordinates per frame, in meters.

    Returns
    -------
    float
        The normalized curvature C = A_dev / D^2, or ``np.nan`` when the
        trajectory has fewer than 2 finite samples, non-increasing
        timestamps, or a degenerate (zero-length) chord.
    """
    t = np.asarray(timestamps_ms, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    n = min(len(t), len(x), len(y), len(z))
    if n < 2:
        return np.nan

    # milliseconds -> seconds (kept for timestamp validation only)
    t, x, y, z = t[:n] / 1000.0, x[:n], y[:n], z[:n]

    valid = np.isfinite(t) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if valid.sum() < 2:
        return np.nan
    t, x, y, z = t[valid], x[valid], y[valid], z[valid]

    if np.any(np.diff(t) <= 0):
        return np.nan

    points = np.column_stack((x, y, z))

    # 1) chord length D = distance between passer (first sample) and the
    #    ball position at the receiver (last sample)
    chord_vector = points[-1] - points[0]
    chord_length = np.linalg.norm(chord_vector)
    if chord_length <= 1e-12:
        return np.nan

    # 2) perpendicular distance of every point from the chord
    relative = points - points[0]
    distances = np.linalg.norm(np.cross(relative, chord_vector), axis=1) / chord_length

    # 3) deviation area: trapezoidal integration of d(s) along arc length
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    deviation_area = np.sum(0.5 * (distances[:-1] + distances[1:]) * segment_lengths)

    # 4) normalize by squared straight-line distance
    return deviation_area / (chord_length ** 2)


def row_curvature(row):
    """Apply the metric to one row of ``worldcup2022_passes.csv``.

    The pass table stores each pass's ball trajectory as stringified lists
    in the columns ``frame_timestamps_ms``, ``ball_x``, ``ball_y``,
    ``ball_z``.
    """
    return curvature_3d_normalized(
        parse_list(row["frame_timestamps_ms"]),
        parse_list(row["ball_x"]),
        parse_list(row["ball_y"]),
        parse_list(row["ball_z"]),
    )
