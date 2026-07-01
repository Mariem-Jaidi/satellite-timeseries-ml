# ============================================================
# curve_fitting.py — upgrades linear interpolation with
# Savitzky-Golay curve fitting for INTERPOLATE rows
# ============================================================

import pandas as pd
from scipy.signal import savgol_filter
from config import INDEX_COLS, SG_WINDOW, SG_POLYORDER


def curve_fit_file(group, window=SG_WINDOW, poly=SG_POLYORDER):
    """
    Applies Savitzky-Golay smoothing to all INDEX_COLS
    within one plot's timeline (one source_file group).
    Automatically shrinks window for short series.
    """
    group = group.sort_values('Date').reset_index(drop=True)
    n_points = len(group)

    # window must be odd and smaller than the number of points available
    actual_window = min(window, n_points if n_points % 2 == 1 else n_points - 1)
    if actual_window < 3:
        actual_window = 3  # smallest meaningful window for polyorder=2

    smoothed = group.copy()
    for col in INDEX_COLS:
        smoothed[col] = savgol_filter(
            group[col],
            window_length=actual_window,
            polyorder=poly
        )

    return smoothed


def run_curve_fitting(df):
    """
    For the 16 INTERPOLATE rows only:
    1. Applies Savitzky-Golay smoothing per source_file
    2. Replaces linear interpolation values with curve-fit values
    3. Returns the upgraded DataFrame
    """
    # identify the rows we're upgrading
    interpolated_mask = (
        (df['was_originally_missing']) &
        (df['decision'] == 'INTERPOLATE')
    )

    # apply curve fitting across all plots
    df_smoothed = df.groupby('source_file', group_keys=False).apply(
        curve_fit_file, include_groups=False
    )
    df_smoothed = df_smoothed.reset_index(drop=True)

    # overwrite only the interpolated rows with curve-fit values
    df_out = df.copy()
    df_out.loc[interpolated_mask, INDEX_COLS] = (
        df_smoothed.loc[interpolated_mask, INDEX_COLS]
    )

    print(f"Curve-fit applied: {interpolated_mask.sum()} rows upgraded")
    return df_out