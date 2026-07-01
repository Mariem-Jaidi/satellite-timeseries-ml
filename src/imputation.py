# ============================================================
# imputation.py — cleans and imputes missing vegetation index values
# ============================================================

import pandas as pd
import numpy as np
from config import (
    CLOUD_THRESHOLD_DELETE,
    CLOUD_THRESHOLD_INTERPOLATE,
    INDEX_COLS
)


def deduplicate_dates(df):
    """
    Averages rows that share the same source_file + variety + Date.
    Handles the Béja-specific dual satellite pass issue (e.g. 2024-12-13
    appearing twice in every file).
    """
    df['Date'] = pd.to_datetime(df['Date'], format='mixed')

    df_deduped = df.groupby(
        ['source_file', 'variety', 'Date'], as_index=False
    ).agg({
        'Nuages_%': 'mean',
        'NDVI': 'mean',
        'EVI': 'mean',
        'NDRE': 'mean',
        'GNDVI': 'mean',
        'SAVI': 'mean'
    })

    df_deduped = df_deduped.sort_values(
        by=['source_file', 'Date']
    ).reset_index(drop=True)

    return df_deduped


def decide_action(cloud_pct):
    """
    Three-tier decision rule based on cloud cover percentage:
    - above CLOUD_THRESHOLD_DELETE    → DELETE (image too obstructed)
    - above CLOUD_THRESHOLD_INTERPOLATE → INTERPOLATE (estimate from neighbors)
    - below CLOUD_THRESHOLD_INTERPOLATE → FLAG (missing despite clear sky, needs review)
    """
    if cloud_pct > CLOUD_THRESHOLD_DELETE:
        return 'DELETE'
    elif cloud_pct >= CLOUD_THRESHOLD_INTERPOLATE:
        return 'INTERPOLATE'
    else:
        return 'FLAG'


def interpolate_file(group):
    """
    Applies linear time interpolation to all INDEX_COLS
    within one plot's timeline (one source_file group).
    """
    group = group.set_index('Date')
    group[INDEX_COLS] = group[INDEX_COLS].interpolate(method='time')
    group = group.reset_index()
    return group


def build_audit_log(df, originally_missing):
    """
    Builds a DataFrame logging the decision made for every
    originally-missing row: source_file, variety, date,
    cloud cover, and decision.
    """
    audit_log = []
    for idx in df[originally_missing].index:
        row = df.loc[idx]
        audit_log.append({
            'source_file': row['source_file'],
            'variety': row['variety'],
            'date': row['Date'],
            'cloud_cover': row['Nuages_%'],
            'decision': row['decision']
        })
    return pd.DataFrame(audit_log)


def run_imputation(master_df):
    """
    Full imputation pipeline:
    1. Deduplicate dates
    2. Track originally-missing rows
    3. Apply linear interpolation per plot
    4. Apply three-tier decision rule
    5. Undo interpolation for FLAG rows (leave as NaN)
    6. Delete DELETE rows
    7. Return cleaned DataFrame + audit log
    """
    # Step 1 — deduplicate
    df = deduplicate_dates(master_df)

    # Step 2 — track which rows were originally missing
    originally_missing_mask = df['NDVI'].isna()
    df['was_originally_missing'] = originally_missing_mask
    df['decision'] = df['Nuages_%'].apply(decide_action)

    # Step 3 — linear interpolation per source_file
    # we interpolate FIRST, then selectively undo for FLAG/DELETE
    df_interp = df.groupby('source_file', group_keys=True).apply(
        interpolate_file, include_groups=False
    )
    df_interp = df_interp.reset_index(level=0)
    df_interp = df_interp.reset_index(drop=True)

    # Step 4 — restore was_originally_missing and decision onto interpolated df
    # groupby/apply can drop or misalign non-grouped columns, so we re-attach them
    # using source_file + Date as the join key (guaranteed unique after dedup)
    key_cols = df[['source_file', 'Date', 'was_originally_missing', 'decision']].copy()
    df_interp['Date'] = pd.to_datetime(df_interp['Date'])
    key_cols['Date'] = pd.to_datetime(key_cols['Date'])

    df_interp = df_interp.drop(
        columns=['was_originally_missing', 'decision'], errors='ignore'
    )
    df_interp = df_interp.merge(key_cols, on=['source_file', 'Date'], how='left')

    # Step 5 — undo interpolation for FLAG rows only
    flag_mask = (df_interp['was_originally_missing']) & (df_interp['decision'] == 'FLAG')
    df_interp.loc[flag_mask, INDEX_COLS] = np.nan

    # Step 6 — remove DELETE rows
    delete_mask = (df_interp['was_originally_missing']) & (df_interp['decision'] == 'DELETE')
    df_interp = df_interp.drop(index=df_interp[delete_mask].index).reset_index(drop=True)

    # Step 7 — build audit log
    audit_df = build_audit_log(df_interp, df_interp['was_originally_missing'])

    return df_interp, audit_df