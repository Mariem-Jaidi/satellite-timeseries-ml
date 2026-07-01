# ============================================================
# phenology.py — extracts seasonal growth metrics per plot
# SOS, POS, EOS, LOS, AUC using amplitude threshold method
# ============================================================

import pandas as pd
import numpy as np
from sqlalchemy import values
from config import (
    PHENOLOGY_AMPLITUDE_THRESHOLD,
    PHENOLOGY_MIN_POINTS,
    PHENOLOGY_TARGET_COLUMN
)


def extract_phenology(group,
                      amplitude_threshold=PHENOLOGY_AMPLITUDE_THRESHOLD,
                      min_points=PHENOLOGY_MIN_POINTS,
                      target_col=PHENOLOGY_TARGET_COLUMN):
    """
    Given one plot's timeline (one source_file group), returns:
    - SOS: Start of Season (date NDVI crosses threshold rising)
    - POS: Peak of Season (date of maximum NDVI)
    - EOS: End of Season (date NDVI crosses threshold falling)
    - LOS: Length of Season in days (EOS - SOS)
    - AUC: Area Under the Curve between SOS and EOS
    - phenology_note: 'ok' or reason for incomplete extraction
    """
    group = group.sort_values('Date').reset_index(drop=True)

    # safety check: not enough points to trust the result
    if len(group) < min_points:
        return pd.Series({
            'SOS': None, 'POS': None, 'EOS': None,
            'LOS': None, 'AUC': None,
            'phenology_note': f'skipped — only {len(group)} points'
        })

    values = group[target_col].values
    dates = group['Date'].values

    # relative threshold: 20% of this plot's own min-max range
    min_val = np.nanmin(values)
    max_val = np.nanmax(values)
    threshold = min_val + amplitude_threshold * (max_val - min_val)

    # POS — date of maximum value
    pos_idx = int(np.nanargmax(values))
    pos_date = dates[pos_idx]

    # SOS — first upward crossing of threshold before POS
    sos_date = None
    for i in range(1, pos_idx + 1):
        if values[i-1] < threshold <= values[i]:
            sos_date = dates[i]
            break

    # EOS — first downward crossing of threshold after POS
    eos_date = None
    for i in range(pos_idx + 1, len(values)):
        if values[i-1] >= threshold > values[i]:
            eos_date = dates[i]
            break

    # LOS and AUC only calculated if both SOS and EOS were found
    los_days = None
    auc = None
    if sos_date is not None and eos_date is not None:
        los_days = (pd.Timestamp(eos_date) - pd.Timestamp(sos_date)).days
        season_mask = (dates >= sos_date) & (dates <= eos_date)
        auc = np.trapezoid(
            values[season_mask],
            x=pd.to_datetime(dates[season_mask]).astype('int64') / 1e9 / 86400
        )

    return pd.Series({
        'SOS': sos_date,
        'POS': pos_date,
        'EOS': eos_date,
        'LOS': los_days,
        'AUC': auc,
        'phenology_note': 'ok' if (sos_date is not None and eos_date is not None)
                          else 'incomplete — SOS or EOS not found within data range'
    })


def run_phenology(df):
    """
    Applies extract_phenology to every plot (source_file) independently.
    Attaches variety name for readability.
    Returns a summary DataFrame: one row per plot, columns = metrics.
    """
    # ensure Date is datetime before any groupby/comparison
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    results = df.groupby('source_file', group_keys=True).apply(
        extract_phenology, include_groups=False
    )
    results = results.reset_index()

    # attach variety name — one variety per source_file, grab the first
    variety_lookup = df.groupby('source_file')['variety'].first()
    results['variety'] = results['source_file'].map(variety_lookup)

    # reorder columns cleanly
    results = results[[
        'source_file', 'variety',
        'SOS', 'POS', 'EOS', 'LOS', 'AUC',
        'phenology_note'
    ]]

    # summary stats
    ok_count = (results['phenology_note'] == 'ok').sum()
    incomplete_count = len(results) - ok_count
    print(f"Phenology extracted: {ok_count}/{len(results)} plots complete")
    if incomplete_count > 0:
        print(f"Incomplete plots ({incomplete_count}):")
        print(results[results['phenology_note'] != 'ok'][
            ['source_file', 'variety', 'phenology_note']
        ].to_string())

    return results