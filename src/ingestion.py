# ============================================================
# ingestion.py — reads and merges all raw CSV plot files
# ============================================================

import os
import re
import pandas as pd
from config import RAW_DATA_PATH, PROTECTED_VARIETY_NAMES


def extract_variety(filename):
    """
    Derives the wheat variety name from a raw CSV filename.
    e.g. 'indices_nuages_dhahbi2.csv' -> 'dhahbi'
         'indices_nuages_inrat100_1.csv' -> 'inrat100'  (protected name, not stripped)
    """
    name = filename.replace('.csv', '')
    name = name.replace('indices_nuages_', '')

    # protected names contain numbers that are part of the variety name itself
    # e.g. inrat100 — the '100' must not be stripped by the regex below
    for protected in PROTECTED_VARIETY_NAMES:
        if name.startswith(protected):
            return protected

    # strip trailing replicate number (e.g. dhahbi2 -> dhahbi, karim1 -> karim)
    variety = re.sub(r'(_?\d+)$', '', name)
    return variety


def load_raw_files(data_path=RAW_DATA_PATH):
    """
    Loads all CSV files from data_path, attaches variety and source_file columns,
    merges into one master DataFrame, and returns it.
    """
    files = [f for f in os.listdir(data_path) if f.endswith('.csv')]

    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_path}")

    all_dfs = []
    errors = []

    for f in files:
        try:
            df = pd.read_csv(os.path.join(data_path, f))
            df['variety'] = extract_variety(f)
            df['source_file'] = f
            all_dfs.append(df)
        except Exception as e:
            errors.append(f"ERROR loading {f}: {type(e).__name__}: {e}")

    if errors:
        print("--- Ingestion warnings ---")
        for err in errors:
            print(err)

    master_df = pd.concat(all_dfs, ignore_index=True)
    return master_df