# ============================================================
# pipeline.py — end-to-end orchestrator
# Runs the full INRAT Béja wheat phenology pipeline:
# ingestion → imputation → curve fitting → ML validation → phenology
#
# Usage: run this file directly from the src/ directory:
#   cd src
#   python pipeline.py
# ============================================================

import pandas as pd
from config import PROCESSED_DATA_PATH, REPORTS_PATH
from ingestion import load_raw_files
from imputation import run_imputation
from curve_fitting import run_curve_fitting
from ml_validation import run_ml_validation
from phenology import run_phenology


def run_pipeline(data_path=None):
    """
    Executes the full pipeline end to end.
    Saves all outputs to data/processed/ and reports/.
    """
    from config import RAW_DATA_PATH
    if data_path is None:
        data_path = RAW_DATA_PATH

    print("=" * 55)
    print("INRAT BÉJA WHEAT PIPELINE — starting")
    print("=" * 55)

    # ── Step 1: Ingestion ──────────────────────────────────
    print("\n[1/5] Ingestion — loading raw CSV files...")
    master_df = load_raw_files(data_path=data_path)
    print(f"      Loaded {len(master_df)} rows from "
          f"{master_df['source_file'].nunique()} files, "
          f"{master_df['variety'].nunique()} varieties")

    master_df.to_csv(PROCESSED_DATA_PATH + "master_beja.csv", index=False)
    print("      Saved → data/processed/master_beja.csv")

    # ── Step 2: Imputation ─────────────────────────────────
    print("\n[2/5] Imputation — deduplication + decision rule + interpolation...")
    df_imputed, audit_df = run_imputation(master_df)

    decision_counts = df_imputed[
        df_imputed['was_originally_missing']
    ]['decision'].value_counts()
    print(f"      Decisions: {decision_counts.to_dict()}")
    print(f"      Rows remaining: {len(df_imputed)}")

    audit_df.to_csv(REPORTS_PATH + "imputation_audit_log.csv", index=False)
    print("      Saved → reports/imputation_audit_log.csv")

    # ── Step 3: Curve fitting ──────────────────────────────
    print("\n[3/5] Curve fitting — Savitzky-Golay upgrade for INTERPOLATE rows...")
    df_clean = run_curve_fitting(df_imputed)

    df_clean.to_csv(PROCESSED_DATA_PATH + "cleaned_beja.csv", index=False)
    print("      Saved → data/processed/cleaned_beja.csv")

    # ── Step 4: ML validation ──────────────────────────────
    print("\n[4/5] ML validation — Random Forest cross-check...")
    comparison, cv_scores = run_ml_validation(df_clean)

    comparison.to_csv(REPORTS_PATH + "rf_validation_comparison.csv", index=False)
    print("      Saved → reports/rf_validation_comparison.csv")

    # ── Step 5: Phenology extraction ───────────────────────
    print("\n[5/5] Phenology extraction — SOS/POS/EOS/LOS/AUC per plot...")
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
    phenology_df = run_phenology(df_clean)

    phenology_df.to_csv(REPORTS_PATH + "phenology_metrics.csv", index=False)
    print("      Saved → reports/phenology_metrics.csv")

    # ── Summary ────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("PIPELINE COMPLETE — outputs saved:")
    print("  data/processed/master_beja.csv")
    print("  data/processed/cleaned_beja.csv")
    print("  reports/imputation_audit_log.csv")
    print("  reports/rf_validation_comparison.csv")
    print("  reports/phenology_metrics.csv")
    print("=" * 55)

    return {
        'master_df': master_df,
        'df_clean': df_clean,
        'audit_df': audit_df,
        'comparison': comparison,
        'cv_scores': cv_scores,
        'phenology_df': phenology_df
    }


if __name__ == "__main__":
    run_pipeline()