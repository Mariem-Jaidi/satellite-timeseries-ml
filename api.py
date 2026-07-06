# ============================================================
# api.py — FastAPI endpoint wrapper for the wheat phenology pipeline
# Exposes the full pipeline as an HTTP API
#
# Run with: uvicorn api:app --reload (from project root)
# Docs at:  http://localhost:8000/docs
# ============================================================

import sys
import os
import tempfile
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List

# make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ingestion import load_raw_files
from imputation import run_imputation
from curve_fitting import run_curve_fitting
from ml_validation import run_ml_validation
from phenology import run_phenology

# ── App definition ────────────────────────────────────────
app = FastAPI(
    title="Wheat Phenology Pipeline API",
    description=(
        "Automated satellite vegetation index cleaning and "
        "phenological metric extraction. "
        "Upload raw CSV files (one per wheat plot) to run the full pipeline."
    ),
    version="1.0.0"
)

# ── Health check ──────────────────────────────────────────
@app.get("/")
def health_check():
    """Confirms the API is running."""
    return {
        "status": "ok",
        "pipeline": "wheat-phenology",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /",
            "predict": "POST /predict",
            "docs": "GET /docs"
        }
    }

# ── Main prediction endpoint ──────────────────────────────
@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    """
    Runs the full pipeline on uploaded CSV files.

    - Accepts: multiple CSV files (one per wheat plot)
    - Returns: JSON with imputation decisions, RF validation,
               and phenology metrics per plot
    """

    # validate input
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    for f in files:
        if not f.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400,
                detail=f"File '{f.filename}' is not a CSV. Only .csv files accepted."
            )

    # save uploaded files to a temp folder
    tmp_dir = tempfile.mkdtemp()
    for f in files:
        contents = await f.read()
        with open(os.path.join(tmp_dir, f.filename), 'wb') as out:
            out.write(contents)

    # run pipeline
    try:
        # Step 1 — ingestion
        master_df = load_raw_files(data_path=tmp_dir + "/")

        # Step 2 — imputation
        df_imputed, audit_df = run_imputation(master_df)

        # Step 3 — curve fitting
        df_clean = run_curve_fitting(df_imputed)
        df_clean['Date'] = pd.to_datetime(df_clean['Date'])

        # Step 4 — ML validation
        comparison, cv_scores = run_ml_validation(df_clean)

        # Step 5 — phenology
        phenology_df = run_phenology(df_clean)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {type(e).__name__}: {str(e)}"
        )

    # build response
    decisions = df_imputed[
        df_imputed['was_originally_missing']
    ]['decision'].value_counts().to_dict()

    ok_plots = int((phenology_df['phenology_note'] == 'ok').sum())

    response = {
        "summary": {
            "raw_rows": len(master_df),
            "clean_rows": len(df_clean),
            "plots": int(master_df['source_file'].nunique()),
            "varieties": int(master_df['variety'].nunique()),
            "imputation_decisions": decisions,
        },
        "ml_validation": {
            "cv_r2_mean": round(float(cv_scores.mean()), 4),
            "cv_r2_std": round(float(cv_scores.std()), 4),
            "cv_r2_per_fold": [round(float(s), 4) for s in cv_scores],
            "mean_abs_diff_rf_vs_curvefit": round(
                float(comparison['diff_rf_vs_curvefit'].abs().mean()), 4
            ),
        },
        "phenology": {
            "plots_complete": ok_plots,
            "plots_total": len(phenology_df),
            "success_rate": round(ok_plots / len(phenology_df), 4),
            "metrics_per_plot": phenology_df[[
                'source_file', 'variety',
                'SOS', 'POS', 'EOS', 'LOS', 'AUC', 'phenology_note'
            ]].astype(str).to_dict(orient='records'),
        },
        "audit_log": audit_df.astype(str).to_dict(orient='records'),
    }

    return JSONResponse(content=response)