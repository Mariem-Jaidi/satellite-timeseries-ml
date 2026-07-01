# INRAT Béja Wheat Phenology Pipeline

An end-to-end automated ML pipeline for cleaning satellite-derived vegetation 
index data and extracting phenological metrics from wheat plots in the Béja 
region, Tunisia.

Built during an internship at **INRAT** (Institut National de la Recherche 
Agronomique de Tunisie), satellite remote sensing department.

---

## What This Project Does

Satellite imagery of wheat fields produces time-series data for vegetation 
indices (NDVI, EVI, NDRE, GNDVI, SAVI) — but cloud cover regularly corrupts 
or eliminates readings entirely. Previously, deciding what to do with each 
missing value required manual judgment. This pipeline replaces that process 
with an automated, statistically grounded, and fully reproducible workflow.

**Input:** 29 raw CSV files (one per wheat plot), 8 varieties, Béja region  
**Output:** Clean dataset + audit log + RF validation report + phenology metrics

---

## Pipeline Steps

| Step | Module | What it does |
|------|--------|--------------|
| 1 | `ingestion.py` | Reads all raw CSVs, extracts variety from filename, merges into master DataFrame |
| 2 | `imputation.py` | Deduplicates dates, applies three-tier decision rule (DELETE / INTERPOLATE / FLAG) based on cloud cover % |
| 3 | `curve_fitting.py` | Upgrades linear interpolation with Savitzky-Golay curve fitting for INTERPOLATE rows |
| 4 | `ml_validation.py` | Trains a Random Forest on trusted rows, independently predicts interpolated values as a cross-check |
| 5 | `phenology.py` | Extracts SOS, POS, EOS, LOS, AUC per plot using the 20% amplitude threshold method |

---

## Key Results

- **696 raw rows** across 29 plots → **667 clean rows** after pipeline
- **41 missing observations** intelligently triaged: 16 interpolated, 24 flagged, 0 deleted
- **Savitzky-Golay curve fitting** upgraded linear interpolation for 16 rows
- **Random Forest cross-validation** (5-fold) independently validated interpolation quality:
  - Mean R² = 0.994 (std = 0.0017)
  - Mean absolute difference vs curve-fit: 0.018 NDVI units
- **Phenology extracted** for 28/29 plots (96.5% success rate)
  - Mean season length (LOS): 161 days (range: 115–200 days)
  - 1 plot (carioka) incomplete due to data boundary — documented, not a bug

---

## Project Structure
inrat_beja_pipeline/
├── data/
│   ├── raw/                  # original CSV files (one per plot)
│   └── processed/            # cleaned outputs
│       ├── master_beja.csv   # merged raw data
│       └── cleaned_beja.csv  # fully cleaned, curve-fit dataset
├── notebooks/
│   ├── 01_exploration.ipynb      # data ingestion + profiling + MNAR proof
│   ├── 02_imputation.ipynb       # decision rule + interpolation + curve fitting
│   ├── 03_ml_validation.ipynb    # Random Forest cross-validation
│   └── 04_phenology.ipynb        # phenology extraction
├── reports/
│   ├── missing_data_analysis.png    # 4-panel missing data profiling chart
│   ├── imputation_audit_log.csv     # per-row decision log
│   ├── rf_validation_comparison.csv # RF vs curve-fit comparison table
│   └── phenology_metrics.csv        # SOS/POS/EOS/LOS/AUC per plot
├── src/
│   ├── config.py          # all thresholds and settings —editherefor new datasets
│   ├── ingestion.py       # file loading and variety extraction
│   ├── imputation.py      # decision rule, deduplication, interpolation
│   ├── curve_fitting.py   # Savitzky-Golay curve fitting
│   ├── ml_validation.py   # Random Forest training and cross-validation
│   ├── phenology.py       # phenology metric extraction
│   └── pipeline.py        # end-to-end orchestrator
├── journal.md             # development log
└── README.md

---

## How to Run

**Requirements:**
```bash
pip install pandas numpy scipy scikit-learn matplotlib
```

**Run the full pipeline:**
```bash
cd src
python pipeline.py
```

**To adapt to a new dataset:**
1. Place new CSV files in `data/raw/`
2. Update `src/config.py` if column names or thresholds differ
3. Run `python pipeline.py`

No other code changes required.

---

## Tech Stack

- **Python 3.11**
- **pandas** — data loading, merging, groupby operations
- **numpy** — array operations
- **scipy** — Savitzky-Golay filter (`savgol_filter`), t-test (`stats.ttest_ind`)
- **scikit-learn** — Random Forest Regressor, LabelEncoder, cross-validation
- **matplotlib** — missing data profiling visualizations

---

## Data

Raw data provided by INRAT (Institut National de la Recherche Agronomique 
de Tunisie). Satellite-derived vegetation indices processed from Sentinel-2 
imagery over the Béja region, Tunisia, 2024–2025 wheat season.

---

## Author

**Mariem** — BI & IT student, IHEC Carthage  
Internship at INRAT, satellite remote sensing department, 2025