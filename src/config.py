# ============================================================
# config.py — all pipeline settings in one place
# To adapt to a new dataset: only edit values in this file
# ============================================================

# --- Paths ---
RAW_DATA_PATH = "../data/raw/"
PROCESSED_DATA_PATH = "../data/processed/"
REPORTS_PATH = "../reports/"

# --- Ingestion ---
PROTECTED_VARIETY_NAMES = ['inrat100']  # variety names that contain numbers — don't strip them

# --- Imputation thresholds ---
# TODO: replace with INRAT's real growth-curve-based thresholds once confirmed by supervisor
CLOUD_THRESHOLD_DELETE = 80       # above this % cloud cover → delete the row
CLOUD_THRESHOLD_INTERPOLATE = 8   # between this and DELETE → interpolate
                                   # below this → flag for manual review

# --- Vegetation index columns ---
INDEX_COLS = ['NDVI', 'EVI', 'NDRE', 'GNDVI', 'SAVI']

# --- Curve fitting (Savitzky-Golay) ---
SG_WINDOW = 7      # number of neighboring points considered per fit
SG_POLYORDER = 2   # degree of polynomial (2 = quadratic, one bend allowed)

# --- ML validation (Random Forest) ---
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 10
RF_RANDOM_STATE = 42
RF_CV_FOLDS = 5
RF_FEATURE_COLS = ['EVI', 'NDRE', 'GNDVI', 'SAVI', 'Nuages_%', 'day_of_year', 'variety_encoded']

# --- Phenology extraction ---
PHENOLOGY_AMPLITUDE_THRESHOLD = 0.20  # 20% of min-max range defines SOS/EOS crossing
PHENOLOGY_MIN_POINTS = 8              # skip plots with fewer points than this
PHENOLOGY_TARGET_COLUMN = 'NDVI'      # which index drives phenology detection