# INRAT Internship — Project Journal

## Day 1 — [fill in today's date]

**Goal today:** Set up project environment and explore raw data structure.

**What I did:**
- Set up project folder structure (data/raw, data/processed, notebooks, src, reports)
- Installed Python libraries: pandas, openpyxl, numpy, matplotlib, seaborn, missingno, jupyter
- Loaded all raw Excel files from the satellite remote sensing dataset (Béja region)
- Confirmed 29 files total, covering 8+ wheat varieties: dhahbi, khiar, inrat100, carioka, karim, maali, saragolla
- Each file contains: Date, Nuages_% (cloud cover), and 5 vegetation indices (NDVI, EVI, NDRE, GNDVI, SAVI)

**Observations:**
- Files are small time series (~18-25 rows each), one row per satellite pass
- Initial visual inspection shows missing values are NOT random: when Nuages_% (cloud cover) is high, vegetation index columns are empty
- This suggests missingness is MNAR (Missing Not At Random) — caused by cloud cover obstructing the satellite sensor, not random sensor failure
- This insight will drive the core design of the imputation pipeline: cloud cover will be used as a feature to decide whether to delete, interpolate, or flag missing values

**Next step:** Build ingestion script to merge all 29 files into one master dataframe, tagging each row with its wheat variety.

---
Day 1 (cont'd): 
**built ingestion script, merged 29 files into master dataframe of N rows across 8 varieties
**Learned regex basics for filename parsing — \d+ matches digits, $ anchors to end of string, _? makes the underscore optional
+Hit ValueError: No objects to concatenate — learned this means the file-loading loop failed silently for every file; debugging by isolating the first real error instead of letting output get truncated
+Learned that Jupyter cells don't auto-rerun dependencies — variables can go stale after a kernel restart. Habit: use Restart & Run All when debugging weird state issues


## Day 2 — [18/06/2026]

**What I accomplished:**
- Successfully merged all 29 raw CSV files into one master dataset (1368 rows, 10 columns)
- 8 wheat varieties confirmed: carioka, dhahbi, inrat100, karim, khiar, maali, salim, saragola
- Saved master dataset to data/processed/master_beja.csv

**Key scientific findings:**
- All 5 vegetation indices (NDVI, EVI, NDRE, GNDVI, SAVI) have exactly 40 missing 
  rows each — always the same rows, never individual columns. This means entire 
  satellite passes are lost when cloud cover is too high, not random sensor failures.
- Statistical proof (t-test, p ≈ 0.000000):
  - Average cloud cover when data is PRESENT: 5.1%
  - Average cloud cover when data is MISSING: 10.7%
  - This CONFIRMS the MNAR pattern (Missing Not At Random)
  - Cloud cover (Nuages_%) is the statistically proven cause of missingness
  - This becomes the primary decision feature for the imputation pipeline

**What this means for the pipeline:**
- We don't need to guess WHY data is missing — we can measure it directly via Nuages_%
- The imputation strategy will use cloud cover thresholds to decide:
  delete vs interpolate vs flag

**Visualization produced:**
- reports/missing_data_analysis.png — 4-panel figure showing:
  1. Missing % per wheat variety
  2. Cloud cover distribution when data present vs missing  
  3. Scatter: cloud cover vs NDVI
  4. Missingness heatmap across all varieties and indices

**Next step:** Build the intelligent imputation pipeline using Nuages_% thresholds
--> what did i do in a nutshelll?
I started by writing an ingestion pipeline that automatically reads all 29 raw files, extracts the wheat variety from the filename, and merges everything into one unified dataset of 1368 observations across 8 varieties. Then I profiled the missing data and found that all 5 vegetation indices are always missing simultaneously — never individually — which suggested cloud cover as the cause rather than random sensor error. I confirmed this statistically using a t-test: average cloud cover is 5.1% when data is present versus 10.7% when it's missing, with a p-value essentially equal to zero. This proves the missingness is MNAR — caused directly by cloud cover — which means I can use the Nuages_% column as the primary decision feature for the intelligent imputation engine I'm building next

Statistical foundation: used independent samples t-test to prove cloud cover causes missingness. p-value ≈ 0 confirms MNAR pattern. This justifies using Nuages_% as the primary decision threshold in the imputation engine: >80% cloud = delete, 20-80% = interpolate, <20% but missing = flag as anomaly
## Day 2 (correction) — [today's date]

**Critical data integrity fix:**
- Discovered indices_nuages_maali.csv was a self-created combined file from 
  earlier experimentation, containing all 8 varieties mislabeled as "maali" 
  (696 rows incorrectly tagged)
- Replaced with genuine per-variety files (maali0.csv through maali5.csv)
- Re-ran full ingestion pipeline with corrected 29 raw files

**Corrected baseline (final, trustworthy numbers):**
- Total rows: 696 (corrected from incorrect 1368)
- 8 varieties confirmed: carioka (1 file), dhahbi (6 files), inrat100 (3 files), 
  karim (2 files), khiar (6 files), maali (6 files), salim (4 files), saragola (1 file)
- All 29 files accounted for, no duplicates or mislabeling

**Lesson learned:** Always inspect unexpected columns in merged data, even ones 
that look mostly empty — they can reveal hidden data integrity issues. A single 
corrupted source file can silently corrupt half a dataset if not caught early.
## Day 2 (continued) — Core Pipeline Complete

**Built and tested the full three-tier imputation engine:**
- Config-driven thresholds (single cell, easy to update once supervisor confirms 
  real INRAT values)
- decide_action() function classifies each missing row: DELETE / INTERPOLATE / FLAG
- Time-aware linear interpolation (pandas .interpolate(method='time')) per 
  individual plot file, not per variety — critical fix after discovering each 
  source_file represents an independent time series
- Flagged rows correctly left as NaN, not silently filled

**Bugs fixed along the way:**
- Discovered and removed a self-created corrupted "maali" file that was 
  mislabeling 696 rows
- Discovered duplicate variety+date combinations across files — root cause: 
  multiple plot replicates per variety, each needing independent interpolation
- Fixed duplicate single-date entries (Dec 13 appeared twice in every file) 
  via averaging
- Fixed mixed date formats (some dashes, some slashes) using format='mixed'
- Fixed non-unique index after groupby().apply() using reset_index(drop=True)

**Results:**
- Cleaned dataset: 667 rows (down from 696 after deduplication)
- 16 rows interpolated, 24 rows flagged, 0 deleted (no row reached 80% threshold 
  — pending real INRAT threshold confirmation)
- Saved: cleaned_beja.csv, imputation_audit_log.csv

**Next step:** Get real cloud cover thresholds from supervisor, then build 
Layer 1 (curve-fitted interpolation) and Layer 2 (ML cross-variety validation)