# ============================================================
# app.py — Streamlit web interface for the wheat phenology pipeline
# Run with: streamlit run app.py (from project root)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import tempfile
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ingestion import load_raw_files
from imputation import run_imputation
from curve_fitting import run_curve_fitting
from ml_validation import run_ml_validation
from phenology import run_phenology

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Wheat Phenology Pipeline",
    page_icon="🌾",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    /* main background */
    .stApp { background-color: #0F1117; }

    /* metric cards */
    [data-testid="metric-container"] {
        background: #1C1F26;
        border: 1px solid #2A2D35;
        border-radius: 8px;
        padding: 16px;
    }

    /* accent on metric values */
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #4CAF50;
        font-size: 2rem;
        font-family: monospace;
    }

    /* step tracker */
    .step-done   { color: #4CAF50; font-family: monospace; font-size: 0.95rem; }
    .step-active { color: #FFC107; font-family: monospace; font-size: 0.95rem; }
    .step-wait   { color: #555;    font-family: monospace; font-size: 0.95rem; }

    /* section headers */
    h2 { color: #4CAF50 !important; letter-spacing: 0.05em; }
    h3 { color: #E0E0E0 !important; }

    /* table */
    .dataframe { font-size: 0.82rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────
st.markdown("## 🌾 Wheat Phenology Pipeline")
st.markdown(
    "Upload your raw satellite CSV files — one per wheat plot — "
    "and the pipeline will clean, validate, and extract phenological "
    "metrics automatically."
)
st.divider()

# ── Upload ────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload raw CSV files (one per plot)",
    type="csv",
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload your CSV files above to begin.")
    st.stop()

st.success(f"{len(uploaded_files)} files uploaded.")

# ── Run button ────────────────────────────────────────────
run = st.button("▶ Run Pipeline", type="primary", use_container_width=True)

if not run:
    st.stop()

# ── Step tracker UI ───────────────────────────────────────
st.divider()
st.markdown("### Pipeline progress")

steps = [
    "Ingestion       — loading and merging raw files",
    "Imputation      — deduplication + decision rule + interpolation",
    "Curve fitting   — Savitzky-Golay upgrade for INTERPOLATE rows",
    "ML validation   — Random Forest cross-check",
    "Phenology       — SOS / POS / EOS / LOS / AUC extraction",
]

placeholders = []
for i, step in enumerate(steps):
    p = st.empty()
    p.markdown(f'<p class="step-wait">○ [{i+1}/5] {step}</p>', unsafe_allow_html=True)
    placeholders.append(p)

def mark_active(i):
    placeholders[i].markdown(
        f'<p class="step-active">◉ [{i+1}/5] {steps[i]} …</p>',
        unsafe_allow_html=True
    )

def mark_done(i, note=""):
    label = f"{steps[i]}  {note}" if note else steps[i]
    placeholders[i].markdown(
        f'<p class="step-done">✓ [{i+1}/5] {label}</p>',
        unsafe_allow_html=True
    )

# ── Save uploads to temp folder ───────────────────────────
tmp_dir = tempfile.mkdtemp()
for f in uploaded_files:
    with open(os.path.join(tmp_dir, f.name), "wb") as out:
        out.write(f.read())

# ── Step 1 — Ingestion ────────────────────────────────────
mark_active(0)
master_df = load_raw_files(data_path=tmp_dir + "/")
mark_done(0, f"→ {len(master_df)} rows, "
             f"{master_df['source_file'].nunique()} plots, "
             f"{master_df['variety'].nunique()} varieties")

# ── Step 2 — Imputation ───────────────────────────────────
mark_active(1)
df_imputed, audit_df = run_imputation(master_df)
decisions = df_imputed[df_imputed['was_originally_missing']]['decision'].value_counts().to_dict()
mark_done(1, f"→ {len(df_imputed)} rows remaining | {decisions}")

# ── Step 3 — Curve fitting ────────────────────────────────
mark_active(2)
df_clean = run_curve_fitting(df_imputed)
df_clean['Date'] = pd.to_datetime(df_clean['Date'])
interp_count = int((df_imputed['was_originally_missing'] &
                    (df_imputed['decision'] == 'INTERPOLATE')).sum())
mark_done(2, f"→ {interp_count} rows upgraded")

# ── Step 4 — ML validation ────────────────────────────────
mark_active(3)
comparison, cv_scores = run_ml_validation(df_clean)
mark_done(3, f"→ CV R²={cv_scores.mean():.4f} | "
             f"mean |diff|={comparison['diff_rf_vs_curvefit'].abs().mean():.4f}")

# ── Step 5 — Phenology ────────────────────────────────────
mark_active(4)
phenology_df = run_phenology(df_clean)
ok_count = (phenology_df['phenology_note'] == 'ok').sum()
mark_done(4, f"→ {ok_count}/{len(phenology_df)} plots complete")

# ── Results dashboard ─────────────────────────────────────
st.divider()
st.markdown("## Results")

# key metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Raw rows", len(master_df))
col2.metric("Clean rows", len(df_clean))
col3.metric("CV R²", f"{cv_scores.mean():.4f}")
col4.metric("Phenology success", f"{ok_count}/{len(phenology_df)}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Audit log", "RF Validation", "Phenology metrics", "NDVI curves"
])

with tab1:
    st.markdown("### Imputation decisions — per originally-missing row")
    st.dataframe(audit_df, use_container_width=True)

with tab2:
    st.markdown("### Random Forest vs curve-fit comparison")
    st.markdown(
        f"Mean CV R² = **{cv_scores.mean():.4f}** (std {cv_scores.std():.4f})  \n"
        f"Mean absolute difference = **{comparison['diff_rf_vs_curvefit'].abs().mean():.4f}** NDVI units"
    )
    st.dataframe(comparison, use_container_width=True)

with tab3:
    st.markdown("### Phenology metrics — SOS / POS / EOS / LOS / AUC per plot")
    st.dataframe(phenology_df, use_container_width=True)

with tab4:
    st.markdown("### NDVI curves — pick a plot")
    plot_files = sorted(df_clean['source_file'].unique())
    selected = st.selectbox("Select plot", plot_files)

    plot_data = df_clean[df_clean['source_file'] == selected].sort_values('Date')

    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#1C1F26')
    ax.set_facecolor('#1C1F26')
    ax.plot(plot_data['Date'], plot_data['NDVI'],
            marker='o', color='#4CAF50', linewidth=2)
    ax.set_title(f'NDVI — {selected}', color='#E0E0E0')
    ax.set_xlabel('Date', color='#888')
    ax.set_ylabel('NDVI', color='#888')
    ax.tick_params(colors='#888')
    ax.spines['bottom'].set_color('#2A2D35')
    ax.spines['left'].set_color('#2A2D35')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.1, color='#4CAF50')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Downloads ─────────────────────────────────────────────
st.divider()
st.markdown("### Download results")

dcol1, dcol2, dcol3 = st.columns(3)

dcol1.download_button(
    "⬇ cleaned_beja.csv",
    df_clean.to_csv(index=False).encode(),
    file_name="cleaned_beja.csv",
    mime="text/csv",
    use_container_width=True
)

dcol2.download_button(
    "⬇ phenology_metrics.csv",
    phenology_df.to_csv(index=False).encode(),
    file_name="phenology_metrics.csv",
    mime="text/csv",
    use_container_width=True
)

dcol3.download_button(
    "⬇ rf_validation.csv",
    comparison.to_csv(index=False).encode(),
    file_name="rf_validation.csv",
    mime="text/csv",
    use_container_width=True
)