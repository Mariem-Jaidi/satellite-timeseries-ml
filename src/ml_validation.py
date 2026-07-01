# ============================================================
# ml_validation.py — Random Forest cross-validation of
# interpolated NDVI values (Layer 2)
# ============================================================

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from config import (
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    RF_RANDOM_STATE,
    RF_CV_FOLDS,
    RF_FEATURE_COLS
)


def prepare_features(df):
    """
    Builds feature matrix X and target y from the trusted rows,
    and X_predict for the 16 interpolated rows.
    Returns: X_train, y_train, X_predict, predict_df, le_variety
    """
    df = df.copy()
    df['day_of_year'] = df['Date'].dt.dayofyear

    le_variety = LabelEncoder()
    df['variety_encoded'] = le_variety.fit_transform(df['variety'])

    reliable_mask = ~df['was_originally_missing']
    interpolated_mask = (
        (df['was_originally_missing']) &
        (df['decision'] == 'INTERPOLATE')
    )

    train_df = df[reliable_mask].copy()
    predict_df = df[interpolated_mask].copy()

    X_train = train_df[RF_FEATURE_COLS]
    y_train = train_df['NDVI']
    X_predict = predict_df[RF_FEATURE_COLS]

    return X_train, y_train, X_predict, predict_df, le_variety


def train_model(X_train, y_train):
    """
    Trains a Random Forest Regressor on the trusted rows.
    Returns the trained model.
    """
    model = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RF_RANDOM_STATE
    )
    model.fit(X_train, y_train)
    return model


def run_cross_validation(model, X_train, y_train):
    """
    Runs k-fold cross-validation on the trusted rows.
    Returns array of R² scores per fold.
    """
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=RF_CV_FOLDS,
        scoring='r2'
    )
    return cv_scores


def build_comparison_table(predict_df, rf_predictions):
    """
    Builds a comparison table: curve-fit value vs RF prediction
    vs difference, for the 16 interpolated rows.
    """
    comparison = predict_df[['source_file', 'Date', 'NDVI']].copy()
    comparison = comparison.rename(columns={'NDVI': 'NDVI_curvefit'})
    comparison['NDVI_rf_predicted'] = rf_predictions
    comparison['diff_rf_vs_curvefit'] = (
        comparison['NDVI_rf_predicted'] - comparison['NDVI_curvefit']
    )
    return comparison


def run_ml_validation(df):
    """
    Full ML validation pipeline:
    1. Prepare features
    2. Train Random Forest
    3. Predict on interpolated rows
    4. Run cross-validation
    5. Return comparison table + CV scores
    """
    # Step 1 — features
    X_train, y_train, X_predict, predict_df, _ = prepare_features(df)

    # Step 2 — train
    model = train_model(X_train, y_train)

    # Step 3 — predict on the 16 interpolated rows
    rf_predictions = model.predict(X_predict)

    # Step 4 — cross-validation
    cv_scores = run_cross_validation(model, X_train, y_train)

    # Step 5 — build outputs
    comparison = build_comparison_table(predict_df, rf_predictions)

    mean_diff = comparison['diff_rf_vs_curvefit'].abs().mean()

    print(f"RF cross-validation mean R²: {cv_scores.mean():.4f} (std: {cv_scores.std():.4f})")
    print(f"Mean absolute diff RF vs curve-fit: {mean_diff:.4f}")

    return comparison, cv_scores