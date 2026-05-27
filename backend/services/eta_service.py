from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from services.distance_service import haversine_km

# ============================================================
# SECTION 7: ETA feature preparation and model training
# Purpose:
# - Builds simple prediction features for delivery time estimation.
# - Trains baseline/enhanced ETA models used for route scoring.
# - Provides training metrics such as MAE, RMSE, and R².
#
# note:
# - ETA prediction supports the routing prototype by estimating travel
#   behavior from distance, rating, and area/cluster information.
# ============================================================

def build_eta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares features used for ETA model training.

    Purpose:
    - Computes direct depot-to-customer distance.
    - Fills missing rating and observed ETA values.
    - Produces a clean feature table for model training.

    Used by:
    - train_eta_models.
    """
    feat = df.copy()
    feat["direct_distance_km"] = [
        haversine_km(r.depot_lat, r.depot_lon, r.customer_lat, r.customer_lon)
        for r in feat.itertuples(index=False)
    ]
    feat["rating"] = feat["rating"].fillna(
        feat["rating"].median() if feat["rating"].notna().any() else 4.0
    )
    feat["observed_eta_min"] = feat["observed_eta_min"].fillna(
        (feat["direct_distance_km"] / 18.0) * 60.0 + 8.0
    )
    return feat

def train_eta_models(df: pd.DataFrame, seed: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Trains ETA prediction models and returns predicted ETA values.

    Purpose:
    - Uses Ridge regression as a simpler baseline-style model.
    - Uses Random Forest as the enhanced prediction model.
    - Returns model performance metrics for reporting.

    note:
    - The routing algorithm mainly depends on distance and workload, but
      ETA prediction provides additional operational context.
    """
    feat = build_eta_features(df)
    target = feat["observed_eta_min"].values
    features = feat[["direct_distance_km", "rating", "area"]]

    numeric = ["direct_distance_km", "rating"]
    categorical = ["area"]

    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )

    ridge = Pipeline([("pre", pre), ("model", Ridge(alpha=1.0))])
    rf = Pipeline(
        [
            ("pre", pre),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=60,
                    max_depth=10,
                    min_samples_leaf=3,
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
        ]
    )

    ridge.fit(features, target)
    rf.fit(features, target)

    pred_baseline = ridge.predict(features)
    pred_enhanced = rf.predict(features)

    metrics = {
        "baseline": {
            "mae": float(mean_absolute_error(target, pred_baseline)),
            "rmse": float(np.sqrt(mean_squared_error(target, pred_baseline))),
            "r2": float(r2_score(target, pred_baseline)),
        },
        "enhanced": {
            "mae": float(mean_absolute_error(target, pred_enhanced)),
            "rmse": float(np.sqrt(mean_squared_error(target, pred_enhanced))),
            "r2": float(r2_score(target, pred_enhanced)),
        },
    }
    return pred_enhanced, metrics
