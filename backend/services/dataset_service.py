import io
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from fastapi import HTTPException, UploadFile

from backend.config import RUN_PROFILES
from backend.schemas import FieldMapping
from backend.services.distance_service import haversine_km
from backend.services.zomato_service import reconstruct_raw_zomato_dataset
from backend.services.base_reconstruction_service import reconstruct_generic_uploaded_dataset

# ============================================================
# SECTION 6: Dataset upload, normalization, and reconstruction
# Purpose:
# - Reads uploaded CSV files.
# - Infers whether the dataset is Amazon, Zomato, or generic.
# - Converts raw or reconstructed datasets into a common route-ready schema.
#
# note:
# - This section standardizes different public datasets so the same
#   baseline and enhanced routing algorithms can process them.
# ============================================================

def read_csv_upload(file: UploadFile) -> pd.DataFrame:
    """
    Reads the uploaded CSV file into a pandas DataFrame.

    Purpose:
    - Validates that the file is not empty.
    - Converts uploaded bytes into tabular data for processing.

    Used by:
    - Dataset validation endpoint.
    """
    content = file.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Failed to read CSV: {exc}"
        ) from exc


def infer_dataset_role(filename: str) -> str:
    """
    Infers dataset role from the uploaded filename.

    Purpose:
    - Classifies files as Amazon, Zomato, or generic uploads.
    - Allows the backend to apply dataset-specific reconstruction logic.
    """
    name = (filename or "").lower()
    if "amazon" in name:
        return "primary_reconstruction"
    if "zomato" in name:
        return "comparative_template"
    return "generic_uploaded_dataset"


def role_label(role: str) -> str:
    if role == "primary_reconstruction":
        return "Amazon Delivery Dataset (Primary Baseline Reconstruction Source)"
    if role == "comparative_template":
        return "Zomato Delivery Dataset (Comparative Template Dataset)"
    return "Uploaded Delivery Dataset"


def autofill_mapping_from_known_columns(
    df: pd.DataFrame,
    mapping: FieldMapping,
    source_role: str,
) -> FieldMapping:
    """
    Fills optional mapping fields when known dataset columns are detected.

    Purpose:
    - Reduces manual mapping work for common Amazon/Zomato columns.
    - Helps normalize raw uploads even when the frontend mapping is incomplete.
    """
    data = mapping.model_dump()

    columns = set(df.columns)

    if not data.get("order_date_col"):
        if "Order_Date" in columns:
            data["order_date_col"] = "Order_Date"
        elif "order_date" in columns:
            data["order_date_col"] = "order_date"

    if source_role == "comparative_template" and not data.get("agent_id"):
        if "Delivery_person_ID" in columns:
            data["agent_id"] = "Delivery_person_ID"
        elif "delivery_person_id" in columns:
            data["agent_id"] = "delivery_person_id"

    return FieldMapping(**data)


def normalize_dataset(
    df: pd.DataFrame, mapping: FieldMapping, source_role: str
) -> pd.DataFrame:
    """
    Converts uploaded data into the standardized backend routing schema.

    Purpose:
    - Accepts already reconstructed datasets without rebuilding them.
    - Applies dataset-specific reconstruction for raw Amazon and Zomato files.
    - Falls back to generic reconstruction for other uploads.

    Used by:
    - Dataset validation endpoint before baseline/enhanced runs.
    """
    needed = [
        mapping.depot_lat,
        mapping.depot_lon,
        mapping.customer_id,
        mapping.customer_lat,
        mapping.customer_lon,
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400, detail=f"Missing mapped columns: {missing}"
        )

    cleaned_cols = {
        "order_id",
        "customer_id",
        "customer_node_id",
        "depot_id",
        "depot_lat",
        "depot_lon",
        "customer_lat",
        "customer_lon",
        "customer_name",
        "observed_eta_min",
        "rating",
        "area",
        "node_order_count",
        "direct_depot_customer_km",
        "is_distance_outlier",
        "is_routing_eligible",
    }

    if cleaned_cols.issubset(set(df.columns)):
        out = df.copy()

        out["depot_lat"] = pd.to_numeric(out["depot_lat"], errors="coerce")
        out["depot_lon"] = pd.to_numeric(out["depot_lon"], errors="coerce")
        out["customer_lat"] = pd.to_numeric(out["customer_lat"], errors="coerce")
        out["customer_lon"] = pd.to_numeric(out["customer_lon"], errors="coerce")
        out["observed_eta_min"] = pd.to_numeric(
            out["observed_eta_min"], errors="coerce"
        )
        out["rating"] = pd.to_numeric(out["rating"], errors="coerce")
        out["node_order_count"] = pd.to_numeric(
            out["node_order_count"], errors="coerce"
        ).fillna(1)

        if "agent_age" in out.columns:
            out["agent_age"] = (
                pd.to_numeric(out["agent_age"], errors="coerce").fillna(-1).astype(int)
            )

        if "agent_id" not in out.columns:
            if "agent_age" in out.columns:
                out["agent_id"] = (
                    "AGENT-"
                    + out["depot_id"].astype(str)
                    + "-AGE-"
                    + out["agent_age"].astype(str)
                )
            else:
                out["agent_id"] = (
                    "AGENT-" + out["depot_id"].astype(str) + "-AGE-UNKNOWN"
                )

        if "order_date" in out.columns:
            out["order_date"] = parse_order_date_series(out["order_date"])

        out = out.dropna(
            subset=["depot_lat", "depot_lon", "customer_lat", "customer_lon"]
        ).copy()
        out = out[
            (out["customer_lat"] != 0)
            & (out["customer_lon"] != 0)
            & (out["depot_lat"] != 0)
            & (out["depot_lon"] != 0)
        ].copy()

        out.reset_index(drop=True, inplace=True)
        return out

    # dataset-specific raw reconstruction
    if source_role == "primary_reconstruction":
        from backend.services.amazon_service import reconstruct_raw_amazon_dataset
        return reconstruct_raw_amazon_dataset(df, mapping)

    if source_role == "comparative_template":
        return reconstruct_raw_zomato_dataset(df, mapping)

    return reconstruct_generic_uploaded_dataset(df, mapping)


def validation_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Builds the dataset validation summary returned to the frontend.

    Purpose:
    - Counts records, depots, customers, orders, duplicate orders,
      near-duplicate coordinates, and average rating.
    - Confirms the uploaded dataset is usable for routing.
    """
    if df.empty:
        raise HTTPException(
            status_code=400, detail="No valid rows remain after coordinate filtering."
        )
    dup_orders = int(df["order_id"].duplicated().sum())
    invalid = 0
    coords = df[["customer_lat", "customer_lon"]].round(4)
    near_dupes = int(coords.duplicated().sum())

    avg_rating = 4.0
    if df["rating"].notna().any():
        avg_rating = float(df["rating"].fillna(df["rating"].median()).mean())

    return {
        "isValid": True,
        "invalidCoordinates": invalid,
        "duplicateRows": dup_orders,
        "nearDuplicates": near_dupes,
        "summary": {
            "records": int(len(df)),
            "depots": int(df["depot_id"].nunique()),
            "customers": (
                int(df["customer_node_id"].nunique())
                if "customer_node_id" in df.columns
                else int(df["customer_id"].nunique())
            ),
            "orders": int(df["order_id"].nunique()),
            "avgRating": round(avg_rating, 2),
        },
    }

def build_routing_nodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert cleaned order-level rows into routing-node rows.
    One node = one customer_node_id within one depot.
    """
    work = df.copy()

    if "is_routing_eligible" in work.columns:
        work = work[work["is_routing_eligible"].fillna(False).astype(bool)].copy()

    required = [
        "depot_id",
        "depot_lat",
        "depot_lon",
        "customer_node_id",
        "customer_lat",
        "customer_lon",
        "customer_name",
        "observed_eta_min",
        "predicted_eta_min",
        "rating",
        "area",
    ]
    missing = [c for c in required if c not in work.columns]
    if missing:
        raise HTTPException(
            status_code=400, detail=f"Missing cleaned dataset columns: {missing}"
        )

    if "agent_id" not in work.columns:
        work["agent_id"] = "UNKNOWN"

    if "order_date" not in work.columns:
        work["order_date"] = pd.NaT

    agg = work.groupby(
        [
            "depot_id",
            "depot_lat",
            "depot_lon",
            "customer_node_id",
            "customer_lat",
            "customer_lon",
        ],
        as_index=False,
    ).agg(
        order_id=("order_id", "first"),
        order_date=("order_date", "first"),
        customer_id=("customer_id", "first"),
        agent_id=("agent_id", "first"),
        customer_name=("customer_name", "first"),
        observed_eta_min=("observed_eta_min", "mean"),
        predicted_eta_min=("predicted_eta_min", "mean"),
        rating=("rating", "mean"),
        area=("area", "first"),
        node_order_count=("node_order_count", "max"),
        direct_depot_customer_km=("direct_depot_customer_km", "mean"),
    )

    agg["order_id"] = agg["order_id"].astype(str)
    agg["customer_id"] = agg["customer_id"].astype(str)
    agg["customer_node_id"] = agg["customer_node_id"].astype(str)
    agg["depot_id"] = agg["depot_id"].astype(str)
    agg["customer_name"] = agg["customer_name"].fillna(agg["customer_node_id"])
    agg["node_name"] = agg["customer_name"]

    return agg.reset_index(drop=True)

def parse_order_date_series(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()

    # First try day-first parsing, which matches Zomato-style Order_Date better
    parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)

    # Fallback: try default parsing for already ISO-like values
    fallback_mask = parsed.isna()
    if fallback_mask.any():
        parsed.loc[fallback_mask] = pd.to_datetime(
            text.loc[fallback_mask], errors="coerce"
        )

    if parsed.notna().any():
        return parsed.dt.normalize()
    return parsed

def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default

def get_run_profile(profile_name: Optional[str]) -> Dict[str, Any]:
    key = (profile_name or "default_balanced").strip()
    if key not in RUN_PROFILES:
        key = "default_balanced"
    profile = RUN_PROFILES[key].copy()
    profile["profile_name"] = key
    return profile
