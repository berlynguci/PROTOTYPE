import pandas as pd
import numpy as np
from fastapi import HTTPException

from schemas import FieldMapping
from services.dataset_service import parse_order_date_series
from services.distance_service import haversine_km

def _base_reconstruct_from_mapping(
    df: pd.DataFrame, mapping: FieldMapping
) -> pd.DataFrame:
    """
    Builds the common cleaned order-level dataset from mapped CSV columns.

    Purpose:
    - Converts mapped latitude/longitude fields to numeric values.
    - Creates order_id, customer_id, order_date, ETA, rating, and area fields.
    - Removes rows with invalid or missing coordinates.
    - Creates stable depot IDs when no depot_id column is provided.

    Used by:
    - Amazon reconstruction.
    - Zomato reconstruction.
    - Generic uploaded dataset reconstruction.
    """
    out = pd.DataFrame()

    out["depot_lat"] = pd.to_numeric(df[mapping.depot_lat], errors="coerce")
    out["depot_lon"] = pd.to_numeric(df[mapping.depot_lon], errors="coerce")
    out["customer_lat"] = pd.to_numeric(df[mapping.customer_lat], errors="coerce")
    out["customer_lon"] = pd.to_numeric(df[mapping.customer_lon], errors="coerce")

    out["customer_id"] = df[mapping.customer_id].astype(str)
    out["order_id"] = (
        df[mapping.order_id].astype(str)
        if mapping.order_id and mapping.order_id in df.columns
        else out["customer_id"].astype(str)
    )

    date_col = None
    if mapping.order_date_col and mapping.order_date_col in df.columns:
        date_col = mapping.order_date_col
    elif "Order_Date" in df.columns:
        date_col = "Order_Date"
    elif "order_date" in df.columns:
        date_col = "order_date"

    out["order_date"] = (
        parse_order_date_series(df[date_col]) if date_col is not None else pd.NaT
    )

    out["observed_eta_min"] = (
        pd.to_numeric(df[mapping.eta_col], errors="coerce")
        if mapping.eta_col and mapping.eta_col in df.columns
        else np.nan
    )

    out["rating"] = (
        pd.to_numeric(df[mapping.rating_col], errors="coerce")
        if mapping.rating_col and mapping.rating_col in df.columns
        else np.nan
    )

    out["area"] = (
        df[mapping.area_col].astype(str)
        if mapping.area_col and mapping.area_col in df.columns
        else "UNSPECIFIED"
    )

    # Remove unusable coordinates
    out = out.dropna(
        subset=["depot_lat", "depot_lon", "customer_lat", "customer_lon"]
    ).copy()
    out = out[
        (out["depot_lat"] != 0)
        & (out["depot_lon"] != 0)
        & (out["customer_lat"] != 0)
        & (out["customer_lon"] != 0)
    ].copy()

    if out.empty:
        raise HTTPException(
            status_code=400, detail="No valid rows remain after coordinate filtering."
        )

    # Stable depot_id from unique depot coordinate pairs unless an explicit depot ID was mapped
    if mapping.depot_id and mapping.depot_id in df.columns:
        out["depot_id"] = df.loc[out.index, mapping.depot_id].astype(str)
    else:
        depot_keys = (
            out["depot_lat"].round(6).astype(str)
            + "_"
            + out["depot_lon"].round(6).astype(str)
        )
        depot_codes, _ = pd.factorize(depot_keys)
        out["depot_id"] = pd.Series(depot_codes, index=out.index).map(
            lambda x: f"DEPOT-{x+1:03d}"
        )

    return out

def reconstruct_generic_uploaded_dataset(
    df: pd.DataFrame, mapping: FieldMapping
) -> pd.DataFrame:
    """
    Reconstructs any other uploaded dataset using the generic route schema.

    Purpose:
    - Provides fallback support when the file is not recognized as Amazon
      or Zomato.
    - Keeps the upload workflow flexible for other delivery-style datasets.
    """
    out = _base_reconstruct_from_mapping(df, mapping)

    node_keys = (
        out["depot_id"].astype(str)
        + "_"
        + out["customer_lat"].round(5).astype(str)
        + "_"
        + out["customer_lon"].round(5).astype(str)
    )
    node_codes, _ = pd.factorize(node_keys)
    out["customer_node_id"] = pd.Series(node_codes, index=out.index).map(
        lambda x: f"NODE-{x+1:04d}"
    )

    node_name_map = {
        node_id: f"Customer {i+1:04d}"
        for i, node_id in enumerate(
            pd.Series(out["customer_node_id"]).drop_duplicates().tolist()
        )
    }
    out["customer_name"] = out["customer_node_id"].map(node_name_map)

    out["node_order_count"] = out.groupby(["depot_id", "customer_node_id"])[
        "order_id"
    ].transform("count")

    out["direct_depot_customer_km"] = out.apply(
        lambda r: haversine_km(
            float(r["depot_lat"]),
            float(r["depot_lon"]),
            float(r["customer_lat"]),
            float(r["customer_lon"]),
        ),
        axis=1,
    )

    out["is_distance_outlier"] = out["direct_depot_customer_km"] > 50.0
    out["is_routing_eligible"] = ~out["is_distance_outlier"]

    if out["rating"].notna().any():
        out["rating"] = out["rating"].fillna(out["rating"].median())
    else:
        out["rating"] = 4.0

    out["observed_eta_min"] = out["observed_eta_min"].fillna(
        (out["direct_depot_customer_km"] / 18.0) * 60.0 + 8.0
    )

    final = out[
        [
            "order_id",
            "order_date",
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
        ]
    ].copy()

    final.reset_index(drop=True, inplace=True)
    return final