import pandas as pd

from schemas import FieldMapping
from services.base_reconstruction_service import _base_reconstruct_from_mapping
from services.distance_service import haversine_km

def reconstruct_raw_zomato_dataset(
    df: pd.DataFrame, mapping: FieldMapping
) -> pd.DataFrame:
    """
    Reconstructs a raw Zomato delivery dataset into the route-ready schema.

    Purpose:
    - Uses Delivery_person_ID as the real agent identifier when available.
    - Builds customer_node_id from destination coordinates.
    - Computes distance, demand count, rating, ETA, and eligibility fields.

    note:
    - Unlike Amazon, Zomato can provide a direct delivery person/agent ID.
    """
    out = _base_reconstruct_from_mapping(df, mapping)

    agent_col = None
    if mapping.agent_id and mapping.agent_id in df.columns:
        agent_col = mapping.agent_id
    elif "Delivery_person_ID" in df.columns:
        agent_col = "Delivery_person_ID"
    elif "delivery_person_id" in df.columns:
        agent_col = "delivery_person_id"

    if agent_col:
        out["agent_id"] = df.loc[out.index, agent_col].astype(str).fillna("UNKNOWN")
        out["agent_id"] = out["agent_id"].replace(
            {"": "UNKNOWN", "nan": "UNKNOWN", "None": "UNKNOWN"}
        )
    else:
        out["agent_id"] = "UNKNOWN"

    # For Zomato, reconstruct node identity from destination coordinates within depot
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

    # Readable UI names
    node_name_map = {
        node_id: f"Customer {i+1:04d}"
        for i, node_id in enumerate(
            pd.Series(out["customer_node_id"]).drop_duplicates().tolist()
        )
    }
    out["customer_name"] = out["customer_node_id"].map(node_name_map)

    # Demand / repeated orders per node
    out["node_order_count"] = out.groupby(["depot_id", "customer_node_id"])[
        "order_id"
    ].transform("count")

    # Direct distance
    out["direct_depot_customer_km"] = out.apply(
        lambda r: haversine_km(
            float(r["depot_lat"]),
            float(r["depot_lon"]),
            float(r["customer_lat"]),
            float(r["customer_lon"]),
        ),
        axis=1,
    )

    # Same initial outlier threshold for consistency
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
            "agent_id",
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