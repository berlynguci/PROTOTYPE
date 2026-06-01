import uuid
from typing import Any, Dict, List

import pandas as pd
from fastapi import HTTPException

from backend.schemas import AddedCustomerPayload
from backend.services.distance_service import haversine_km
from backend.services.metrics_service import compute_thesis_priority_scores
from backend.services.routing_node_service import ensure_preview_node_ids

# ============================================================
# SECTION 11: Add-customer rerouting workflow
# Purpose:
# - Handles customers manually added from the frontend map.
# - Assigns each new customer to the nearest suitable representative.
# - Rebuilds the distance matrix and reroutes the updated baseline result.
#
# note:
# - Added customers are processed sequentially. After one customer is
#   assigned, it becomes part of the route state before the next customer
#   is assigned. This prevents all added customers from automatically
#   going to only one representative.
# ============================================================


def append_added_customers_to_assign_df(
    assign_df: pd.DataFrame,
    customers: List[AddedCustomerPayload],
) -> pd.DataFrame:
    """
    Appends manually added customers to an existing assignment DataFrame.

    Purpose:
    - Creates backend-compatible rows for customers added from the map UI.
    - Assigns generated customer/order/node IDs.
    - Preserves the same schema used by normal routing rows.
    """
    work = ensure_preview_node_ids(assign_df.copy())

    if work.empty:
        raise HTTPException(status_code=400, detail="Baseline preview is empty.")

    depot_lat = float(work.iloc[0]["depot_lat"])
    depot_lon = float(work.iloc[0]["depot_lon"])
    depot_id = str(work.iloc[0]["depot_id"])

    rows: List[Dict[str, Any]] = []

    for idx, customer in enumerate(customers, start=1):
        customer_number = customer.customer_number or (100000 + idx)
        customer_name = f"Customer {customer_number}"
        customer_node_id = f"ADDED-NODE-{uuid.uuid4().hex[:10].upper()}"
        order_id = f"ADDED-ORDER-{uuid.uuid4().hex[:10].upper()}"

        direct_km = haversine_km(
            depot_lat,
            depot_lon,
            float(customer.lat),
            float(customer.lon),
        )

        rows.append(
            {
                "depot_id": depot_id,
                "depot_lat": depot_lat,
                "depot_lon": depot_lon,
                "customer_node_id": customer_node_id,
                "node_id": customer_node_id,
                "customer_id": f"ADDED-CUST-{uuid.uuid4().hex[:10].upper()}",
                "order_id": order_id,
                "order_date": pd.NaT,
                "agent_id": customer.assigned_rep or "UNASSIGNED",
                "customer_name": customer_name,
                "node_name": customer_name,
                "customer_lat": float(customer.lat),
                "customer_lon": float(customer.lon),
                "observed_eta_min": 8.0 + (direct_km / 18.0) * 60.0,
                "predicted_eta_min": 8.0 + (direct_km / 18.0) * 60.0,
                "rating": 4.0,
                "area": "ADDED_CUSTOMER",
                "node_order_count": 1,
                "direct_depot_customer_km": direct_km,
                "rep_id": customer.assigned_rep or "UNASSIGNED",
            }
        )

    added_df = pd.DataFrame(rows)
    combined = pd.concat([work, added_df], ignore_index=True)
    return combined.reset_index(drop=True)

def assign_new_customer_to_nearest_rep(
    assign_df: pd.DataFrame,
    customer_lat: float,
    customer_lon: float,
) -> str:
    """
    Finds the representative route nearest to a newly added customer.

    Purpose:
    - Compares the new customer's coordinate against each representative's
      existing assigned stops.
    - Selects the route with the nearest existing stop.
    - Uses current workload count as a tie-breaker.

    note:
    - This keeps added-customer assignment spatially reasonable instead
      of assigning all new customers to a fixed representative.
    """
    work = ensure_preview_node_ids(assign_df.copy())
    if work.empty or "rep_id" not in work.columns:
        return "UNASSIGNED"

    best_rep = None
    best_distance = None
    best_workload_count = None

    for rep_id, grp in work.groupby("rep_id"):
        rep_id_str = str(rep_id)
        if grp.empty:
            continue

        nearest_km = min(
            haversine_km(
                float(customer_lat),
                float(customer_lon),
                float(r["customer_lat"]),
                float(r["customer_lon"]),
            )
            for _, r in grp.iterrows()
        )
        workload_count = int(len(grp))

        # Main rule: nearest representative route wins.
        # Tie-breaker: fewer currently assigned stops.
        candidate = (nearest_km, workload_count, rep_id_str)
        if best_distance is None or candidate < (
            best_distance,
            best_workload_count or 0,
            best_rep or "",
        ):
            best_distance = nearest_km
            best_workload_count = workload_count
            best_rep = rep_id_str

    return best_rep or "UNASSIGNED"


def assign_new_customer_by_priority_queue(
    assign_df: pd.DataFrame,
    rep_df: pd.DataFrame,
    customer_lat: float,
    customer_lon: float,
    alpha: float = 0.60,
    beta: float = 0.40,
) -> str:
    """
    Alternative added-customer assignment helper based on DEQ priority.

    Purpose:
    - Ranks representatives using priority score.
    - Can be used for a priority-aware assignment strategy.

    Notes:
    - The current baseline add-customer endpoint uses nearest-route
      assignment instead, so this helper is optional/experimental.
    """
    scored = compute_thesis_priority_scores(assign_df, rep_df, alpha=alpha, beta=beta)

    best_rep = None
    best_score = None

    for row in scored.itertuples(index=False):
        rep_id = str(row.rep_id)
        rep_points = assign_df[assign_df["rep_id"] == rep_id].copy()

        if rep_points.empty:
            return rep_id

        nearest_km = min(
            haversine_km(
                customer_lat,
                customer_lon,
                float(r["customer_lat"]),
                float(r["customer_lon"]),
            )
            for _, r in rep_points.iterrows()
        )

        # Lower PS first, distance as tie-breaker
        candidate_score = (float(row.priority_score), nearest_km)

        if best_score is None or candidate_score < best_score:
            best_score = candidate_score
            best_rep = rep_id

    return best_rep or str(scored.iloc[0]["rep_id"])

