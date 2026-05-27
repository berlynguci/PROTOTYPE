from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, HTTPException, Response

from config import (
    AMAZON_DEFAULT_REPRESENTATIVES,
    AMAZON_FIXED_DEMO_AGENTS,
    AMAZON_FIXED_DEMO_NODES,
    AMAZON_MIN_PREVIEW_STOPS,
    DEMO_PREVIEW_DEPOTS,
    MIN_FIXED_DEMO_AGENTS,
    MIN_FIXED_DEMO_NODES,
)
from schemas import AddedCustomerPayload, BaselineAddCustomersRequest, BaselineRequest
from state import DATASETS, RUNS
from services.added_customer_service import (
    append_added_customers_to_assign_df,
    assign_new_customer_to_nearest_rep,
)
from services.amazon_service import (
    build_amazon_order_routing_rows,
    build_local_preview_subset_amazon,
)
from services.dataset_service import build_routing_nodes, get_run_profile
from services.distance_service import (
    attach_route_display_geometry,
    build_preview_distance_matrix,
    haversine_km,
    preview_matrix_stats,
)
from services.eta_service import train_eta_models
from services.metrics_service import make_algorithm_run
from services.routing_node_service import ensure_preview_node_ids
from services.routing_service import (
    build_local_preview_subset,
    filter_df_to_demo_depot,
    preview_summary_from_assign_df,
    route_all,
)

# ============================================================
# SECTION 12: API endpoints
# Purpose:
# - Exposes backend functions to the React frontend.
# - Supports dataset validation, dataset metadata, baseline routing,
#   add-customer rerouting, enhanced DEQ routing, and export/download.
#
# note:
# - The frontend does not directly run routing algorithms. It sends API
#   requests to these endpoints, and the backend returns route/KPI payloads.
# ============================================================

router = APIRouter()

@router.post("/api/runs/baseline")
@router.post("/api/runs/baseline/add-customers")


def run_baseline(req: BaselineRequest) -> Dict[str, Any]:
    """
    Executes the baseline routing algorithm.

    Purpose:
    - Loads the validated dataset.
    - Trains ETA prediction models.
    - Builds routing rows and preview subsets.
    - Creates the distance matrix.
    - Runs GNN-based route construction.
    - Returns routes, representative summaries, KPIs, and map data.

    note:
    - This endpoint represents the baseline GNN + Dijkstra workflow.
    """
    print("run_baseline started")

    payload = DATASETS.get(req.dataset_id)
    if not payload:
        return Response(
            content="Dataset not found", status_code=404, media_type="text/plain"
        )

    if req.num_representatives < 4 or req.num_representatives > 15:
        raise HTTPException(
            status_code=400,
            detail="Number of representatives must be between 4 and 15.",
        )

    profile = get_run_profile(req.run_profile)
    print("baseline profile:", profile["profile_name"])

    print("dataset found")
    df = payload["data"].copy()
    print(f"data copied: {len(df)} rows")

    predicted_eta, metrics = train_eta_models(df, req.seed)
    print("train_eta_models done")
    df["predicted_eta_min"] = predicted_eta

    role = payload["datasetRole"]
    effective_num_representatives = (
        max(AMAZON_DEFAULT_REPRESENTATIVES, req.num_representatives)
        if role == "primary_reconstruction"
        else req.num_representatives
    )

    if role == "primary_reconstruction":
        routing_df = build_amazon_order_routing_rows(df)
        print(f"amazon order-level routing_df built: {len(routing_df)} order rows")
    else:
        routing_df = build_routing_nodes(df)
        print(f"routing_df built: {len(routing_df)} node rows")

    # Keep Zomato using the original fixed-depot strength check.
    # Amazon keeps the no-minimum fixed-depot behavior without changing its routing logic.
    depot_min_nodes = (
        AMAZON_FIXED_DEMO_NODES
        if role == "primary_reconstruction"
        else MIN_FIXED_DEMO_NODES
    )
    depot_min_agents = (
        AMAZON_FIXED_DEMO_AGENTS
        if role == "primary_reconstruction"
        else MIN_FIXED_DEMO_AGENTS
    )

    routing_df = filter_df_to_demo_depot(
        routing_df,
        payload["datasetRole"],
        min_nodes=depot_min_nodes,
        min_agents=depot_min_agents,
    )
    print(f"routing_df after demo depot filter: {len(routing_df)} rows")
    role_note = (
        "Primary Amazon-based reconstructed baseline workflow"
        if role == "primary_reconstruction"
        else (
            "Comparative/template workflow using Zomato-aligned structure"
            if role == "comparative_template"
            else "Generic uploaded dataset workflow"
        )
    )

    preview_max_total_stops = (
        max(profile["preview_max_total_stops"], 40)
        if role == "comparative_template"
        else (
            max(profile["preview_max_total_stops"], AMAZON_MIN_PREVIEW_STOPS)
            if role == "primary_reconstruction"
            else profile["preview_max_total_stops"]
        )
    )

    if role == "primary_reconstruction":
        preview_df = build_local_preview_subset_amazon(
            routing_df,
            num_representatives=effective_num_representatives,
            max_total_stops=preview_max_total_stops,
            initial_radius_km=profile["preview_initial_radius_km"],
            max_radius_km=profile["preview_max_radius_km"],
            local_cap_km=profile["preview_local_cap_km"],
            use_existing_agents=True,
            strict_existing_agents=True,
            min_nodes_per_rep=1,
        )
    else:
        preview_df = build_local_preview_subset(
            routing_df,
            num_representatives=effective_num_representatives,
            max_total_stops=preview_max_total_stops,
            initial_radius_km=profile["preview_initial_radius_km"],
            max_radius_km=profile["preview_max_radius_km"],
            local_cap_km=profile["preview_local_cap_km"],
            use_existing_agents=(
                role in {"primary_reconstruction", "comparative_template"}
            ),
            strict_existing_agents=(
                role in {"primary_reconstruction", "comparative_template"}
            ),
        )
    print(f"preview_df built: {len(preview_df)} rows")

    preview_df = ensure_preview_node_ids(preview_df)
    depot_lat = float(preview_df.iloc[0]["depot_lat"])
    depot_lon = float(preview_df.iloc[0]["depot_lon"])

    preview_df["debug_to_depot_km"] = preview_df.apply(
        lambda r: haversine_km(
            depot_lat,
            depot_lon,
            float(r["customer_lat"]),
            float(r["customer_lon"]),
        ),
        axis=1,
    )

    print("preview stop distances from depot (km):")
    print(
        preview_df[["customer_node_id", "customer_name", "debug_to_depot_km"]]
        .sort_values("debug_to_depot_km", ascending=False)
        .head(12)
    )
    print(
        "max preview distance from depot:", float(preview_df["debug_to_depot_km"].max())
    )
    preview_matrix = build_preview_distance_matrix(
        preview_df,
        osm_threshold_km=profile["preview_osm_threshold_km"],
    )
    print("preview_matrix built")
    matrix_stats = preview_matrix_stats(preview_df)

    preview_routes, preview_rep_df, preview_total = route_all(
        preview_df,
        req.avg_speed_kmph,
        req.service_minutes_per_stop,
        "baseline",
        preview_matrix,
    )
    print("preview route_all done")

    preview_routes = attach_route_display_geometry(preview_routes, preview_df)
    print("baseline display geometry attached")

    preview_run = make_algorithm_run(
        "Baseline G-NN + Dijkstra",
        preview_routes,
        preview_rep_df,
        preview_total,
        len(preview_df),
        metrics["baseline"],
        notes=[
            role_note,
            "Preview mode for UI rendering",
            "Preview restricted to nearest customers within the fixed demo depot",
            "Uses existing agent_id where available; Zomato strongly preserves real agents before any fallback",
            f"Preview target: {preview_max_total_stops} stops; Amazon order-level preview is not capped to 12 customer nodes or 3 customers per rep",
            f"Fixed demo depot override: {DEMO_PREVIEW_DEPOTS.get(role) or 'automatic'}",
        ],
        assign_df=preview_df,
    )

    preview_run["datasetId"] = req.dataset_id
    preview_run["runType"] = "baseline"
    preview_run["datasetRole"] = role
    preview_run["sourceLabel"] = payload["sourceLabel"]
    preview_run["trainingComparison"] = metrics
    preview_run["previewMode"] = True
    preview_run["previewSummary"] = preview_summary_from_assign_df(preview_df)
    preview_run["matrixMode"] = "osm_or_proxy_preview_matrix"
    preview_run["matrixStats"] = matrix_stats
    preview_run["runProfile"] = profile["profile_name"]
    preview_run["profileConfig"] = profile

    RUNS[preview_run["id"]] = {
        "assign_df": preview_df,
        "distance_matrix": preview_matrix,
        "request": req.model_dump(),
        "run": preview_run,
        "profile": profile,
    }

    print("run_baseline finished")
    return preview_run

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

def add_customers_to_baseline(req: BaselineAddCustomersRequest) -> Dict[str, Any]:
    """
    Updates an existing baseline run with newly added customers.

    Purpose:
    - Retrieves the previous baseline assignment.
    - Assigns each new customer to the nearest representative route.
    - Appends the new customer rows into the routing dataset.
    - Rebuilds the preview distance matrix.
    - Reruns baseline routing and returns the updated run.

    Used by:
    - Add Customer modal in the React frontend.
    """
    baseline_payload = RUNS.get(req.baseline_run_id)
    if not baseline_payload:
        raise HTTPException(status_code=404, detail="Baseline run not found.")

    if not req.customers:
        raise HTTPException(status_code=400, detail="No customers supplied.")

    assign_df = baseline_payload["assign_df"].copy()
    distance_matrix = baseline_payload["distance_matrix"]
    baseline_req = BaselineRequest(**baseline_payload["request"])
    base_run = baseline_payload["run"]
    profile = baseline_payload.get("profile", get_run_profile(None))

    current_routes, current_rep_df, current_total = route_all(
        assign_df,
        baseline_req.avg_speed_kmph,
        baseline_req.service_minutes_per_stop,
        "baseline",
        distance_matrix,
    )

    resolved_customers: List[AddedCustomerPayload] = []
    updated_assign_df = assign_df.copy()

    # Assign added customers sequentially using the nearest existing route.
    # After each customer is assigned, append it immediately so the next added
    # customer sees the updated route state.
    for customer in req.customers:
        assigned_rep = assign_new_customer_to_nearest_rep(
            updated_assign_df,
            float(customer.lat),
            float(customer.lon),
        )

        resolved_customer = AddedCustomerPayload(
            label=customer.label,
            lat=customer.lat,
            lon=customer.lon,
            address=customer.address,
            assigned_rep=assigned_rep,
            customer_number=customer.customer_number,
        )
        resolved_customers.append(resolved_customer)
        updated_assign_df = append_added_customers_to_assign_df(
            updated_assign_df, [resolved_customer]
        )

    updated_assign_df = ensure_preview_node_ids(updated_assign_df)

    updated_matrix = build_preview_distance_matrix(
        updated_assign_df,
        osm_threshold_km=profile["preview_osm_threshold_km"],
    )
    updated_matrix_stats = preview_matrix_stats(updated_assign_df)

    routes, rep_df, total = route_all(
        updated_assign_df,
        baseline_req.avg_speed_kmph,
        baseline_req.service_minutes_per_stop,
        "baseline",
        updated_matrix,
    )

    routes = attach_route_display_geometry(routes, updated_assign_df)

    updated_run = make_algorithm_run(
        "Baseline G-NN + Dijkstra",
        routes,
        rep_df,
        total,
        len(updated_assign_df),
        base_run.get("trainingMetrics", {}),
        notes=(base_run.get("notes", []) + [f"Added customers: {len(req.customers)}"]),
        assign_df=updated_assign_df,
    )

    updated_run["datasetId"] = base_run["datasetId"]
    updated_run["runType"] = "baseline"
    updated_run["datasetRole"] = base_run.get("datasetRole")
    updated_run["sourceLabel"] = base_run.get("sourceLabel")
    updated_run["trainingComparison"] = base_run.get("trainingComparison")
    updated_run["previewMode"] = True
    updated_run["previewSummary"] = preview_summary_from_assign_df(updated_assign_df)
    updated_run["matrixMode"] = "osm_or_proxy_preview_matrix"
    updated_run["matrixStats"] = updated_matrix_stats
    updated_run["runProfile"] = base_run.get("runProfile")
    updated_run["profileConfig"] = base_run.get("profileConfig")
    updated_run["addedCustomers"] = [
        {
            "label": c.label,
            "lat": c.lat,
            "lon": c.lon,
            "address": c.address,
            "assignedRep": c.assigned_rep,
            "customerNumber": c.customer_number,
        }
        for c in resolved_customers
    ]

    RUNS[updated_run["id"]] = {
        "assign_df": updated_assign_df,
        "distance_matrix": updated_matrix,
        "request": baseline_req.model_dump(),
        "run": updated_run,
        "profile": profile,
    }

    return updated_run
