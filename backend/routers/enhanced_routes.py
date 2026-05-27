from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Response

from schemas import BaselineRequest, EnhancedRequest
from state import DATASETS, RUNS
from services.amazon_service import amazon_distance_polish_assignment
from services.dataset_service import get_run_profile
from services.distance_service import attach_route_display_geometry
from services.enhanced_service import enhance_assignment
from services.metrics_service import make_algorithm_run
from services.routing_service import preview_summary_from_assign_df, route_all

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

@router.post("/api/runs/enhanced")

def run_enhanced(req: EnhancedRequest) -> Dict[str, Any]:
    """
    Executes the enhanced DEQ rebalancing algorithm.

    Purpose:
    - Starts from a previous baseline run.
    - Applies DEQ-based rebalancing using priority score and workload.
    - Recalculates routes and KPIs after accepted changes.
    - Returns the enhanced route result for comparison.

    note:
    - This endpoint represents the enhanced algorithm evaluated against
      the baseline output.
    """
    print("enhanced started")
    dataset_payload = DATASETS.get(req.dataset_id)
    baseline_payload = RUNS.get(req.baseline_run_id)

    if not dataset_payload:
        return Response(
            content="Dataset not found", status_code=404, media_type="text/plain"
        )
    if not baseline_payload:
        raise HTTPException(status_code=404, detail="Baseline run not found.")

    role = dataset_payload["datasetRole"]
    baseline_profile = baseline_payload.get("profile", get_run_profile(None))
    profile = get_run_profile(req.run_profile or baseline_profile.get("profile_name"))

    print("enhanced profile:", profile["profile_name"])
    print("enhanced params:", req.model_dump())

    distance_matrix = baseline_payload.get("distance_matrix", {})

    # task 6: alpha/beta for priority scoring
    alpha = float(req.alpha_weight if req.alpha_weight is not None else 0.60)
    beta = float(req.beta_weight if req.beta_weight is not None else 0.40)

    if alpha < 0:
        alpha = 0.0
    if beta < 0:
        beta = 0.0

    weight_sum = alpha + beta
    if weight_sum <= 0:
        alpha, beta = 0.60, 0.40
    else:
        alpha = alpha / weight_sum
        beta = beta / weight_sum

    # keep current optimization weights for now
    effective_fairness_weight = profile["enhanced_fairness_weight"]
    effective_distance_weight = profile["enhanced_distance_weight"]
    effective_time_weight = profile["enhanced_time_weight"]

    effective_max_iterations = (
        req.max_iterations
        if req.max_iterations is not None
        else profile["enhanced_max_iterations"]
    )
    effective_border_fraction = (
        req.border_fraction
        if req.border_fraction is not None
        else profile["enhanced_border_fraction"]
    )

    print("normalized alpha/beta:", {"alpha": alpha, "beta": beta})

    print("enhanced dataset and baseline found")
    baseline_req = BaselineRequest(**baseline_payload["request"])
    assign_df = baseline_payload["assign_df"].copy()
    print(f"enhanced assign_df rows: {len(assign_df)}")

    is_zomato_mode = role == "comparative_template"

    improved_df, logs = enhance_assignment(
        assign_df,
        baseline_req.avg_speed_kmph,
        baseline_req.service_minutes_per_stop,
        alpha,
        beta,
        effective_fairness_weight,
        effective_distance_weight,
        effective_time_weight,
        effective_max_iterations,
        effective_border_fraction,
        distance_matrix,
        is_zomato_mode=is_zomato_mode,
    )

    if role == "primary_reconstruction":
        improved_df, amazon_polish_logs = amazon_distance_polish_assignment(
            improved_df,
            baseline_req.avg_speed_kmph,
            baseline_req.service_minutes_per_stop,
            distance_matrix,
            max_iterations=12,
        )
        logs.extend(amazon_polish_logs)

    print("enhance_assignment done")

    routes, rep_df, total = route_all(
        improved_df,
        baseline_req.avg_speed_kmph,
        baseline_req.service_minutes_per_stop,
        "enhanced",
        distance_matrix,
    )
    print("enhanced route_all done")

    routes = attach_route_display_geometry(routes, improved_df)
    print("enhanced display geometry attached")

    training_metrics = baseline_payload["run"].get("trainingComparison", {})
    role = dataset_payload["datasetRole"]
    role_note = (
        "Enhanced DEQ run over Amazon-derived reconstructed baseline"
        if role == "primary_reconstruction"
        else (
            "Enhanced DEQ run for comparative/template Zomato evaluation"
            if role == "comparative_template"
            else "Enhanced DEQ run over uploaded dataset"
        )
    )

    run = make_algorithm_run(
        "Enhanced G-NN + DEQ Rebalancing",
        routes,
        rep_df,
        total,
        len(improved_df),
        training_metrics.get("enhanced", baseline_payload["run"]["trainingMetrics"]),
        notes=[
            role_note,
            "Baseline-seeded DEQ rebalancing",
            "Priority scoring uses alpha/beta for time difference and rating",
            "Joint acceptance on workload balance, distance, and operational time",
            f"Accepted rebalances: {sum(1 for x in logs if x.get('accepted'))}",
        ],
        assign_df=improved_df,
    )

    run["datasetId"] = req.dataset_id
    run["baselineRunId"] = req.baseline_run_id
    run["runType"] = "enhanced"
    run["datasetRole"] = role
    run["sourceLabel"] = dataset_payload["sourceLabel"]
    run["runLog"] = logs
    run["runProfile"] = profile["profile_name"]
    run["profileConfig"] = profile
    run["previewSummary"] = preview_summary_from_assign_df(improved_df)
    run["previewMode"] = True

    RUNS[run["id"]] = {
        "assign_df": improved_df,
        "distance_matrix": distance_matrix,
        "request": baseline_req.model_dump(),
        "run": run,
        "profile": profile,
    }
    return run
