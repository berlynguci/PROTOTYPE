import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================
# SECTION 9: Priority scoring, fairness, workload, and KPI metrics
# Purpose:
# - Computes DEQ priority scores.
# - Measures workload balance and fairness.
# - Builds KPI summaries used in the frontend comparison page.
#
# note:
# - These metrics connect the implementation to the thesis evaluation:
#   distance, travel time, operational time, WBI, JFI, and coverage ratio.
# ============================================================

def compute_thesis_priority_scores(
    assign_df: pd.DataFrame,
    rep_df: pd.DataFrame,
    alpha: float = 0.60,
    beta: float = 0.40,
) -> pd.DataFrame:
    """
    Computes the DEQ priority score for each representative.

    Formula:
    - PS = alpha * ΔT + beta * (1 - Rating)

    Purpose:
    - Ranks representatives based on delay/workload and rating.
    - Lower priority score means the representative is prioritized earlier
      in the queue.

    Used by:
    - Enhanced DEQ rebalancing.
    - Representative card/queue display.
    """
    if rep_df.empty:
        return rep_df.copy()

    work = assign_df.copy()
    reps = rep_df.copy()

    # Representative rating = mean assigned-customer rating
    rep_rating = (
        work.groupby("rep_id")["rating"]
        .mean()
        .reset_index()
        .rename(columns={"rating": "avg_rating"})
    )

    reps = reps.merge(rep_rating, on="rep_id", how="left")
    reps["avg_rating"] = pd.to_numeric(reps["avg_rating"], errors="coerce").fillna(1.0)

    # Delta T from operational minutes, normalized to [0,1]
    op = pd.to_numeric(reps["operational_minutes"], errors="coerce").fillna(0.0)
    op_min = float(op.min())
    op_max = float(op.max())

    if op_max > op_min:
        reps["delta_t"] = (op - op_min) / (op_max - op_min)
    else:
        reps["delta_t"] = 0.0

    # Ratings assumed already on 0..1 scale in your thesis examples
    # If your actual ratings are 1..5, normalize first:
    if reps["avg_rating"].max() > 1.0:
        reps["avg_rating_norm"] = reps["avg_rating"] / 5.0
    else:
        reps["avg_rating_norm"] = reps["avg_rating"]

    reps["priority_score"] = alpha * reps["delta_t"] + beta * (
        1.0 - reps["avg_rating_norm"]
    )

    reps = reps.sort_values("priority_score", ascending=True).reset_index(drop=True)
    reps["queue_position"] = np.arange(1, len(reps) + 1)

    return reps


def jains_fairness(values: List[float]) -> float:
    """
    Computes Jain's Fairness Index.

    Purpose:
    - Measures how evenly workload is distributed across representatives.
    - Higher values indicate better fairness.

    note:
    - The thesis target is typically close to 1.0, such as JFI ≥ 0.95.
    """
    arr = np.array(values, dtype=float)
    if len(arr) == 0 or np.allclose(arr.sum(), 0):
        return 1.0
    return float((arr.sum() ** 2) / (len(arr) * np.square(arr).sum()))


def workload_balance_index(values: List[float]) -> float:
    """
    Computes Workload Balance Index.

    Formula:
    - WBI = standard deviation of workload / mean workload

    Purpose:
    - Measures workload imbalance across representatives.
    - Lower values indicate better balance.
    """
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return 0.0
    mean = arr.mean()
    if mean == 0:
        return 0.0
    return float(arr.std(ddof=0) / mean)


def rep_cards(
    rep_df: pd.DataFrame, assign_df: Optional[pd.DataFrame] = None
) -> List[Dict[str, Any]]:
    """
    Builds representative summary cards for the frontend.

    Purpose:
    - Converts per-representative route metrics into UI-ready objects.
    - Includes workload, priority score, queue position, assigned customers,
      distance, and total time.
    """
    if rep_df.empty:
        return []

    scored = rep_df.copy()

    if assign_df is not None and not assign_df.empty:
        scored = compute_thesis_priority_scores(
            assign_df, scored, alpha=0.60, beta=0.40
        )
    else:
        scored["priority_score"] = 0.0
        scored["queue_position"] = np.arange(1, len(scored) + 1)
        scored["avg_rating_norm"] = 1.0

    max_workload = max(float(scored["workload_min"].max()), 1.0)

    out = []
    for _, row in scored.iterrows():
        out.append(
            {
                "id": row["rep_id"],
                "name": row["rep_id"],
                "workload": round(float(row["workload_min"]), 2),
                "opportunityScore": round(
                    max(
                        0.0, 100.0 - (float(row["workload_min"]) / max_workload) * 100.0
                    ),
                    1,
                ),
                "priorityScore": round(float(row["priority_score"]), 3),
                "queuePosition": int(row["queue_position"]),
                "assignedCustomers": int(row["customers"]),
                "totalDistance": round(float(row["distance_km"]), 2),
                "totalTime": round(float(row["operational_minutes"]), 2),
            }
        )
    return out


def kpis_from_totals(
    total: Dict[str, float], rep_df: pd.DataFrame, dataset_size: int
) -> Dict[str, Any]:
    """
    Builds KPI values for baseline/enhanced comparison.

    Purpose:
    - Computes total distance, travel time, operational time, fairness,
      workload balance, coverage, and averages.
    - Returns the metric structure expected by the React frontend.
    """
    fairness = jains_fairness(rep_df["workload_min"].tolist())
    wbi = workload_balance_index(rep_df["workload_min"].tolist())

    n_routes = max(1, len(rep_df))

    total_distance_km = float(total["distance_km"])
    total_travel_hr = float(total["travel_minutes"]) / 60.0
    total_operational_hr = float(total["operational_minutes"]) / 60.0

    avg_total_distance = total_distance_km / n_routes
    avg_travel_time = total_travel_hr / n_routes

    assigned_customers = int(rep_df["customers"].sum()) if not rep_df.empty else 0
    coverage_ratio = (assigned_customers / max(1, dataset_size)) * 100.0

    return {
        "totalDistance": round(total_distance_km, 2),
        "travelTime": round(total_travel_hr, 2),
        "operationalTime": round(total_operational_hr, 2),
        "computeTime": round(max(0.5, dataset_size / 80.0), 2),
        "fairness": round(fairness, 6),
        "workloadBalance": round(wbi * 100.0, 2),
        "coverage": round(coverage_ratio, 2),
        "scalability": round(dataset_size / n_routes, 2),
        # new compare-specific fields
        "avgTotalDistance": round(avg_total_distance, 2),
        "avgTravelTime": round(avg_travel_time, 2),
        "coverageRatio": round(coverage_ratio, 2),
        # compatibility fields
        "totalTime": round(total_operational_hr, 2),
        "numberOfStops": assigned_customers,
        "delayScore": 0.0,
        "ratingPenalty": 0.0,
        "workloadBalanceIndex": round(wbi * 100.0, 2),
        "jainsFairnessIndex": round(fairness, 6),
    }


def make_algorithm_run(
    name: str,
    routes: List[Dict[str, Any]],
    rep_df: pd.DataFrame,
    total: Dict[str, float],
    dataset_size: int,
    metrics: Dict[str, float],
    notes: Optional[List[str]] = None,
    assign_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Builds the final algorithm run payload returned to the frontend.

    Purpose:
    - Combines routes, representative cards, KPIs, training metrics,
      and run notes into one API response.
    """
    return {
        "id": str(uuid.uuid4()),
        "name": name,
        "algorithm": name,
        "routes": routes,
        "representatives": rep_cards(rep_df, assign_df),
        "kpis": kpis_from_totals(total, rep_df, dataset_size),
        "trainingMetrics": metrics,
        "notes": notes or [],
    }

def objective_value(
    wbi: float,
    total_distance_km: float,
    operational_minutes: float,
    fairness_weight: float,
    distance_weight: float,
    time_weight: float,
) -> float:
    # Lower is better
    return (
        fairness_weight * (wbi * 100.0)
        + distance_weight * total_distance_km
        + time_weight * (operational_minutes / 10.0)
    )
