import math
from typing import Any, Dict, List, Tuple

import pandas as pd

from backend.services.distance_service import haversine_km
from backend.services.metrics_service import (
    compute_thesis_priority_scores,
    jains_fairness,
    objective_value,
    workload_balance_index,
)
from backend.services.routing_node_service import ensure_preview_node_ids
from backend.services.routing_service import route_all

# ============================================================
# SECTION 10: Enhanced DEQ rebalancing logic
# Purpose:
# - Improves the baseline assignment by evaluating possible customer
#   movements or swaps between representatives.
# - Uses workload/fairness, distance, time, and priority score conditions
#   to decide whether a reassignment should be accepted.
# - Applies the Double-Ended Queue concept by prioritizing representatives
#   based on workload and priority score.
#
# note:
# - This is the enhanced part of the system. The baseline finds routes
#   using GNN, while the enhanced model attempts to rebalance assignments
#   without worsening operational performance.
# ============================================================

def border_candidates(
    assign_df: pd.DataFrame, heavy_rep: str, light_rep: str, fraction: float
) -> List[int]:
    """
    Selects candidate customers from a heavier representative for possible transfer.

    Purpose:
    - Finds customers near the boundary between a heavy representative and
      a lighter representative.
    - Limits the search using border_fraction so the enhanced algorithm does
      not evaluate every possible customer movement.

    Used by:
    - Enhanced DEQ reassignment search.
    """
    heavy = assign_df[assign_df["rep_id"] == heavy_rep].copy()
    light = assign_df[assign_df["rep_id"] == light_rep].copy()

    if heavy.empty or light.empty:
        return []

    target_lat = light["customer_lat"].mean()
    target_lon = light["customer_lon"].mean()

    heavy["to_target"] = heavy.apply(
        lambda r: haversine_km(
            r["customer_lat"], r["customer_lon"], target_lat, target_lon
        ),
        axis=1,
    )
    heavy = heavy.sort_values(["to_target", "predicted_eta_min"]).reset_index()
    take = max(1, int(math.ceil(len(heavy) * fraction)))
    return heavy.head(take)["index"].astype(int).tolist()


def swap_candidates(
    assign_df: pd.DataFrame,
    heavy_rep: str,
    light_rep: str,
    fraction: float,
) -> Tuple[List[int], List[int]]:
    """
    Selects candidate customers from two representatives for possible swapping.

    Purpose:
    - Finds near-border customers from both the heavy and light representative.
    - Supports one-for-one reassignment attempts when direct movement is not enough.

    Notes:
    - This keeps the enhanced search focused on practical route-border changes.
    """
    heavy = assign_df[assign_df["rep_id"] == heavy_rep].copy()
    light = assign_df[assign_df["rep_id"] == light_rep].copy()

    if heavy.empty or light.empty:
        return [], []

    heavy_target_lat = light["customer_lat"].mean()
    heavy_target_lon = light["customer_lon"].mean()

    light_target_lat = heavy["customer_lat"].mean()
    light_target_lon = heavy["customer_lon"].mean()

    heavy["to_target"] = heavy.apply(
        lambda r: haversine_km(
            r["customer_lat"], r["customer_lon"], heavy_target_lat, heavy_target_lon
        ),
        axis=1,
    )
    light["to_target"] = light.apply(
        lambda r: haversine_km(
            r["customer_lat"], r["customer_lon"], light_target_lat, light_target_lon
        ),
        axis=1,
    )

    heavy = heavy.sort_values(["to_target", "predicted_eta_min"]).reset_index()
    light = light.sort_values(["to_target", "predicted_eta_min"]).reset_index()

    heavy_take = max(1, int(math.ceil(len(heavy) * fraction)))
    light_take = max(1, int(math.ceil(len(light) * fraction)))

    heavy_idx = heavy.head(heavy_take)["index"].astype(int).tolist()
    light_idx = light.head(light_take)["index"].astype(int).tolist()

    return heavy_idx, light_idx


def evaluate_assignment(
    assign_df: pd.DataFrame,
    speed_kmph: float,
    service_min: float,
    distance_matrix: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """
    Evaluates a complete assignment by rerouting all representatives.

    Purpose:
    - Recomputes routes after a possible move or swap.
    - Measures total distance, travel time, operational time, fairness,
      and workload balance.

    Used by:
    - Enhanced DEQ acceptance checking.
    """
    routes, rep_df, total = route_all(
        assign_df, speed_kmph, service_min, "eval", distance_matrix
    )
    fairness = (
        jains_fairness(rep_df["workload_min"].tolist()) if not rep_df.empty else 1.0
    )
    wbi = (
        workload_balance_index(rep_df["workload_min"].tolist())
        if not rep_df.empty
        else 0.0
    )

    assigned_customers = int(rep_df["customers"].sum()) if not rep_df.empty else 0
    coverage_ratio = assigned_customers / max(1, len(assign_df))
    return {
        "routes": routes,
        "rep_df": rep_df,
        "total": total,
        "fairness": fairness,
        "wbi": wbi,
        "coverage_ratio": coverage_ratio,
    }


def enhance_assignment(
    assign_df: pd.DataFrame,
    speed_kmph: float,
    service_min: float,
    alpha_weight: float,
    beta_weight: float,
    fairness_weight: float,
    distance_weight: float,
    time_weight: float,
    max_iterations: int,
    border_fraction: float,
    distance_matrix: Dict[str, Dict[str, float]],
    is_zomato_mode: bool = False,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Performs enhanced DEQ-based assignment rebalancing.

    Purpose:
    - Starts from the baseline customer-to-representative assignment.
    - Identifies overloaded and underloaded representatives.
    - Tests candidate moves/swaps between representatives.
    - Accepts changes only when fairness improves and distance/time
      constraints are not worsened beyond tolerance.

    note:
    - This function implements the main enhancement over the baseline.
    - It connects the DEQ concept with operational constraints such as
      workload balance, travel distance, and service time.
    """
    current = ensure_preview_node_ids(assign_df.copy())
    logs: List[Dict[str, Any]] = []
    current_eval = evaluate_assignment(
        current, speed_kmph, service_min, distance_matrix
    )

    for iteration in range(1, max_iterations + 1):
        rep_perf = compute_thesis_priority_scores(
            current,
            current_eval["rep_df"],
            alpha=alpha_weight,
            beta=beta_weight,
        ).reset_index(drop=True)

        if len(rep_perf) < 2:
            break

        overall_gap = float(
            rep_perf.iloc[-1]["operational_minutes"]
            - rep_perf.iloc[0]["operational_minutes"]
        )
        if overall_gap < 5.0:
            break

        current_score = objective_value(
            current_eval["wbi"],
            current_eval["total"]["distance_km"],
            current_eval["total"]["operational_minutes"],
            fairness_weight,
            distance_weight,
            time_weight,
        )

        n_reps = len(rep_perf)
        heavy_count = min(3, n_reps - 1)
        light_count = min(3, n_reps - 1)

        light_ids = [
            str(rep_perf.iloc[i]["rep_id"]) for i in range(min(light_count, n_reps))
        ]
        heavy_ids = [
            str(rep_perf.iloc[n_reps - 1 - j]["rep_id"])
            for j in range(min(heavy_count, n_reps))
        ]

        # ---------------------------
        # 1) TRANSFER SEARCH
        # ---------------------------
        best_trial = None
        best_trial_eval = None
        best_log = None
        best_score_gain = 0.0

        tried_pairs = set()

        for heavy_rep in heavy_ids:
            for light_rep in light_ids:
                if heavy_rep == light_rep:
                    continue
                if (heavy_rep, light_rep) in tried_pairs:
                    continue
                tried_pairs.add((heavy_rep, light_rep))

                heavy_minutes = float(
                    rep_perf.loc[
                        rep_perf["rep_id"] == heavy_rep, "operational_minutes"
                    ].iloc[0]
                )
                light_minutes = float(
                    rep_perf.loc[
                        rep_perf["rep_id"] == light_rep, "operational_minutes"
                    ].iloc[0]
                )

                workload_gap = heavy_minutes - light_minutes
                if workload_gap < 5.0:
                    continue

                candidates = border_candidates(
                    current, heavy_rep, light_rep, border_fraction
                )

                for idx in candidates:
                    # do not allow a move that would empty the source rep
                    current_source_count = int((current["rep_id"] == heavy_rep).sum())
                    if current_source_count <= 1:
                        continue

                    trial = current.copy()
                    trial.loc[idx, "rep_id"] = light_rep

                    source_rep_remaining = int((trial["rep_id"] == heavy_rep).sum())
                    if source_rep_remaining == 0:
                        continue

                    trial_eval = evaluate_assignment(
                        trial, speed_kmph, service_min, distance_matrix
                    )

                    trial_score = objective_value(
                        trial_eval["wbi"],
                        trial_eval["total"]["distance_km"],
                        trial_eval["total"]["operational_minutes"],
                        fairness_weight,
                        distance_weight,
                        time_weight,
                    )

                    fairness_gain = trial_eval["fairness"] - current_eval["fairness"]
                    distance_gain = (
                        current_eval["total"]["distance_km"]
                        - trial_eval["total"]["distance_km"]
                    )
                    time_gain = (
                        current_eval["total"]["operational_minutes"]
                        - trial_eval["total"]["operational_minutes"]
                    )
                    score_gain = current_score - trial_score

                    EPS = 1e-6

                    # Acceptance rule:
                    # - Keep only changes that improve workload/fairness.
                    # - Reject changes that worsen distance or operational time beyond EPS.
                    # - This prevents the enhanced algorithm from improving fairness at the
                    #   expense of unreasonable route cost.

                    if fairness_gain <= 0:
                        continue

                    if is_zomato_mode:
                        # Slightly looser for Zomato:
                        # allow small distance/time worsening if fairness gain is meaningful
                        if distance_gain < -12.0:
                            continue
                        if distance_gain < -4.0:
                            continue
                        if fairness_gain < 0.01 and (
                            distance_gain < -EPS or time_gain < -EPS
                        ):
                            continue
                    else:
                        # Keep Amazon strict
                        if distance_gain < -EPS:
                            continue
                        if time_gain < -EPS:
                            continue

                    if score_gain > best_score_gain:
                        best_score_gain = score_gain
                        best_trial = trial
                        best_trial_eval = trial_eval
                        best_log = {
                            "iteration": iteration,
                            "move_type": "transfer",
                            "moved_order": str(current.loc[idx, "order_id"]),
                            "from_rep": heavy_rep,
                            "to_rep": light_rep,
                            "fairness_before": round(current_eval["fairness"], 6),
                            "fairness_after": round(trial_eval["fairness"], 6),
                            "distance_before": round(
                                current_eval["total"]["distance_km"], 2
                            ),
                            "distance_after": round(
                                trial_eval["total"]["distance_km"], 2
                            ),
                            "operational_before": round(
                                current_eval["total"]["operational_minutes"], 2
                            ),
                            "operational_after": round(
                                trial_eval["total"]["operational_minutes"], 2
                            ),
                            "score_before": round(current_score, 4),
                            "score_after": round(trial_score, 4),
                            "score_gain": round(score_gain, 4),
                            "fairness_gain": round(fairness_gain, 6),
                            "distance_gain": round(distance_gain, 4),
                            "time_gain": round(time_gain, 4),
                            "accepted": score_gain > 0,
                        }

        if best_trial is not None and best_score_gain > 0:
            current = best_trial
            current_eval = best_trial_eval
            logs.append(best_log)
            continue

        # ---------------------------
        # 2) SWAP SEARCH
        # ---------------------------
        swap_best_trial = None
        swap_best_trial_eval = None
        swap_best_log = None
        swap_best_score_gain = 0.0

        tried_pairs = set()

        for heavy_rep in heavy_ids:
            for light_rep in light_ids:
                if heavy_rep == light_rep:
                    continue
                if (heavy_rep, light_rep) in tried_pairs:
                    continue
                tried_pairs.add((heavy_rep, light_rep))

                heavy_minutes = float(
                    rep_perf.loc[
                        rep_perf["rep_id"] == heavy_rep, "operational_minutes"
                    ].iloc[0]
                )
                light_minutes = float(
                    rep_perf.loc[
                        rep_perf["rep_id"] == light_rep, "operational_minutes"
                    ].iloc[0]
                )

                workload_gap = heavy_minutes - light_minutes
                if workload_gap < 5.0:
                    continue

                heavy_idx_list, light_idx_list = swap_candidates(
                    current,
                    heavy_rep,
                    light_rep,
                    border_fraction,
                )

                for idx_h in heavy_idx_list:
                    for idx_l in light_idx_list:
                        if idx_h == idx_l:
                            continue

                        trial = current.copy()

                        rep_h = str(trial.loc[idx_h, "rep_id"])
                        rep_l = str(trial.loc[idx_l, "rep_id"])

                        if rep_h == rep_l:
                            continue

                        trial.loc[idx_h, "rep_id"] = rep_l
                        trial.loc[idx_l, "rep_id"] = rep_h

                        trial_eval = evaluate_assignment(
                            trial,
                            speed_kmph,
                            service_min,
                            distance_matrix,
                        )

                        trial_score = objective_value(
                            trial_eval["wbi"],
                            trial_eval["total"]["distance_km"],
                            trial_eval["total"]["operational_minutes"],
                            alpha_weight,
                            beta_weight,
                            time_weight,
                        )

                        fairness_gain = (
                            trial_eval["fairness"] - current_eval["fairness"]
                        )
                        distance_gain = (
                            current_eval["total"]["distance_km"]
                            - trial_eval["total"]["distance_km"]
                        )
                        time_gain = (
                            current_eval["total"]["operational_minutes"]
                            - trial_eval["total"]["operational_minutes"]
                        )
                        score_gain = current_score - trial_score

                    EPS = 1e-6

                    if fairness_gain <= 0:
                        continue

                    if is_zomato_mode:
                        # Slightly looser for Zomato:
                        # allow small distance/time worsening if fairness gain is meaningful
                        if distance_gain < -12.0:
                            continue
                        if distance_gain < -4.0:
                            continue
                        if fairness_gain < 0.01 and (
                            distance_gain < -EPS or time_gain < -EPS
                        ):
                            continue
                    else:
                        # Keep Amazon strict
                        if distance_gain < -EPS:
                            continue
                        if time_gain < -EPS:
                            continue

                        if score_gain > swap_best_score_gain:
                            swap_best_score_gain = score_gain
                            swap_best_trial = trial
                            swap_best_trial_eval = trial_eval
                            swap_best_log = {
                                "iteration": iteration,
                                "move_type": "swap",
                                "moved_order": str(current.loc[idx_h, "order_id"]),
                                "swapped_with_order": str(
                                    current.loc[idx_l, "order_id"]
                                ),
                                "from_rep": heavy_rep,
                                "to_rep": light_rep,
                                "fairness_before": round(current_eval["fairness"], 6),
                                "fairness_after": round(trial_eval["fairness"], 6),
                                "distance_before": round(
                                    current_eval["total"]["distance_km"], 2
                                ),
                                "distance_after": round(
                                    trial_eval["total"]["distance_km"], 2
                                ),
                                "operational_before": round(
                                    current_eval["total"]["operational_minutes"], 2
                                ),
                                "operational_after": round(
                                    trial_eval["total"]["operational_minutes"], 2
                                ),
                                "score_before": round(current_score, 4),
                                "score_after": round(trial_score, 4),
                                "score_gain": round(score_gain, 4),
                                "fairness_gain": round(fairness_gain, 6),
                                "distance_gain": round(distance_gain, 4),
                                "time_gain": round(time_gain, 4),
                                "accepted": score_gain > 0,
                            }

        if swap_best_trial is not None and swap_best_score_gain > 0:
            current = swap_best_trial
            current_eval = swap_best_trial_eval
            logs.append(swap_best_log)
            continue

        # ---------------------------
        # 3) STOP
        # ---------------------------
        logs.append(
            {
                "iteration": iteration,
                "from_rep": heavy_ids[0] if heavy_ids else "",
                "to_rep": light_ids[0] if light_ids else "",
                "accepted": False,
                "reason": "no improving transfer or swap found",
            }
        )
        break

    print(
        "rep priority/workload snapshot:",
        rep_perf[
            ["rep_id", "priority_score", "operational_minutes", "customers"]
        ].to_dict("records"),
    )
    print("overall_gap minutes:", round(overall_gap, 2))

    print("accepted moves:", sum(1 for x in logs if x.get("accepted")))
    print("enhancement logs:", logs)

    return current, logs

