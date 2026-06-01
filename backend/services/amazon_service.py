from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import HTTPException

from backend.config import AMAZON_DEFAULT_REPRESENTATIVES, AMAZON_MIN_PREVIEW_STOPS
from backend.schemas import FieldMapping
from backend.services.distance_service import haversine_km
from backend.services.base_reconstruction_service import _base_reconstruct_from_mapping
from backend.services.routing_service import (
    assign_preview_rep_ids_uneven,
    choose_best_local_depot_cluster,
)
from backend.services.enhanced_service import evaluate_assignment
from backend.services.routing_node_service import ensure_preview_node_ids

def reconstruct_raw_amazon_dataset(
    df: pd.DataFrame, mapping: FieldMapping
) -> pd.DataFrame:
    """
    Reconstructs a raw Amazon delivery dataset into the route-ready schema.

    Purpose:
    - Creates synthetic agent IDs from depot and Agent_Age.
    - Builds customer_node_id from customer coordinates.
    - Computes direct depot-to-customer distance.
    - Marks extreme distance outliers as not routing-eligible.

    note:
    - Amazon does not always provide a direct agent ID, so the prototype
      synthesizes one for representative-level routing and comparison.
    """
    out = _base_reconstruct_from_mapping(df, mapping)

    # Synthetic agent identity from depot + raw Amazon Agent_Age
    age_col = "Agent_Age"
    if age_col in df.columns:
        aligned_age = pd.to_numeric(df.loc[out.index, age_col], errors="coerce")
        out["agent_age"] = aligned_age.fillna(-1).astype(int)
        out["agent_id"] = (
            "AGENT-"
            + out["depot_id"].astype(str)
            + "-AGE-"
            + out["agent_age"].astype(str)
        )
    else:
        out["agent_age"] = -1
        out["agent_id"] = "AGENT-" + out["depot_id"].astype(str) + "-AGE-UNKNOWN"

    # Node identity: group destination coordinates within depot
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

    # Conservative outlier threshold from the old reconstruction guidance
    out["is_distance_outlier"] = out["direct_depot_customer_km"] > 50.0
    out["is_routing_eligible"] = ~out["is_distance_outlier"]

    # Fill weak fields gently
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
            "agent_age",
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

def build_amazon_order_routing_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Amazon preview routing should preserve order-level rows instead of collapsing
    repeated customer_node_id values into only 12 physical nodes per depot.

    The original customer_node_id is retained as physical_customer_node_id, while
    customer_node_id is made unique per order so each row can appear as a route stop.
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
        "order_id",
        "customer_id",
        "agent_id",
    ]
    missing = [c for c in required if c not in work.columns]
    if missing:
        raise HTTPException(
            status_code=400, detail=f"Missing Amazon routing columns: {missing}"
        )

    if "node_order_count" not in work.columns:
        work["node_order_count"] = 1
    if "direct_depot_customer_km" not in work.columns:
        work["direct_depot_customer_km"] = work.apply(
            lambda r: haversine_km(
                float(r["depot_lat"]),
                float(r["depot_lon"]),
                float(r["customer_lat"]),
                float(r["customer_lon"]),
            ),
            axis=1,
        )
    if "order_date" not in work.columns:
        work["order_date"] = pd.NaT

    out = work[
        [
            "depot_id",
            "depot_lat",
            "depot_lon",
            "customer_node_id",
            "customer_lat",
            "customer_lon",
            "order_id",
            "order_date",
            "customer_id",
            "agent_id",
            "customer_name",
            "observed_eta_min",
            "predicted_eta_min",
            "rating",
            "area",
            "node_order_count",
            "direct_depot_customer_km",
        ]
    ].copy()

    out["physical_customer_node_id"] = out["customer_node_id"].astype(str)
    out["order_id"] = out["order_id"].astype(str)
    out["customer_id"] = out["customer_id"].astype(str)
    out["agent_id"] = (
        out["agent_id"]
        .astype(str)
        .replace({"": "UNKNOWN", "nan": "UNKNOWN", "None": "UNKNOWN"})
    )
    out["depot_id"] = out["depot_id"].astype(str)

    # Unique route stop ID: avoids the old 12-node Amazon collapse while still
    # showing where repeated orders share the same physical customer node.
    out["customer_node_id"] = (
        out["physical_customer_node_id"].astype(str)
        + "-ORDER-"
        + out["order_id"].astype(str)
    )
    out["node_name"] = out["customer_name"].astype(str)

    return out.reset_index(drop=True)

def build_local_preview_subset_amazon(
    df: pd.DataFrame,
    num_representatives: int,
    max_total_stops: Optional[int] = None,
    initial_radius_km: float = 25.0,
    max_radius_km: float = 120.0,
    local_cap_km: float = 100.0,
    use_existing_agents: bool = True,
    strict_existing_agents: bool = True,
    min_nodes_per_rep: int = 1,
) -> pd.DataFrame:
    """
    Amazon-specific preview builder without the old max-3-per-rep cap.

    Amazon behavior:
    - uses only Amazon logic; Zomato flow is untouched
    - preserves order-level Amazon stops instead of collapsing to 12 physical nodes
    - preserves agent_id as rep_id
    - selects the strongest 6 agent groups from one depot
    - does not limit each selected agent to 3 customers/orders
    - keeps the total preview size controlled by max_total_stops/profile setting
    """
    work = df.copy()

    effective_reps = max(AMAZON_DEFAULT_REPRESENTATIVES, int(num_representatives))
    target_preview_rows = max(
        int(max_total_stops or AMAZON_MIN_PREVIEW_STOPS),
        effective_reps * max(2, int(min_nodes_per_rep)),
    )

    depot_lat, depot_lon, depot_cluster = choose_best_local_depot_cluster(
        work,
        candidate_pool_size=max(target_preview_rows * 3, effective_reps * 10),
        prefer_agent_coverage=True,
        min_agents=effective_reps,
    )

    print(f"chosen preview depot: ({depot_lat}, {depot_lon})")

    depot_cluster["to_depot_km"] = depot_cluster.apply(
        lambda r: haversine_km(
            depot_lat,
            depot_lon,
            float(r["customer_lat"]),
            float(r["customer_lon"]),
        ),
        axis=1,
    )

    radius = initial_radius_km
    local = depot_cluster[depot_cluster["to_depot_km"] <= radius].copy()

    while len(local) < target_preview_rows and radius < max_radius_km:
        radius *= 1.5
        local = depot_cluster[depot_cluster["to_depot_km"] <= radius].copy()

    # Keep the cluster local. If the radius/cap is too restrictive, refill from
    # the same chosen depot only, sorted by compactness.
    local = local[local["to_depot_km"] <= local_cap_km].copy()
    if len(local) < target_preview_rows:
        local = (
            depot_cluster.sort_values("to_depot_km")
            .head(target_preview_rows * 3)
            .copy()
        )

    local = local.sort_values("to_depot_km").copy()

    if "agent_id" in local.columns:
        local["agent_id"] = local["agent_id"].astype(str).fillna("UNKNOWN")
        local["agent_id"] = local["agent_id"].replace(
            {"": "UNKNOWN", "nan": "UNKNOWN", "None": "UNKNOWN"}
        )

        valid_local = local[local["agent_id"] != "UNKNOWN"].copy()

        if not valid_local.empty:
            agent_summary_rows: List[Dict[str, Any]] = []
            for agent_id, grp in valid_local.groupby("agent_id"):
                grp = grp.copy()
                rows = int(len(grp))
                physical_nodes = (
                    int(grp["physical_customer_node_id"].nunique())
                    if "physical_customer_node_id" in grp.columns
                    else int(grp["customer_node_id"].nunique())
                )
                mean_dist = float(grp["to_depot_km"].mean())
                max_dist = float(grp["to_depot_km"].max())
                mean_eta = float(
                    pd.to_numeric(grp["predicted_eta_min"], errors="coerce")
                    .fillna(0)
                    .mean()
                )
                mean_rating = float(
                    pd.to_numeric(grp["rating"], errors="coerce").fillna(4.0).mean()
                )

                # Prefer agents with enough available rows, compact stops,
                # reasonable ETA, and good rating. No per-rep max-3 cap here.
                shortage = 1 if rows < max(1, min_nodes_per_rep) else 0
                agent_summary_rows.append(
                    {
                        "agent_id": str(agent_id),
                        "rows": rows,
                        "physical_nodes": physical_nodes,
                        "mean_dist": mean_dist,
                        "max_dist": max_dist,
                        "mean_eta": mean_eta,
                        "mean_rating": mean_rating,
                        "score": (
                            shortage,
                            -rows,
                            mean_dist,
                            max_dist,
                            mean_eta,
                            -mean_rating,
                        ),
                    }
                )

            agent_summary = pd.DataFrame(agent_summary_rows)
            agent_summary = agent_summary.sort_values("score").reset_index(drop=True)
            top_agents = (
                agent_summary.head(effective_reps)["agent_id"].astype(str).tolist()
            )

            selected = valid_local[
                valid_local["agent_id"].astype(str).isin(top_agents)
            ].copy()
            selected["_pred_eta_sort"] = pd.to_numeric(
                selected["predicted_eta_min"], errors="coerce"
            ).fillna(selected["to_depot_km"] * 3.0 + 8.0)
            selected["_rating_sort"] = pd.to_numeric(
                selected["rating"], errors="coerce"
            ).fillna(4.0)

            selected = selected.sort_values(
                ["agent_id", "to_depot_km", "_pred_eta_sort", "_rating_sort"],
                ascending=[True, True, True, False],
            ).copy()

            # Control only the total Amazon preview size, not the per-rep size.
            # This allows naturally uneven agent workloads for the DEQ process.
            if len(selected) > target_preview_rows:
                counts = selected["agent_id"].value_counts()
                total = int(counts.sum())
                allocations: Dict[str, int] = {}

                for agent_id, cnt in counts.items():
                    share = max(
                        1, int(round((int(cnt) / max(1, total)) * target_preview_rows))
                    )
                    allocations[str(agent_id)] = min(int(cnt), share)

                allocated_total = sum(allocations.values())

                while allocated_total > target_preview_rows:
                    for agent_id in sorted(
                        allocations, key=allocations.get, reverse=True
                    ):
                        if (
                            allocations[agent_id] > 1
                            and allocated_total > target_preview_rows
                        ):
                            allocations[agent_id] -= 1
                            allocated_total -= 1

                while allocated_total < target_preview_rows:
                    for agent_id, cnt in counts.items():
                        aid = str(agent_id)
                        if (
                            allocations[aid] < int(cnt)
                            and allocated_total < target_preview_rows
                        ):
                            allocations[aid] += 1
                            allocated_total += 1

                selected_parts: List[pd.DataFrame] = []
                for agent_id in top_agents:
                    grp = selected[
                        selected["agent_id"].astype(str) == str(agent_id)
                    ].copy()
                    selected_parts.append(grp.head(allocations.get(str(agent_id), 0)))

                selected = (
                    pd.concat(selected_parts, ignore_index=True)
                    if selected_parts
                    else selected.head(0).copy()
                )

            selected["rep_id"] = selected["agent_id"].astype(str)

            print(
                "Amazon agent-based clustering active without max-3-per-rep cap; "
                f"selected agents: {selected['rep_id'].nunique()}, "
                f"target preview stops: {target_preview_rows}"
            )
            print(f"chosen preview depot: ({depot_lat}, {depot_lon})")
            print(
                f"chosen local preview stop count before rep assignment: {len(selected)}"
            )
            print(
                f"chosen local max distance from depot: "
                f"{float(selected['to_depot_km'].max()) if not selected.empty else 0.0:.2f} km"
            )
            print(f"local rows after all fallback stages: {len(selected)}")
            print(
                "customers/orders per selected agent:",
                selected["rep_id"].value_counts().to_dict(),
            )
            if "customer_node_id" in selected.columns:
                print(
                    f"distinct route customer_node_id in local: {selected['customer_node_id'].nunique()}"
                )
            if "physical_customer_node_id" in selected.columns:
                print(
                    f"distinct physical_customer_node_id in local: {selected['physical_customer_node_id'].nunique()}"
                )

            return selected.drop(
                columns=["to_depot_km", "_pred_eta_sort", "_rating_sort"],
                errors="ignore",
            ).reset_index(drop=True)

    # Fallback only if Amazon has no usable agent_id. This is not expected for
    # the reconstructed Amazon dataset, but keeps the backend safe.
    print("Amazon agent_id unavailable; falling back to uneven spatial assignment.")
    local = (
        local.head(target_preview_rows)
        .drop(columns=["to_depot_km"], errors="ignore")
        .copy()
    )
    preview_assigned = assign_preview_rep_ids_uneven(local, effective_reps)
    return preview_assigned

def amazon_distance_polish_assignment(
    assign_df: pd.DataFrame,
    speed_kmph: float,
    service_min: float,
    distance_matrix: Dict[str, Dict[str, float]],
    max_iterations: int = 12,
    min_distance_gain_km: float = 0.25,
    min_fairness_floor: float = 0.995,
    max_wbi_increase: float = 0.0,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Applies an Amazon-specific final distance improvement pass.

    Purpose:
    - Reviews Amazon assignments after DEQ rebalancing.
    - Attempts distance-improving swaps while preserving workload/fairness rules.

    Notes:
    - This is dataset-specific tuning for Amazon preview behavior.
    - It should not replace the general enhanced DEQ logic.
    """
    current = ensure_preview_node_ids(assign_df.copy()).reset_index(drop=True)
    logs: List[Dict[str, Any]] = []
    current_eval = evaluate_assignment(
        current, speed_kmph, service_min, distance_matrix
    )

    for iteration in range(1, max_iterations + 1):
        rep_ids = [str(x) for x in current["rep_id"].dropna().unique().tolist()]
        if len(rep_ids) < 2:
            break

        best_trial = None
        best_eval = None
        best_log = None
        best_score = 0.0

        # 1) Try distance-improving transfers.
        for idx, row in current.iterrows():
            source_rep = str(row["rep_id"])
            if int((current["rep_id"] == source_rep).sum()) <= 2:
                continue

            for target_rep in rep_ids:
                if target_rep == source_rep:
                    continue

                trial = current.copy()
                trial.loc[idx, "rep_id"] = target_rep
                trial_eval = evaluate_assignment(
                    trial, speed_kmph, service_min, distance_matrix
                )

                distance_gain = (
                    current_eval["total"]["distance_km"]
                    - trial_eval["total"]["distance_km"]
                )
                time_gain = (
                    current_eval["total"]["operational_minutes"]
                    - trial_eval["total"]["operational_minutes"]
                )
                wbi_increase = trial_eval["wbi"] - current_eval["wbi"]

                if distance_gain < min_distance_gain_km:
                    continue
                if time_gain < -1e-6:
                    continue
                if trial_eval["fairness"] < min_fairness_floor:
                    continue
                # WBI is sigma/mu, so lower is better. Do not accept an Amazon polish
                # move that makes WBI worse, even if distance improves.
                if wbi_increase > max_wbi_increase + 1e-9:
                    continue

                wbi_gain = current_eval["wbi"] - trial_eval["wbi"]
                score = distance_gain + (time_gain / 60.0) + max(0.0, wbi_gain * 20.0)
                if score > best_score:
                    best_score = score
                    best_trial = trial
                    best_eval = trial_eval
                    best_log = {
                        "iteration": iteration,
                        "move_type": "amazon_distance_transfer",
                        "moved_order": str(current.loc[idx, "order_id"]),
                        "from_rep": source_rep,
                        "to_rep": target_rep,
                        "fairness_before": round(current_eval["fairness"], 6),
                        "fairness_after": round(trial_eval["fairness"], 6),
                        "distance_before": round(
                            current_eval["total"]["distance_km"], 2
                        ),
                        "distance_after": round(trial_eval["total"]["distance_km"], 2),
                        "operational_before": round(
                            current_eval["total"]["operational_minutes"], 2
                        ),
                        "operational_after": round(
                            trial_eval["total"]["operational_minutes"], 2
                        ),
                        "distance_gain": round(distance_gain, 4),
                        "time_gain": round(time_gain, 4),
                        "wbi_before_pct": round(current_eval["wbi"] * 100.0, 2),
                        "wbi_after_pct": round(trial_eval["wbi"] * 100.0, 2),
                        "wbi_gain_pct": round(
                            (current_eval["wbi"] - trial_eval["wbi"]) * 100.0, 2
                        ),
                        "accepted": True,
                    }

        # 2) Try swaps too, because swaps usually preserve workload balance better.
        for idx_a in range(len(current)):
            rep_a = str(current.loc[idx_a, "rep_id"])
            for idx_b in range(idx_a + 1, len(current)):
                rep_b = str(current.loc[idx_b, "rep_id"])
                if rep_a == rep_b:
                    continue

                trial = current.copy()
                trial.loc[idx_a, "rep_id"] = rep_b
                trial.loc[idx_b, "rep_id"] = rep_a
                trial_eval = evaluate_assignment(
                    trial, speed_kmph, service_min, distance_matrix
                )

                distance_gain = (
                    current_eval["total"]["distance_km"]
                    - trial_eval["total"]["distance_km"]
                )
                time_gain = (
                    current_eval["total"]["operational_minutes"]
                    - trial_eval["total"]["operational_minutes"]
                )
                wbi_increase = trial_eval["wbi"] - current_eval["wbi"]

                if distance_gain < min_distance_gain_km:
                    continue
                if time_gain < -1e-6:
                    continue
                if trial_eval["fairness"] < min_fairness_floor:
                    continue
                # WBI is sigma/mu, so lower is better. Do not accept an Amazon polish
                # move that makes WBI worse, even if distance improves.
                if wbi_increase > max_wbi_increase + 1e-9:
                    continue

                wbi_gain = current_eval["wbi"] - trial_eval["wbi"]
                score = distance_gain + (time_gain / 60.0) + max(0.0, wbi_gain * 20.0)
                if score > best_score:
                    best_score = score
                    best_trial = trial
                    best_eval = trial_eval
                    best_log = {
                        "iteration": iteration,
                        "move_type": "amazon_distance_swap",
                        "moved_order": str(current.loc[idx_a, "order_id"]),
                        "swapped_with_order": str(current.loc[idx_b, "order_id"]),
                        "from_rep": rep_a,
                        "to_rep": rep_b,
                        "fairness_before": round(current_eval["fairness"], 6),
                        "fairness_after": round(trial_eval["fairness"], 6),
                        "distance_before": round(
                            current_eval["total"]["distance_km"], 2
                        ),
                        "distance_after": round(trial_eval["total"]["distance_km"], 2),
                        "operational_before": round(
                            current_eval["total"]["operational_minutes"], 2
                        ),
                        "operational_after": round(
                            trial_eval["total"]["operational_minutes"], 2
                        ),
                        "distance_gain": round(distance_gain, 4),
                        "time_gain": round(time_gain, 4),
                        "wbi_before_pct": round(current_eval["wbi"] * 100.0, 2),
                        "wbi_after_pct": round(trial_eval["wbi"] * 100.0, 2),
                        "wbi_gain_pct": round(
                            (current_eval["wbi"] - trial_eval["wbi"]) * 100.0, 2
                        ),
                        "accepted": True,
                    }

        if best_trial is None or best_eval is None:
            break

        current = best_trial.reset_index(drop=True)
        current_eval = best_eval
        logs.append(best_log)

    if logs:
        print("amazon distance polish accepted moves:", len(logs))
        print("amazon distance polish logs:", logs)

    return current, logs
