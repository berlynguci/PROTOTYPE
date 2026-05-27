import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import HTTPException

from config import (
    DEMO_PREVIEW_DEPOTS,
    MIN_FIXED_DEMO_AGENTS,
    MIN_FIXED_DEMO_NODES,
)
from services.dataset_service import parse_order_date_series
from services.distance_service import haversine_km, matrix_cost
from services.routing_node_service import ensure_preview_node_ids

from services.workload_service import (
    compute_customer_workload_contribution,
    compute_normalized_delay_series,
    compute_route_workload,
)

# ============================================================
# SECTION 8: Baseline routing and workload construction
# Purpose:
# - Assigns customers to representatives.
# - Builds route order using Greedy Nearest Neighbor.
# - Computes distance, travel time, operational time, and workload.
#
# note:
# - The baseline routing process follows a distance-driven GNN approach
#   using the distance matrix built from OSM/Dijkstra or fallback costs.
# ============================================================

def static_assignment(df: pd.DataFrame, reps: int) -> pd.DataFrame:
    """
    Creates a simple baseline assignment of customers to representatives.

    Purpose:
    - Sorts customers spatially by angle around the depot.
    - Distributes customers across the requested number of representatives.

    Notes:
    - This is a fallback/static assignment helper. Other dataset-specific
      assignment logic may be used for Amazon/Zomato preview runs.
    """
    work = df.copy()
    c_lat, c_lon = work["depot_lat"].median(), work["depot_lon"].median()
    work["angle"] = np.arctan2(
        work["customer_lat"] - c_lat, work["customer_lon"] - c_lon
    )
    work = work.sort_values(["angle", "customer_id"]).reset_index(drop=True)

    rep_ids = [f"REP-{i+1}" for i in range(reps)]
    base = len(work) // reps
    rem = len(work) % reps

    assignments: List[str] = []
    for i, rep in enumerate(rep_ids):
        size = base + (1 if i < rem else 0)
        assignments.extend([rep] * size)

    work["rep_id"] = assignments[: len(work)]
    return work.drop(columns=["angle"])


def route_one_rep(
    group: pd.DataFrame,
    speed_kmph: float,
    service_min: float,
    distance_matrix: Dict[str, Dict[str, float]],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    rows = ensure_preview_node_ids(group).to_dict("records")
    if not rows:
        return [], {
            "distance_km": 0.0,
            "travel_minutes": 0.0,
            "operational_minutes": 0.0,
        }

    current_node = "DEPOT"
    unvisited = rows[:]

    route: List[Dict[str, Any]] = []
    cumulative_distance = 0.0
    cumulative_eta = 0.0
    stop_no = 1

    while unvisited:
        best = min(
            unvisited,
            key=lambda r: matrix_cost(distance_matrix, current_node, str(r["node_id"])),
        )

        leg = matrix_cost(distance_matrix, current_node, str(best["node_id"]))
        cumulative_distance += leg

        travel_min = (leg / speed_kmph) * 60.0 if speed_kmph > 0 else 0.0
        service_component = float(service_min)
        cumulative_eta += travel_min + service_component

        route.append(
            {
                "stopNumber": stop_no,
                "nodeId": best.get("customer_node_id", best["customer_id"]),
                "nodeName": best["customer_name"],
                "orderCount": int(best.get("node_order_count", 1)),
                "lat": float(best["customer_lat"]),
                "lon": float(best["customer_lon"]),
                "legDistance": round(leg, 2),
                "cumulativeDistance": round(cumulative_distance, 2),
                "eta": round(cumulative_eta, 2),
                "orderId": best["order_id"],
                "predictedEtaMin": round(float(best.get("predicted_eta_min", 0.0)), 2),
            }
        )

        current_node = str(best["node_id"])
        unvisited.remove(best)
        stop_no += 1

    return_leg = matrix_cost(distance_matrix, current_node, "DEPOT")
    total_distance = cumulative_distance + return_leg
    travel_minutes = (total_distance / speed_kmph) * 60.0 if speed_kmph > 0 else 0.0
    operational_minutes = travel_minutes + (len(rows) * float(service_min))
    return route, {
        "distance_km": total_distance,
        "travel_minutes": travel_minutes,
        "operational_minutes": operational_minutes,
    }

def route_all(
    assign_df: pd.DataFrame,
    speed_kmph: float,
    service_min: float,
    name: str,
    distance_matrix: Dict[str, Dict[str, float]],
) -> Tuple[List[Dict[str, Any]], pd.DataFrame, Dict[str, float]]:
    """
    Routes all representatives and summarizes their route statistics.

    Purpose:
    - Groups assigned customers by representative.
    - Calls route_one_rep for each representative.
    - Builds route objects for the frontend map/table.
    - Creates per-representative workload, distance, and time summaries.

    Used by:
    - Baseline run.
    - Enhanced evaluation.
    - Add-customer rerouting.
    """
    work = ensure_preview_node_ids(assign_df)
    routes = []
    rep_rows = []

    palette = [
        "#2563eb",
        "#16a34a",
        "#dc2626",
        "#ca8a04",
        "#9333ea",
        "#0891b2",
        "#db2777",
        "#4f46e5",
    ]

    for idx, (rep_id, grp) in enumerate(work.groupby("rep_id"), start=1):
        ordered_stops, stats = route_one_rep(
            grp, speed_kmph, service_min, distance_matrix
        )
        color = palette[(idx - 1) % len(palette)]

        routes.append(
            {
                "id": f"{name}-{rep_id}",
                "representativeId": rep_id,
                "representativeName": rep_id,
                "color": color,
                "stops": ordered_stops,
            }
        )

        rep_rows.append(
            {
                "rep_id": rep_id,
                "customers": int(len(grp)),
                "workload_min": float(stats["operational_minutes"]),
                "distance_km": float(stats["distance_km"]),
                "travel_minutes": float(stats["travel_minutes"]),
                "operational_minutes": float(stats["operational_minutes"]),
                "centroid_lat": float(grp["customer_lat"].mean()),
                "centroid_lon": float(grp["customer_lon"].mean()),
            }
        )

    rep_df = pd.DataFrame(rep_rows)
    total = {
        "distance_km": float(rep_df["distance_km"].sum()) if not rep_df.empty else 0.0,
        "travel_minutes": (
            float(rep_df["travel_minutes"].sum()) if not rep_df.empty else 0.0
        ),
        "operational_minutes": (
            float(rep_df["operational_minutes"].sum()) if not rep_df.empty else 0.0
        ),
    }
    return routes, rep_df, total

# ============================================================
# SECTION 5: Preview, depot selection, and routing-node helpers
# Purpose:
# - Prepares the subset of data used for demo-scale routing.
# - Selects fixed or fallback depots for repeatable experiments.
# - Ensures each customer/order has a usable routing node ID.
#
# note:
# - The prototype uses preview-sized routing runs to keep computation
#   practical during demonstration while preserving the routing logic.
# ============================================================


def filter_df_to_demo_depot(
    df: pd.DataFrame,
    dataset_role: str,
    min_nodes: int = MIN_FIXED_DEMO_NODES,
    min_agents: int = MIN_FIXED_DEMO_AGENTS,
) -> pd.DataFrame:
    """
    Filters the dataset to a configured demo depot when available.

    Purpose:
    - Uses fixed demo depots for Amazon, Zomato, or generic uploads.
    - Falls back to the strongest available depot if the configured depot
      is missing or too weak.

    Used by:
    - Baseline run preparation.

    note:
    - Fixed depots make baseline/enhanced comparisons repeatable.
    """
    demo_depot_id = DEMO_PREVIEW_DEPOTS.get(dataset_role)

    if not demo_depot_id or "depot_id" not in df.columns:
        return df.copy()

    filtered = df[df["depot_id"].astype(str) == str(demo_depot_id)].copy()

    if filtered.empty:
        print(
            f"demo depot override {demo_depot_id} not found; selecting strongest available depot instead"
        )
        fallback_depot_id = choose_best_demo_depot_id(
            df, min_nodes=min_nodes, min_agents=min_agents
        )
        if fallback_depot_id is None:
            return df.copy()
        filtered = df[df["depot_id"].astype(str) == str(fallback_depot_id)].copy()
        print(
            f"using fallback demo depot: {fallback_depot_id} ({len(filtered)} rows before preview trimming)"
        )
        return filtered

    summary = summarize_demo_depot_strength(filtered)
    print(f"configured fixed demo depot: {demo_depot_id} summary: {summary}")

    too_weak = summary["nodes"] < min_nodes or summary["agents"] < min_agents

    if too_weak:
        print(
            f"configured demo depot {demo_depot_id} is too weak "
            f"(needs at least {min_nodes} nodes and {min_agents} agents). "
            f"Selecting strongest available depot instead."
        )
        fallback_depot_id = choose_best_demo_depot_id(
            df, min_nodes=min_nodes, min_agents=min_agents
        )
        if fallback_depot_id and fallback_depot_id != str(demo_depot_id):
            filtered = df[df["depot_id"].astype(str) == str(fallback_depot_id)].copy()
            print(
                f"using stronger fallback demo depot: {fallback_depot_id} ({len(filtered)} rows before preview trimming)"
            )
            return filtered

    print(
        f"using fixed demo depot: {demo_depot_id} ({len(filtered)} rows before preview trimming)"
    )
    return filtered

def summarize_demo_depot_strength(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Summarizes whether a depot has enough usable routing data.

    Purpose:
    - Counts rows, unique customer nodes, agents, and orders.
    - Helps determine if a depot is strong enough for demo routing.

    Notes:
    - This avoids selecting depots with too few customers or representatives.
    """
    work = df.copy()
    if work.empty:
        return {
            "rows": 0,
            "nodes": 0,
            "agents": 0,
            "orders": 0,
        }

    nodes = (
        int(work["customer_node_id"].nunique())
        if "customer_node_id" in work.columns
        else int(len(work))
    )
    agents = 0
    if "agent_id" in work.columns:
        agent_series = (
            work["agent_id"]
            .astype(str)
            .replace({"": "UNKNOWN", "nan": "UNKNOWN", "None": "UNKNOWN"})
        )
        agents = int(agent_series[agent_series != "UNKNOWN"].nunique())

    orders = 0
    if "node_order_count" in work.columns:
        orders = int(
            pd.to_numeric(work["node_order_count"], errors="coerce").fillna(1).sum()
        )
    else:
        orders = int(len(work))

    return {
        "rows": int(len(work)),
        "nodes": nodes,
        "agents": agents,
        "orders": orders,
    }


def choose_best_demo_depot_id(
    routing_df: pd.DataFrame,
    min_nodes: int = MIN_FIXED_DEMO_NODES,
    min_agents: int = MIN_FIXED_DEMO_AGENTS,
) -> Optional[str]:
    """
    Selects the strongest available depot when the configured demo depot
    is unavailable or insufficient.

    Purpose:
    - Scores depots based on agent count, customer nodes, and orders.
    - Returns the best candidate depot ID for preview routing.
    """
    if routing_df.empty or "depot_id" not in routing_df.columns:
        return None

    best_score = None
    best_depot_id = None

    for depot_id, grp in routing_df.groupby("depot_id"):
        summary = summarize_demo_depot_strength(grp)

        score = (
            0 if summary["nodes"] >= min_nodes else 1,
            0 if summary["agents"] >= min_agents else 1,
            -summary["agents"],
            -summary["nodes"],
            -summary["orders"],
        )

        print(
            "demo depot candidate:",
            {
                "depot_id": str(depot_id),
                **summary,
            },
        )

        if best_score is None or score < best_score:
            best_score = score
            best_depot_id = str(depot_id)

    print("best demo depot selected:", best_depot_id, "score:", best_score)
    return best_depot_id

def select_spatially_spread_rows(
    df: pd.DataFrame,
    target_n: int,
    depot_lat: float,
    depot_lon: float,
) -> pd.DataFrame:
    if df.empty or target_n <= 0:
        return df.head(0).copy()

    pool = df.copy()

    # Start with the nearest point to depot
    pool["to_depot_km"] = pool.apply(
        lambda r: haversine_km(
            r["customer_lat"], r["customer_lon"], depot_lat, depot_lon
        ),
        axis=1,
    )
    pool = pool.sort_values("to_depot_km").reset_index(drop=True)

    selected_idx = [0]
    remaining = set(range(1, len(pool)))

    while len(selected_idx) < min(target_n, len(pool)) and remaining:
        best_i = None
        best_score = -1.0

        for i in remaining:
            row = pool.iloc[i]
            min_dist_to_selected = min(
                haversine_km(
                    row["customer_lat"],
                    row["customer_lon"],
                    pool.iloc[j]["customer_lat"],
                    pool.iloc[j]["customer_lon"],
                )
                for j in selected_idx
            )

            # prefer points that are still reasonably near depot,
            # but also far from already selected points
            depot_dist = row["to_depot_km"]
            score = min_dist_to_selected - (0.15 * depot_dist)

            if score > best_score:
                best_score = score
                best_i = i

        if best_i is None:
            break

        selected_idx.append(best_i)
        remaining.remove(best_i)

    out = pool.iloc[selected_idx].copy()
    out = out.drop(columns=["to_depot_km"], errors="ignore")
    return out


def assign_preview_rep_ids_from_agent(
    preview_df: pd.DataFrame,
    num_representatives: int,
    max_total_stops: Optional[int] = None,
    strict_existing_agents: bool = False,
    cap_total_stops: bool = True,
) -> pd.DataFrame:
    if preview_df.empty:
        return preview_df.copy()

    work = preview_df.copy().reset_index(drop=True)

    if "agent_id" not in work.columns:
        return assign_preview_rep_ids_uneven(work, num_representatives)

    work["agent_id"] = work["agent_id"].astype(str).fillna("UNKNOWN")
    work["agent_id"] = work["agent_id"].replace(
        {"": "UNKNOWN", "nan": "UNKNOWN", "None": "UNKNOWN"}
    )

    valid = work[work["agent_id"] != "UNKNOWN"].copy()
    if valid.empty:
        return assign_preview_rep_ids_uneven(work, num_representatives)

    agent_counts = valid.groupby("agent_id").size().sort_values(ascending=False)

    unique_agents = agent_counts.index.tolist()
    if strict_existing_agents:
        top_agents = unique_agents
    else:
        if len(unique_agents) < num_representatives:
            print(
                f"existing agent-based preview has only {len(unique_agents)} unique agents "
                f"for {num_representatives} requested reps."
            )
            return assign_preview_rep_ids_uneven(work, num_representatives)
        top_agents = unique_agents[:num_representatives]

    filtered = valid[valid["agent_id"].isin(top_agents)].copy()

    if len(filtered) < num_representatives:
        print(
            f"existing-agent filtered rows too small ({len(filtered)} rows) for "
            f"{num_representatives} requested reps."
        )
        if strict_existing_agents:
            filtered["rep_id"] = filtered["agent_id"].astype(str)
            return filtered.drop(columns=["to_depot_km"], errors="ignore").reset_index(
                drop=True
            )
        return assign_preview_rep_ids_uneven(work, num_representatives)

    filtered["to_depot_km"] = filtered.apply(
        lambda r: haversine_km(
            float(r["depot_lat"]),
            float(r["depot_lon"]),
            float(r["customer_lat"]),
            float(r["customer_lon"]),
        ),
        axis=1,
    )

    filtered = filtered.sort_values(["agent_id", "to_depot_km"]).copy()

    # Zomato/default preview can still cap rows. Amazon calls this with
    # cap_total_stops=False so all selected local order rows survive.
    if (
        cap_total_stops
        and max_total_stops is not None
        and len(filtered) > max_total_stops
    ):
        counts = filtered["agent_id"].value_counts()
        total = counts.sum()

        keep_rows = []
        allocations = {}
        for agent_id, cnt in counts.items():
            share = max(1, int(round((cnt / total) * max_total_stops)))
            allocations[agent_id] = min(cnt, share)

        allocated_total = sum(allocations.values())

        while allocated_total > max_total_stops:
            for agent_id in sorted(allocations, key=allocations.get, reverse=True):
                if allocations[agent_id] > 1 and allocated_total > max_total_stops:
                    allocations[agent_id] -= 1
                    allocated_total -= 1

        while allocated_total < max_total_stops:
            for agent_id, cnt in counts.items():
                if allocations[agent_id] < cnt and allocated_total < max_total_stops:
                    allocations[agent_id] += 1
                    allocated_total += 1

        for agent_id in counts.index:
            grp = filtered[filtered["agent_id"] == agent_id].copy()
            keep_rows.append(grp.head(allocations[agent_id]))

        filtered = pd.concat(keep_rows, ignore_index=True)

    filtered["rep_id"] = filtered["agent_id"].astype(str)
    return filtered.drop(columns=["to_depot_km"], errors="ignore").reset_index(
        drop=True
    )


def assign_preview_rep_ids_uneven(
    preview_df: pd.DataFrame,
    num_representatives: int,
) -> pd.DataFrame:
    if preview_df.empty:
        return preview_df.copy()

    work = preview_df.copy().reset_index(drop=True)

    depot_lat = float(work["depot_lat"].iloc[0])
    depot_lon = float(work["depot_lon"].iloc[0])

    # Angle of each customer relative to depot
    work["angle"] = work.apply(
        lambda r: math.atan2(
            float(r["customer_lat"]) - depot_lat,
            float(r["customer_lon"]) - depot_lon,
        ),
        axis=1,
    )

    # Secondary sort by distance to keep each sector internally coherent
    work["to_depot_km"] = work.apply(
        lambda r: haversine_km(
            depot_lat,
            depot_lon,
            float(r["customer_lat"]),
            float(r["customer_lon"]),
        ),
        axis=1,
    )

    work = work.sort_values(["angle", "to_depot_km"]).reset_index(drop=True)

    rep_ids = [f"REP-{i}" for i in range(1, num_representatives + 1)]
    n = len(work)

    # Slightly uneven but still spatially grouped
    base = n // num_representatives
    rem = n % num_representatives

    sizes = [base] * num_representatives

    # Front-load only a little, not a hardcoded 4-3-3-2 row chunk pattern
    for i in range(rem):
        sizes[i] += 1

    # Optional slight imbalance for the first rep if possible
    # if num_representatives > 1 and n >= num_representatives * 2:
    #     for j in range(num_representatives - 1, 0, -1):
    #         if sizes[j] > 1:
    #             sizes[0] += 1
    #             sizes[j] -= 1
    #             break

    assigned = []
    for rep_id, size in zip(rep_ids, sizes):
        assigned.extend([rep_id] * size)

    work["rep_id"] = assigned[:n]

    return work.drop(columns=["angle", "to_depot_km"], errors="ignore")


def choose_best_local_depot_cluster(
    df: pd.DataFrame,
    candidate_pool_size: int = 12,
    prefer_agent_coverage: bool = False,
    min_agents: int = 1,
) -> Tuple[float, float, pd.DataFrame]:
    """
    Choose the depot whose nearby customer cluster is most suitable for preview.

    Default behavior:
    - prefers compact clusters

    When prefer_agent_coverage=True:
    - prefers more distinct agents
    - then higher total order demand
    - then more nearby nodes
    - then compactness
    """
    work = df.copy()

    depot_groups = (
        work.groupby(["depot_lat", "depot_lon"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
    )

    if depot_groups.empty:
        raise HTTPException(
            status_code=400, detail="No depot coordinates available for preview."
        )

    best_score = None
    best_depot_lat = None
    best_depot_lon = None
    best_cluster = None

    for depot in depot_groups.itertuples(index=False):
        depot_lat = float(depot.depot_lat)
        depot_lon = float(depot.depot_lon)

        cluster = work[
            (work["depot_lat"] == depot_lat) & (work["depot_lon"] == depot_lon)
        ].copy()

        if cluster.empty:
            continue

        cluster["to_depot_km"] = cluster.apply(
            lambda r: haversine_km(
                depot_lat,
                depot_lon,
                float(r["customer_lat"]),
                float(r["customer_lon"]),
            ),
            axis=1,
        )

        if "customer_node_id" in cluster.columns:
            cluster = (
                cluster.sort_values("to_depot_km")
                .drop_duplicates(subset=["customer_node_id"])
                .copy()
            )
        else:
            cluster["lat_round"] = cluster["customer_lat"].round(4)
            cluster["lon_round"] = cluster["customer_lon"].round(4)
            cluster = (
                cluster.sort_values("to_depot_km")
                .drop_duplicates(subset=["lat_round", "lon_round"])
                .copy()
            )
            cluster = cluster.drop(columns=["lat_round", "lon_round"], errors="ignore")

        if cluster.empty:
            continue

        nearest = cluster.nsmallest(candidate_pool_size, "to_depot_km").copy()

        distinct_agents = 0
        if "agent_id" in nearest.columns:
            agent_series = (
                nearest["agent_id"]
                .astype(str)
                .replace({"": "UNKNOWN", "nan": "UNKNOWN", "None": "UNKNOWN"})
            )
            distinct_agents = int(agent_series[agent_series != "UNKNOWN"].nunique())

        total_orders = 0.0
        if "node_order_count" in cluster.columns:
            total_orders = float(
                pd.to_numeric(cluster["node_order_count"], errors="coerce")
                .fillna(1)
                .sum()
            )
        else:
            total_orders = float(len(cluster))

        nearby_orders = 0.0
        if "node_order_count" in nearest.columns:
            nearby_orders = float(
                pd.to_numeric(nearest["node_order_count"], errors="coerce")
                .fillna(1)
                .sum()
            )
        else:
            nearby_orders = float(len(nearest))

        nearby_nodes = int(len(nearest))
        mean_dist = float(nearest["to_depot_km"].mean())
        max_dist = float(nearest["to_depot_km"].max())

        if prefer_agent_coverage:
            insufficient_agent_penalty = 1 if distinct_agents < min_agents else 0
            score = (
                insufficient_agent_penalty,
                -distinct_agents,
                -total_orders,
                -nearby_orders,
                -nearby_nodes,
                mean_dist,
                max_dist,
            )
        else:
            score = (
                -nearby_nodes,
                -distinct_agents,
                -total_orders,
                mean_dist,
                max_dist,
            )

        print(
            "candidate depot:",
            {
                "depot_lat": depot_lat,
                "depot_lon": depot_lon,
                "distinct_agents": distinct_agents,
                "total_orders": round(total_orders, 2),
                "nearby_orders": round(nearby_orders, 2),
                "nearby_nodes": nearby_nodes,
                "mean_dist": round(mean_dist, 2),
                "max_dist": round(max_dist, 2),
            },
        )

        if best_score is None or score < best_score:
            best_score = score
            best_depot_lat = depot_lat
            best_depot_lon = depot_lon
            best_cluster = cluster.copy()

    if best_cluster is None:
        raise HTTPException(
            status_code=400, detail="Could not build a local depot preview cluster."
        )

    print(
        "selected depot cluster:",
        {
            "depot_lat": best_depot_lat,
            "depot_lon": best_depot_lon,
            "score": best_score,
        },
    )

    return best_depot_lat, best_depot_lon, best_cluster


def build_local_preview_subset(
    df: pd.DataFrame,
    num_representatives: int,
    max_total_stops: int = 12,
    initial_radius_km: float = 7.0,
    max_radius_km: float = 16.0,
    local_cap_km: float = 14.0,
    use_existing_agents: bool = False,
    strict_existing_agents: bool = False,
    min_nodes_per_rep: int = 3,
) -> pd.DataFrame:
    work = df.copy()

    if "order_date" in work.columns:
        work["order_date"] = parse_order_date_series(work["order_date"])
        valid_dates = sorted(work["order_date"].dropna().unique())

        if len(valid_dates) > 0:
            selected_dates = [valid_dates[-1]]
            dated = work[work["order_date"].isin(selected_dates)].copy()

            # Expand backward in time until we have enough candidate rows
            idx = len(valid_dates) - 2
            target_min_rows = max(max_total_stops, num_representatives * 2)

            while len(dated) < target_min_rows and idx >= 0:
                selected_dates.append(valid_dates[idx])
                dated = work[work["order_date"].isin(selected_dates)].copy()
                idx -= 1

            work = dated.copy()
            selected_dates_sorted = sorted(pd.to_datetime(selected_dates))
            print(
                "order_date window used for preview:",
                [d.strftime("%Y-%m-%d") for d in selected_dates_sorted],
            )

    if len(work) < num_representatives:
        print("date-filtered preview too small, falling back to all dates")
        work = df.copy()

    depot_lat, depot_lon, depot_cluster = choose_best_local_depot_cluster(
        work,
        candidate_pool_size=max(max_total_stops, num_representatives * 3),
        prefer_agent_coverage=use_existing_agents,
        min_agents=num_representatives,
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

    while len(local) < max_total_stops and radius < max_radius_km:
        radius *= 1.5
        local = depot_cluster[depot_cluster["to_depot_km"] <= radius].copy()

    target_local_nodes = max(max_total_stops, num_representatives * min_nodes_per_rep)

    if local.empty:
        local = depot_cluster.nsmallest(target_local_nodes, "to_depot_km").copy()
    else:
        local = local.nsmallest(target_local_nodes, "to_depot_km").copy()

    if "customer_node_id" in local.columns:
        local = (
            local.sort_values("to_depot_km")
            .drop_duplicates(subset=["customer_node_id"])
            .copy()
        )
    else:
        local["lat_round"] = local["customer_lat"].round(4)
        local["lon_round"] = local["customer_lon"].round(4)
        local = (
            local.sort_values("to_depot_km")
            .drop_duplicates(subset=["lat_round", "lon_round"])
            .copy()
        )
        local = local.drop(columns=["lat_round", "lon_round"], errors="ignore")

    # IMPORTANT:
    # for preview mode, keep the nearest local stops only.
    # do not spatially spread them outward.
    local = local.sort_values("to_depot_km").copy()

    # Hard cap for local preview geometry
    local = local[local["to_depot_km"] <= local_cap_km].copy()

    if use_existing_agents:
        # Keep a larger pool first so real agent coverage survives.
        if len(local) < max(max_total_stops * 2, num_representatives * 3):
            refill = depot_cluster.sort_values("to_depot_km").copy()
            refill = refill.head(
                max(max_total_stops * 3, num_representatives * 4)
            ).copy()
            local = refill.copy()
    else:

        target_local_nodes = max(
            max_total_stops, num_representatives * min_nodes_per_rep, 24
        )

        if len(local) >= target_local_nodes:
            local = local.head(target_local_nodes).copy()
        else:
            refill = depot_cluster.sort_values("to_depot_km").copy()
            refill = refill.head(target_local_nodes).copy()
            local = refill.copy()

    # If still too small, refill from the same chosen depot cluster only
    # If still too small, first refill from the same chosen depot cluster with a much bigger pool
    if len(local) < num_representatives:
        refill_target = max(max_total_stops * 4, num_representatives * 6)
        refill = depot_cluster.nsmallest(refill_target, "to_depot_km").copy()
        if "customer_node_id" in refill.columns:
            refill = (
                refill.sort_values("to_depot_km")
                .drop_duplicates(subset=["customer_node_id"])
                .copy()
            )
        local = refill.copy()

    # If still too small, refill from the same chosen depot cluster with a much bigger pool
    if len(local) < num_representatives:
        refill_target = max(max_total_stops * 4, num_representatives * 6)
        refill = depot_cluster.nsmallest(refill_target, "to_depot_km").copy()
        if "customer_node_id" in refill.columns:
            refill = (
                refill.sort_values("to_depot_km")
                .drop_duplicates(subset=["customer_node_id"])
                .copy()
            )
        local = refill.copy()

    # Absolute fallback: use all dates, but still only for the same chosen depot
    if len(local) < num_representatives:
        same_depot_all_dates = df[
            (df["depot_lat"] == depot_lat) & (df["depot_lon"] == depot_lon)
        ].copy()

        same_depot_all_dates["to_depot_km"] = same_depot_all_dates.apply(
            lambda r: haversine_km(
                depot_lat,
                depot_lon,
                float(r["customer_lat"]),
                float(r["customer_lon"]),
            ),
            axis=1,
        )

        if "customer_node_id" in same_depot_all_dates.columns:
            same_depot_all_dates = (
                same_depot_all_dates.sort_values("to_depot_km")
                .drop_duplicates(subset=["customer_node_id"])
                .copy()
            )

        local = same_depot_all_dates.head(
            max(max_total_stops * 4, num_representatives * 5)
        ).copy()

    print(f"chosen preview depot: ({depot_lat}, {depot_lon})")
    print(f"chosen local preview stop count before rep assignment: {len(local)}")
    print(
        f"chosen local max distance from depot: {float(local['to_depot_km'].max()) if not local.empty else 0.0:.2f} km"
    )
    print(
        f"final preview local max distance before drop: {float(local['to_depot_km'].max()) if not local.empty else 0.0:.2f} km"
    )

    print(f"local rows after all fallback stages: {len(local)}")
    if "agent_id" in local.columns:
        print(
            f"distinct agent_id in local: {local['agent_id'].astype(str).replace({'': 'UNKNOWN', 'nan': 'UNKNOWN', 'None': 'UNKNOWN'}).nunique()}"
        )
    if "customer_node_id" in local.columns:
        print(
            f"distinct customer_node_id in local: {local['customer_node_id'].nunique()}"
        )

    local = local.drop(columns=["to_depot_km"], errors="ignore").copy()

    if use_existing_agents and strict_existing_agents and "agent_id" in local.columns:
        strict_local = local.copy()
        strict_local["agent_id"] = (
            strict_local["agent_id"].astype(str).fillna("UNKNOWN")
        )
        strict_local["agent_id"] = strict_local["agent_id"].replace(
            {"": "UNKNOWN", "nan": "UNKNOWN", "None": "UNKNOWN"}
        )

        strict_local = strict_local[strict_local["agent_id"] != "UNKNOWN"].copy()

        if not strict_local.empty:
            strict_local["rep_id"] = strict_local["agent_id"].astype(str)
            print(
                "strict existing-agent preservation active; "
                f"keeping all available real agents: {strict_local['rep_id'].nunique()}"
            )
            return strict_local.reset_index(drop=True)

    if use_existing_agents:
        preview_assigned = assign_preview_rep_ids_from_agent(
            local,
            num_representatives,
            max_total_stops=max_total_stops,
            strict_existing_agents=strict_existing_agents,
        )
        if not preview_assigned.empty and preview_assigned["rep_id"].nunique() > 0:
            return preview_assigned

    preview_assigned = assign_preview_rep_ids_uneven(local, num_representatives)
    return preview_assigned

def preview_summary_from_assign_df(assign_df: pd.DataFrame) -> Dict[str, Any]:
    if assign_df.empty:
        return {
            "selectionStrategy": "single-depot high-node local preview",
            "maxRoutes": 0,
            "maxTotalStops": 0,
            "maxDistanceFromDepotKm": 0.0,
            "depotLat": None,
            "depotLon": None,
        }

    depot_row = assign_df.iloc[0]
    depot_lat = float(depot_row["depot_lat"])
    depot_lon = float(depot_row["depot_lon"])

    distances = assign_df.apply(
        lambda r: haversine_km(
            r["customer_lat"], r["customer_lon"], depot_lat, depot_lon
        ),
        axis=1,
    )

    return {
        "selectionStrategy": "single-depot nearest-customer compact preview",
        "maxRoutes": int(assign_df["rep_id"].nunique()),
        "maxTotalStops": int(len(assign_df)),
        "maxDistanceFromDepotKm": round(float(distances.max()), 2),
        "depotLat": depot_lat,
        "depotLon": depot_lon,
    }
