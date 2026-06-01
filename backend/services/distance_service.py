from typing import Any, Dict, List, Optional, Tuple

import math
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from backend.config import EARTH_RADIUS_KM, OSM_CACHE_DIR
from backend.services.routing_node_service import ensure_preview_node_ids

# ============================================================
# SECTION 4: Distance, OSM, and map geometry helpers
# Purpose:
# - Computes direct and road-adjusted distance estimates.
# - Builds preview distance matrices for baseline and enhanced routing.
# - Uses OSM/Dijkstra when feasible, with a proxy fallback when the
#   preview area is too large or OSM lookup fails.
# - Generates map display geometry for route polylines.
#
# Note:
# - The baseline routing uses Greedy Nearest Neighbor over a distance
#   matrix. OSM shortest paths are preferred for route realism, while
#   the road-adjusted fallback keeps the prototype fast and reliable.
# ============================================================

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes straight-line distance between two latitude/longitude points.

    Used by:
    - Fallback distance estimation.
    - Depot/customer filtering.
    - Candidate selection and workload calculations.

    Notes:
    - This is faster than OSM routing and is used as a reliable fallback.
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def road_adjusted_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes a fast road-distance proxy from haversine distance.

    Purpose:
    - Inflates straight-line distance to approximate real travel distance.
    - Keeps routing responsive when OSM graph lookup is skipped or fails.

    Notes:
    - This is not a true road-network path, but it is more realistic than
      pure straight-line distance.
    """
    direct = haversine_km(lat1, lon1, lat2, lon2)
    return direct * 1.25


def _expand_bbox(
    lat_min: float,
    lon_min: float,
    lat_max: float,
    lon_max: float,
    pad_ratio: float = 0.20,
    min_pad_deg: float = 0.01,
) -> Tuple[float, float, float, float]:
    lat_span = max(lat_max - lat_min, 0.0)
    lon_span = max(lon_max - lon_min, 0.0)

    lat_pad = max(lat_span * pad_ratio, min_pad_deg)
    lon_pad = max(lon_span * pad_ratio, min_pad_deg)

    south = lat_min - lat_pad
    west = lon_min - lon_pad
    north = lat_max + lat_pad
    east = lon_max + lon_pad
    return south, west, north, east


def _graph_cache_name_from_bbox(
    south: float,
    west: float,
    north: float,
    east: float,
) -> str:
    return f"osm_drive_{south:.5f}_{west:.5f}_{north:.5f}_{east:.5f}.graphml".replace(
        "-", "m"
    )


def build_preview_points(assign_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build unique depot/customer preview points for matrix construction.
    """
    work = ensure_preview_node_ids(assign_df)
    if work.empty:
        return pd.DataFrame(columns=["point_id", "lat", "lon", "kind"])

    depot_lat = float(work.iloc[0]["depot_lat"])
    depot_lon = float(work.iloc[0]["depot_lon"])

    rows: List[Dict[str, Any]] = [
        {"point_id": "DEPOT", "lat": depot_lat, "lon": depot_lon, "kind": "depot"}
    ]

    seen = set()
    for row in work.itertuples(index=False):
        pid = str(row.node_id)
        if pid in seen:
            continue
        seen.add(pid)
        rows.append(
            {
                "point_id": pid,
                "lat": float(row.customer_lat),
                "lon": float(row.customer_lon),
                "kind": "customer",
            }
        )

    return pd.DataFrame(rows)


def load_or_build_osm_preview_graph(points_df: pd.DataFrame):
    if points_df.empty:
        return None

    depot_rows = points_df[points_df["kind"] == "depot"].copy()
    if depot_rows.empty:
        return None

    depot_lat = float(depot_rows.iloc[0]["lat"])
    depot_lon = float(depot_rows.iloc[0]["lon"])

    max_preview_km = 0.0
    for row in points_df.itertuples(index=False):
        d = haversine_km(depot_lat, depot_lon, float(row.lat), float(row.lon))
        if d > max_preview_km:
            max_preview_km = d

    graph_radius_km = min(max(3.0, max_preview_km * 1.15), 20.0)
    graph_radius_m = int(graph_radius_km * 1000.0)

    cache_name = (
        f"osm_point_{depot_lat:.5f}_{depot_lon:.5f}_{graph_radius_m}m.graphml".replace(
            "-", "m"
        )
    )
    cache_path = OSM_CACHE_DIR / cache_name

    if cache_path.exists():
        G = ox.load_graphml(cache_path)
    else:
        G = ox.graph_from_point(
            (depot_lat, depot_lon),
            dist=graph_radius_m,
            network_type="drive",
            simplify=True,
        )
        ox.save_graphml(G, cache_path)

    return ox.project_graph(G)


def snap_preview_points_to_osm(points_df: pd.DataFrame, G_proj) -> pd.DataFrame:
    """
    Snap preview depot/customers to nearest OSM nodes.
    """
    out = points_df.copy()
    if out.empty or G_proj is None:
        out["osm_node"] = np.nan
        return out

    import geopandas as gpd
    from shapely.geometry import Point

    pts = gpd.GeoDataFrame(
        out.copy(),
        geometry=[Point(lon, lat) for lon, lat in zip(out["lon"], out["lat"])],
        crs="EPSG:4326",
    ).to_crs(G_proj.graph["crs"])

    xs = pts.geometry.x.to_numpy()
    ys = pts.geometry.y.to_numpy()

    out["osm_node"] = ox.distance.nearest_nodes(G_proj, X=xs, Y=ys)
    return out


def build_preview_distance_matrix(
    assign_df: pd.DataFrame,
    osm_threshold_km: float = 14.0,
) -> Dict[str, Dict[str, float]]:
    """
    Builds the pairwise distance matrix used by the routing algorithm.

    Purpose:
    - Creates depot-to-customer and customer-to-customer costs.
    - Uses OSM shortest-path distance when the preview area is manageable.
    - Falls back to road-adjusted haversine distance for large/spread-out
      previews or OSM failures.

    Used by:
    - Baseline route generation.
    - Enhanced DEQ evaluation and rerouting.

    note:
    - This is where Dijkstra-derived travel cost enters the routing process.
    """
    work = ensure_preview_node_ids(assign_df)
    if work.empty:
        return {}

    depot_lat = float(work.iloc[0]["depot_lat"])
    depot_lon = float(work.iloc[0]["depot_lon"])

    max_spread_km = float(
        work.apply(
            lambda r: haversine_km(
                depot_lat, depot_lon, float(r["customer_lat"]), float(r["customer_lon"])
            ),
            axis=1,
        ).max()
    )

    print(f"max_spread_km before OSM check: {max_spread_km:.2f}")

    # If preview is too geographically spread out, skip OSM and use proxy only
    if max_spread_km > osm_threshold_km:
        points_df = build_preview_points(work)
        point_ids = points_df["point_id"].astype(str).tolist()
        matrix: Dict[str, Dict[str, float]] = {k: {} for k in point_ids}

        coord_lookup = {
            str(r["point_id"]): (float(r["lat"]), float(r["lon"]))
            for _, r in points_df.iterrows()
        }

        for i, a in enumerate(point_ids):
            a_lat, a_lon = coord_lookup[a]
            for j, b in enumerate(point_ids):
                if i == j:
                    matrix[a][b] = 0.0
                elif b in matrix and a in matrix[b]:
                    matrix[a][b] = matrix[b][a]
                else:
                    b_lat, b_lon = coord_lookup[b]
                    matrix[a][b] = road_adjusted_km(a_lat, a_lon, b_lat, b_lon)

        print(
            f"Preview spread too large for OSM ({max_spread_km:.2f} km > {osm_threshold_km:.2f} km). Using proxy matrix."
        )
        return matrix

    points_df = build_preview_points(work)

    try:
        G_proj = load_or_build_osm_preview_graph(points_df)
        snapped = snap_preview_points_to_osm(points_df, G_proj)
    except Exception:
        G_proj = None
        snapped = points_df.copy()
        snapped["osm_node"] = np.nan

    point_ids = snapped["point_id"].astype(str).tolist()
    matrix: Dict[str, Dict[str, float]] = {k: {} for k in point_ids}

    node_lookup = {str(r["point_id"]): r["osm_node"] for _, r in snapped.iterrows()}
    coord_lookup = {
        str(r["point_id"]): (float(r["lat"]), float(r["lon"]))
        for _, r in snapped.iterrows()
    }

    dijkstra_cache: Dict[Any, Dict[Any, float]] = {}

    for a in point_ids:
        a_node = node_lookup.get(a)

        if pd.notna(a_node) and G_proj is not None:
            try:
                dijkstra_cache[a_node] = nx.single_source_dijkstra_path_length(
                    G_proj,
                    source=a_node,
                    weight="length",
                )
            except Exception:
                dijkstra_cache[a_node] = {}

    for i, a in enumerate(point_ids):
        a_lat, a_lon = coord_lookup[a]
        a_node = node_lookup.get(a)

        for j, b in enumerate(point_ids):
            if i == j:
                matrix[a][b] = 0.0
                continue

            if b in matrix and a in matrix[b]:
                matrix[a][b] = matrix[b][a]
                continue

            b_lat, b_lon = coord_lookup[b]
            b_node = node_lookup.get(b)

            dist_km: Optional[float] = None

            if G_proj is not None and pd.notna(a_node) and pd.notna(b_node):
                dist_m = dijkstra_cache.get(a_node, {}).get(b_node)
                if dist_m is not None:
                    dist_km = float(dist_m) / 1000.0

            if dist_km is None:
                dist_km = road_adjusted_km(a_lat, a_lon, b_lat, b_lon)

            matrix[a][b] = dist_km

    return matrix


def load_or_build_osm_preview_graphs(points_df: pd.DataFrame):
    """
    Return both:
    - G_raw: unprojected graph for lat/lon geometry extraction
    - G_proj: projected graph for nearest-node snapping
    """
    if points_df.empty:
        return None, None

    depot_rows = points_df[points_df["kind"] == "depot"].copy()
    if depot_rows.empty:
        return None, None

    depot_lat = float(depot_rows.iloc[0]["lat"])
    depot_lon = float(depot_rows.iloc[0]["lon"])

    max_preview_km = 0.0
    for row in points_df.itertuples(index=False):
        d = haversine_km(depot_lat, depot_lon, float(row.lat), float(row.lon))
        max_preview_km = max(max_preview_km, d)

    graph_radius_km = min(max(3.0, max_preview_km * 1.15), 20.0)
    graph_radius_m = int(graph_radius_km * 1000.0)

    cache_name = (
        f"osm_point_{depot_lat:.5f}_{depot_lon:.5f}_{graph_radius_m}m.graphml".replace(
            "-", "m"
        )
    )
    cache_path = OSM_CACHE_DIR / cache_name

    if cache_path.exists():
        G_raw = ox.load_graphml(cache_path)
    else:
        G_raw = ox.graph_from_point(
            (depot_lat, depot_lon),
            dist=graph_radius_m,
            network_type="drive",
            simplify=True,
        )
        ox.save_graphml(G_raw, cache_path)

    G_proj = ox.project_graph(G_raw)
    return G_raw, G_proj


def build_snapped_point_lookup(points_df: pd.DataFrame, G_proj) -> Dict[str, Any]:
    snapped = snap_preview_points_to_osm(points_df, G_proj)
    return {str(r["point_id"]): r["osm_node"] for _, r in snapped.iterrows()}


def path_coords_from_osm(
    G_raw,
    node_path: List[Any],
) -> List[Dict[str, float]]:
    coords: List[Dict[str, float]] = []

    for idx, node_id in enumerate(node_path):
        node_data = G_raw.nodes[node_id]
        point = {"lat": float(node_data["y"]), "lon": float(node_data["x"])}

        if idx == 0 or coords[-1] != point:
            coords.append(point)

    return coords


def build_display_leg_path(
    start_point_id: str,
    end_point_id: str,
    coord_lookup: Dict[str, Tuple[float, float]],
    node_lookup: Dict[str, Any],
    G_raw,
) -> List[Dict[str, float]]:
    start_lat, start_lon = coord_lookup[start_point_id]
    end_lat, end_lon = coord_lookup[end_point_id]

    start_node = node_lookup.get(start_point_id)
    end_node = node_lookup.get(end_point_id)

    if G_raw is not None and pd.notna(start_node) and pd.notna(end_node):
        try:
            node_path = nx.shortest_path(
                G_raw,
                source=start_node,
                target=end_node,
                weight="length",
            )
            coords = path_coords_from_osm(G_raw, node_path)
            if len(coords) >= 2:
                return coords
        except Exception:
            pass

    return [
        {"lat": float(start_lat), "lon": float(start_lon)},
        {"lat": float(end_lat), "lon": float(end_lon)},
    ]


def attach_route_display_geometry(
    routes: List[Dict[str, Any]],
    assign_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Attaches visual route geometry for frontend map rendering.

    Purpose:
    - Adds legPath and returnPath coordinates to each route.
    - Allows the React Leaflet map to draw route lines more clearly.

    Notes:
    - If OSM path geometry is unavailable, the frontend can still display
      fallback straight-line route segments.
    """
    work = ensure_preview_node_ids(assign_df)
    if work.empty:
        return routes

    points_df = build_preview_points(work)
    coord_lookup = {
        str(r["point_id"]): (float(r["lat"]), float(r["lon"]))
        for _, r in points_df.iterrows()
    }

    try:
        G_raw, G_proj = load_or_build_osm_preview_graphs(points_df)
        node_lookup = build_snapped_point_lookup(points_df, G_proj)
    except Exception:
        G_raw, G_proj = None, None
        node_lookup = {pid: np.nan for pid in coord_lookup.keys()}

    for route in routes:
        prev_point_id = "DEPOT"

        for stop in route.get("stops", []):
            stop_point_id = str(stop["nodeId"])
            stop["legPath"] = build_display_leg_path(
                prev_point_id,
                stop_point_id,
                coord_lookup,
                node_lookup,
                G_raw,
            )
            prev_point_id = stop_point_id

        if route.get("stops"):
            route["returnPath"] = build_display_leg_path(
                prev_point_id,
                "DEPOT",
                coord_lookup,
                node_lookup,
                G_raw,
            )
        else:
            route["returnPath"] = []

    return routes

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

def preview_matrix_stats(assign_df: pd.DataFrame) -> Dict[str, Any]:
    work = ensure_preview_node_ids(assign_df)
    if work.empty:
        return {
            "previewPoints": 0,
            "matrixPairs": 0,
        }

    unique_points = 1 + int(work["node_id"].nunique())  # depot + customers
    return {
        "previewPoints": unique_points,
        "matrixPairs": unique_points * unique_points,
    }

def matrix_cost(matrix: Dict[str, Dict[str, float]], a: str, b: str) -> float:
    if a == b:
        return 0.0
    return float(matrix.get(a, {}).get(b, 0.0))
