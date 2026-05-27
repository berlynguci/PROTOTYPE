import pandas as pd

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

def ensure_preview_node_ids(assign_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures every row has a node_id used by the distance matrix.

    Purpose:
    - Uses customer_node_id when available.
    - Creates fallback customer IDs when node_id is missing.

    Used by:
    - Distance matrix generation.
    - Route construction.
    - Map geometry generation.
    """
    work = assign_df.copy().reset_index(drop=True)

    if "customer_node_id" in work.columns:
        work["node_id"] = work["customer_node_id"].astype(str)
    elif "node_id" not in work.columns:
        work["node_id"] = [f"CUST-{i+1}" for i in range(len(work))]

    return work