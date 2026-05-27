from typing import List, Optional

from pydantic import BaseModel

# ============================================================
# SECTION 3: Request models / input schemas
# Purpose:
# - Defines the expected request body structure for the frontend.
# - Validates dataset field mapping, baseline run parameters,
#   enhanced run parameters, and added-customer payloads.
#
# note:
# - These Pydantic models act as a validation layer before the
#   backend executes routing, reconstruction, or optimization logic.
# ============================================================

class FieldMapping(BaseModel):
    """
    Stores the frontend-to-backend column mapping for uploaded CSV files.

    Used by:
    - Dataset validation endpoint.
    - Dataset reconstruction functions.

    Notes:
    - Required fields identify depot/customer coordinates.
    - Optional fields support order date, ETA, rating, area, and agent ID.
    """
    depot_id: Optional[str] = None
    depot_lat: str
    depot_lon: str
    customer_id: str
    agent_id: Optional[str] = None
    customer_lat: str
    customer_lon: str
    order_id: Optional[str] = None
    order_date_col: Optional[str] = None 
    eta_col: Optional[str] = None
    rating_col: Optional[str] = None
    area_col: Optional[str] = None


class BaselineRequest(BaseModel):
    """
    Stores parameters for the baseline route generation run.

    Used by:
    - /api/runs/baseline endpoint.

    Notes:
    - Controls number of representatives, travel speed, service time,
      random seed, and selected run profile.
    """
    dataset_id: str
    num_representatives: int = 4
    avg_speed_kmph: float = 40.0
    service_minutes_per_stop: float = 8.0
    seed: int = 42
    run_profile: Optional[str] = "default_balanced"


class EnhancedRequest(BaseModel):
    """
    Stores parameters for enhanced DEQ rebalancing.

    Used by:
    - /api/runs/enhanced endpoint.

    Notes:
    - alpha_weight and beta_weight control the priority score formula.
    - max_iterations and border_fraction control the rebalancing search.
    """
    dataset_id: str
    baseline_run_id: str
    alpha_weight: Optional[float] = None
    beta_weight: Optional[float] = None
    max_iterations: Optional[int] = None
    border_fraction: Optional[float] = None
    run_profile: Optional[str] = None


class AddedCustomerPayload(BaseModel):
    """
    Represents one new customer manually added from the frontend map.

    Used by:
    - /api/runs/baseline/add-customers endpoint.

    Notes:
    - The backend assigns the added customer to the nearest suitable
      representative and then reroutes the baseline result.
    """
    label: str
    lat: float
    lon: float
    address: Optional[str] = None
    assigned_rep: Optional[str] = None
    customer_number: Optional[int] = None


class BaselineAddCustomersRequest(BaseModel):
    """
    Groups added customers under an existing baseline run.

    Used by:
    - Add-customer rerouting workflow.

    Notes:
    - baseline_run_id tells the backend which previous baseline result
      should be updated.
    """
    baseline_run_id: str
    customers: List[AddedCustomerPayload]