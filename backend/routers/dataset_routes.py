import io
import json
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, UploadFile, Response
from fastapi.responses import StreamingResponse

from schemas import FieldMapping
from state import DATASETS
from services.dataset_service import (
    autofill_mapping_from_known_columns,
    infer_dataset_role,
    normalize_dataset,
    read_csv_upload,
    role_label,
    validation_summary,
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

@router.post("/api/datasets/validate")
async def validate_dataset(
    file: UploadFile = File(...),
    mapping_json: str = Form(...),
    dataset_role: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """
    Validates and reconstructs an uploaded dataset.

    Purpose:
    - Reads the uploaded CSV file.
    - Applies frontend field mapping.
    - Infers or accepts dataset role.
    - Normalizes the file into the route-ready schema.
    - Stores the cleaned dataset in memory for later routing.

    Used by:
    - Dataset Upload page.
    """
    try:
        mapping = FieldMapping(**json.loads(mapping_json))
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid mapping JSON: {exc}"
        ) from exc

    resolved_role = dataset_role or infer_dataset_role(file.filename or "")
    df = read_csv_upload(file)

    mapping = autofill_mapping_from_known_columns(df, mapping, resolved_role)

    normalized = normalize_dataset(df, mapping, resolved_role)

    print("validate dataset role:", resolved_role)
    print("normalized rows:", len(normalized))
    print("effective mapping:", mapping.model_dump())
    if "order_date" in normalized.columns:
        print(
            "normalized unique order_date sample:",
            sorted(normalized["order_date"].dropna().astype(str).unique())[:10],
        )
    if "agent_id" in normalized.columns:
        print(
            "normalized distinct agent_id:",
            int(normalized["agent_id"].astype(str).nunique()),
        )
    if "depot_id" in normalized.columns:
        print(
            "normalized distinct depot_id:",
            int(normalized["depot_id"].astype(str).nunique()),
        )

    summary = validation_summary(normalized)

    dataset_id = str(uuid.uuid4())
    reconstructed_name = (
        f"reconstructed_{(file.filename or 'dataset').replace('.csv', '')}.csv"
    )

    DATASETS[dataset_id] = {
        "data": normalized,
        "mapping": mapping.model_dump(),
        "filename": file.filename,
        "datasetRole": resolved_role,
        "sourceLabel": role_label(resolved_role),
        "reconstructedBaselineName": reconstructed_name,
    }

    return {
        "datasetId": dataset_id,
        "datasetRole": resolved_role,
        "sourceLabel": role_label(resolved_role),
        "reconstructedBaselineReady": True,
        "reconstructedBaselineName": reconstructed_name,
        **summary,
    }

@router.get("/api/datasets/{dataset_id}/meta")
def dataset_meta(dataset_id: str) -> Dict[str, Any]:
    """
    Returns dataset metadata needed by the frontend.

    Purpose:
    - Provides dataset role, source label, depot information,
      record count, customer count, and order count.
    """
    payload = DATASETS.get(dataset_id)
    if not payload:
        return Response(
            content="Dataset not found", status_code=404, media_type="text/plain"
        )

    df = payload["data"]

    depot_row = df.iloc[0]
    depot = {
        "id": str(depot_row["depot_id"]),
        "lat": float(depot_row["depot_lat"]),
        "lon": float(depot_row["depot_lon"]),
        "name": str(depot_row["depot_id"]),
    }

    return {
        "datasetId": dataset_id,
        "filename": payload["filename"],
        "datasetRole": payload["datasetRole"],
        "sourceLabel": payload["sourceLabel"],
        "reconstructedBaselineName": payload["reconstructedBaselineName"],
        "records": int(len(df)),
        "depots": int(df["depot_id"].nunique()),
        "customers": int(df["customer_id"].nunique()),
        "customerNodes": (
            int(df["customer_node_id"].nunique())
            if "customer_node_id" in df.columns
            else int(df["customer_id"].nunique())
        ),
        "orders": int(df["order_id"].nunique()),
        "depot": depot,
    }

@router.get("/api/datasets/{dataset_id}/reconstructed")
def download_reconstructed_dataset(dataset_id: str):
    """
    Downloads the reconstructed route-ready dataset as CSV.

    Purpose:
    - Allows users to inspect or save the normalized dataset produced
      after validation.
    """
    payload = DATASETS.get(dataset_id)
    if not payload:
        return Response(
            content="Dataset not found", status_code=404, media_type="text/plain"
        )

    df = payload["data"].copy()
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={payload['reconstructedBaselineName']}"
        },
    )

