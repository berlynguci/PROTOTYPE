from typing import Any, Dict

# ============================================================
# SECTION 1: FastAPI setup and runtime storage
# Purpose:
# - Initializes the backend API used by the React frontend.
# - Enables CORS so the frontend can call the backend locally.
# - Stores uploaded datasets and generated algorithm runs in memory.
#
# Note:
# - DATASETS and RUNS are runtime dictionaries used for prototype/demo
#   execution. They are not permanent database storage.
# ============================================================

DATASETS: Dict[str, Dict[str, Any]] = {}
RUNS: Dict[str, Dict[str, Any]] = {}