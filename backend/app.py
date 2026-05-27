from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from routers.baseline_routes import router as baseline_router
from routers.dataset_routes import router as dataset_router
from routers.enhanced_routes import router as enhanced_router

STATIC_DIR = Path(__file__).parent / "static"

app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

@app.get("/")
def serve_frontend():
    return FileResponse(STATIC_DIR / "index.html")

# ============================================================
# FastAPI setup
# Purpose:
# - Initializes the backend API used by the React frontend.
# - Enables CORS so the frontend can call the backend locally.
# - Registers dataset, baseline, and enhanced route modules.
#
# Note:
# - Runtime storage is now in state.py.
# - Request schemas are now in schemas.py.
# - Algorithm logic is now inside services/.
# - API endpoints are now inside routers/.
# ============================================================

app = FastAPI(title="Delivery Prototype Backend", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "Backend is running"}

@app.get("/api/health")
def health():
    """
    Simple backend health-check endpoint.

    Purpose:
    - Lets the frontend verify that the FastAPI server is reachable.
    """
    return {"status": "ok"}


app.include_router(dataset_router)
app.include_router(baseline_router)
app.include_router(enhanced_router)
