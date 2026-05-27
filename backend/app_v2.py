from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.dataset_routes import router as dataset_router
from routers.baseline_routes import router as baseline_router
from routers.enhanced_routes import router as enhanced_router

app = FastAPI(title="Delivery Prototype Backend", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dataset_router)
app.include_router(baseline_router)
app.include_router(enhanced_router)


@app.get("/api/health")
def health():
    return {"status": "ok"}