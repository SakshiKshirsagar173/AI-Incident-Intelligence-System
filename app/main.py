"""
Phase 6-7: FastAPI application

Endpoints:
  GET  /health   - liveness + model-loaded check
  POST /predict  - predict Priority (if not given) and Resolution_Time_Hours

Run locally:
  uvicorn main:app --reload --app-dir app
"""
import logging
import sys

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from model import IncidentPredictor
from schemas import HealthResponse, IncidentRequest, IncidentResponse

# --- Logging setup (Phase 7) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("incident_intelligence")

app = FastAPI(
    title="AI Incident Intelligence System",
    description=(
        "Predicts incident Priority and estimates Resolution_Time_Hours from "
        "ticket-intake fields. NOTE: trained on a dataset with weak feature-target "
        "correlation -- see /health and training_summary*.json for honest metrics."
    ),
    version="0.1.0",
)

predictor: IncidentPredictor | None = None


@app.on_event("startup")
def load_model():
    global predictor
    try:
        predictor = IncidentPredictor()
        logger.info("Predictor initialized and ready.")
    except Exception:
        logger.exception("Failed to load model artifacts on startup.")
        predictor = None


@app.get("/health", response_model=HealthResponse)
def health():
    ready = predictor is not None and predictor.is_ready
    return HealthResponse(
        status="ok" if ready else "degraded",
        priority_model_loaded=predictor is not None and predictor.priority_model is not None,
        resolution_time_model_loaded=predictor is not None and predictor.resolution_model is not None,
    )


@app.post("/predict", response_model=IncidentResponse)
def predict(request: IncidentRequest):
    if predictor is None or not predictor.is_ready:
        logger.error("Predict called but model artifacts are not loaded.")
        raise HTTPException(status_code=503, detail="Model is not ready. Check /health.")

    logger.info(
        "Predict request: type=%s dept=%s location=%s priority_given=%s",
        request.incident_type.value,
        request.assigned_department.value,
        request.location.value,
        request.priority.value if request.priority else None,
    )

    try:
        result = predictor.predict(
            incident_type=request.incident_type.value,
            assigned_department=request.assigned_department.value,
            location=request.location.value,
            reported_time=request.reported_time,
            priority=request.priority.value if request.priority else None,
        )
    except Exception:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail="Internal error while generating prediction.")

    logger.info("Predict result: %s", result)
    return IncidentResponse(**result)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."},
    )
