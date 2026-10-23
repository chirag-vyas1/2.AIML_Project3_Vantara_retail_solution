
import shap
from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.routers.metadata import router as metadata_router
from api.routers.predict import router as prediction_router
from src.models.model_persistance import get_project_root
from src.models.model_persistance import load_model_artifacts
from src.utils.config_loader import load_config
from api.routers.explainability import router as explainability_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load configuration and model when the API starts."""

    config = load_config()

    model_bundle = load_model_artifacts(
        config=config,
        get_project_root=get_project_root,
        model_key="xgboost",
        model_type="classification",
    )

    app.state.model_bundle = model_bundle

    # Initialize the explainer once and reuse it for API requests.
    app.state.shap_explainer = shap.TreeExplainer(model_bundle["model"])

    yield

app = FastAPI(
    title="Vantara Customer Behaviour Prediction API",
    description="API for customer churn prediction.",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Vantara Customer Behaviour Prediction API",
    }


app.include_router(metadata_router)
app.include_router(prediction_router)
app.include_router(explainability_router)
