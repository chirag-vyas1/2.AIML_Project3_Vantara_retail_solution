
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/metadata")
def get_model_metadata(request: Request):
    """Return metadata for the model loaded by the API."""

    bundle = request.app.state.model_bundle

    return {
        "model_name": bundle["model_name"],
        "model_type": bundle["model_type"],
        "n_features": bundle["n_features"],
        "feature_names": bundle["feature_names"],
    }
