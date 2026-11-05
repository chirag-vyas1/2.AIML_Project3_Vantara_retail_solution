"""API tests for the Vantara Retail Customer Behaviour Prediction service.

Run from the customer-behaviour-prediction project root:
    python -m pytest .\\tests\\test_api.py -v

The prediction tests use a deterministic fake model and do not require a
trained model artifact or a PostgreSQL connection for those endpoints.
"""

import warnings


warnings.filterwarnings(
    "ignore",
    message=r"Using `httpx` with `starlette\.testclient` is deprecated.*",
    category=Warning,
    module=r"fastapi\.testclient",
)


warnings.filterwarnings(
    "ignore",
    message=r"The set_(bad|over|under) function will be deprecated.*",
    category=PendingDeprecationWarning,
    module=r"shap\.plots\.colors\._colors",
)

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.schemas.prediction import CustomerFeatures


class DummyChurnModel:
    """Deterministic model substitute for API tests."""

    classes_ = np.array([0, 1])

    def predict(self, features):
        """Return one deterministic class per input row."""
        return np.array([1] * len(features))

    def predict_proba(self, features):
        """Return valid probabilities for classes 0 and 1."""
        return np.tile(np.array([0.25, 0.75]), (len(features), 1))


@pytest.fixture
def api_client():
    """Provide a TestClient configured with a fake model bundle."""
    feature_names = list(CustomerFeatures.model_fields.keys())
    app.state.model_bundle = {
    "model": DummyChurnModel(),
    "model_name": "test_model",
    "model_type": "classification",
    "feature_names": feature_names,
    "n_features": len(feature_names),
     }
    return TestClient(app)



@pytest.fixture
def valid_customer_payload():
    payload = {}

    for field_name, field in CustomerFeatures.model_fields.items():
        if field.default is not None and not field.is_required():
            payload[field_name] = field.default
        elif field_name == "Frequency":
            payload[field_name] = 0
        else:
            payload[field_name] = 0.0

    return payload



def test_api_endpoints(api_client, valid_customer_payload):
    """Test metadata, single/batch predictions, and request validation."""

    # 1. Metadata endpoint responds successfully with JSON.
    metadata_response = api_client.get("/metadata")
    assert metadata_response.status_code == 200
    assert isinstance(metadata_response.json(), dict)

    # 2. Single-customer prediction response.
    single_response = api_client.post("/predict", json=valid_customer_payload)

    if single_response.status_code != 200:
        print("Prediction validation error:", single_response.json())
        print("Submitted payload:", valid_customer_payload)

    assert single_response.status_code == 200
    single_data = single_response.json()
    assert single_data["model_name"] == "test_model"
    assert single_data["predicted_class"] == 1
    assert single_data["churn_probability"] == pytest.approx(0.75)

    # 3. Batch prediction returns one result per customer.
    batch_response = api_client.post(
        "/predict/batch",
        json=[valid_customer_payload, valid_customer_payload],
    )
    assert batch_response.status_code == 200
    batch_data = batch_response.json()
    assert batch_data["model_name"] == "test_model"
    assert batch_data["total_customers"] == 2
    assert len(batch_data["predictions"]) == 2
    assert all(
        prediction["churn_probability"] == pytest.approx(0.75)
        for prediction in batch_data["predictions"]
    )

    # 4. Empty batch must be rejected.
    empty_batch_response = api_client.post("/predict/batch", json=[])
    assert empty_batch_response.status_code == 422

    # 5. Missing required fields must be rejected.
    invalid_customer_response = api_client.post("/predict", json={})
    assert invalid_customer_response.status_code == 422

    # 6. Churn probability must be within the valid probability range.
    assert 0.0 <= single_data["churn_probability"] <= 1.0
