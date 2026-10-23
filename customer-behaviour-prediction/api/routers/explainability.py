import numpy as np
import pandas as pd

import shap
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from api.schemas.prediction import CustomerFeatures


router = APIRouter(
    prefix="/explain/shap",
    tags=["SHAP Explainability"],
)


class GlobalShapRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(
        min_length=1,
        max_length=500,
    )
    max_features: int = Field(default=15, ge=1, le=27)


def get_shap_values(explanation, model):
    """Return a 2D SHAP array: customers x features."""

    values = np.asarray(explanation.values)

    # Some classifier explainers return:
    # customers x features x classes.
    if values.ndim == 3:
        classes = list(model.classes_)

        if 1 not in classes:
            raise ValueError(
                "The model does not contain churn class 1."
            )

        class_index = classes.index(1)
        values = values[:, :, class_index]

    if values.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP output shape: {values.shape}"
        )

    return values


@router.post("/customer")
def explain_customer(
    customer: CustomerFeatures,
    request: Request,
):
    """Return an individual churn prediction and SHAP contributions."""

    bundle = request.app.state.model_bundle
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    explainer = request.app.state.shap_explainer

    customer_data = customer.model_dump()

    X_customer = pd.DataFrame(
        [[customer_data[name] for name in feature_names]],
        columns=feature_names,
    )

    try:
        prediction = int(model.predict(X_customer)[0])
        probabilities = model.predict_proba(X_customer)[0]

        classes = list(model.classes_)
        if 1 not in classes:
            raise ValueError("Churn class 1 is not available.")

        churn_index = classes.index(1)
        churn_probability = float(probabilities[churn_index])

        explanation = explainer(X_customer)
        shap_values = get_shap_values(explanation, model)

        base_values = np.asarray(explanation.base_values)

        if base_values.ndim == 0:
            base_value = float(base_values)
        elif base_values.ndim == 1:
            base_value = float(base_values[0])
        else:
            base_value = float(base_values[0, 0])

        contributions = pd.DataFrame({
            "feature": feature_names,
            "value": X_customer.iloc[0].to_numpy(dtype=float),
            "shap_value": shap_values[0],
        })

        contributions["absolute_shap"] = (
            contributions["shap_value"].abs()
        )

        contributions = contributions.sort_values(
            "absolute_shap",
            ascending=False,
        )

        return {
            "model_name": bundle["model_name"],
            "predicted_class": prediction,
            "prediction_label": ("Churn" if prediction == 1 else "No Churn"),
            "churn_probability": churn_probability,
            "contributions": contributions.to_dict(orient="records"),
            "base_value": base_value,
            "feature_names": feature_names,
            "feature_values": X_customer.iloc[0].to_numpy(dtype=float).tolist(),
            "shap_values": shap_values[0].tolist(),
        }

    except (ValueError, IndexError, KeyError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to generate the customer SHAP explanation.",
        ) from exc

@router.post("/global")
def explain_global(
    payload: GlobalShapRequest,
    request: Request,
):
    """Return global mean absolute SHAP feature importance."""

    bundle = request.app.state.model_bundle
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    explainer = request.app.state.shap_explainer

    records = [
        customer.model_dump()
        for customer in payload.customers
    ]

    X = pd.DataFrame(records, columns=feature_names)

    try:
        explanation = explainer(X)
        shap_values = get_shap_values(explanation, model)

        importance = pd.DataFrame({
            "feature": feature_names,
            "mean_absolute_shap": np.abs(
                shap_values
            ).mean(axis=0),
        })

        importance = importance.sort_values(
            "mean_absolute_shap",
            ascending=False,
        ).head(payload.max_features)

        return {
            "model_name": bundle["model_name"],
            "customers_analysed": len(X),
            "feature_names": feature_names,
            "feature_values": X.to_numpy(dtype=float).tolist(),
            "shap_values": shap_values.tolist(),
            "importance": importance.to_dict(orient="records"),

        }

    except (ValueError, IndexError, KeyError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to generate global SHAP insights.",
        ) from exc
