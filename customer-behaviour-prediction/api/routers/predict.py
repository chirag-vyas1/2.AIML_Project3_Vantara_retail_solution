
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Request
from fastapi import UploadFile
from fastapi import File
from pydantic import BaseModel, FiniteFloat
from api.schemas.prediction import CustomerFeatures
from fastapi.responses import StreamingResponse
from io import BytesIO

from src.utils.prediction_repository import get_customer_features
from src.utils.prediction_repository import save_prediction
from src.services.batch_prediction_service import predict_batch_from_csv

router = APIRouter()


class PredictionResponse(BaseModel):
    model_name: str
    predicted_class: int
    churn_probability: FiniteFloat


@router.post("/predict", response_model=PredictionResponse)
def predict_customer(
    customer: CustomerFeatures,
    request: Request,
):
    """Predict churn for one customer."""

    bundle = request.app.state.model_bundle
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    # Convert the validated request to a DataFrame
    customer_data = customer.model_dump()

    # Match the exact feature order used by the trained model
    X_customer = pd.DataFrame(
        [[customer_data[name] for name in feature_names]],
        columns=feature_names,
    )

    try:
        predicted_class = int(model.predict(X_customer)[0])
        probabilities = model.predict_proba(X_customer)[0]

        # Locate class 1 rather than assuming its position
        class_index = list(model.classes_).index(1)
        churn_probability = float(probabilities[class_index])

    except (ValueError, IndexError, KeyError) as exc:
        raise HTTPException(
            status_code=500,
            detail="The model could not generate a prediction.",
        ) from exc

    return PredictionResponse(
        model_name=bundle["model_name"],
        predicted_class=predicted_class,
        churn_probability=churn_probability,
    )

class BatchPredictionResponse(BaseModel):
    model_name: str
    total_customers: int
    predictions: list[PredictionResponse]

@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
)
def predict_batch(
    customers: list[CustomerFeatures],
    request: Request,
):
    if not customers:
        raise HTTPException(
            status_code=422,
            detail="At least one customer must be provided.",
        )

    bundle = request.app.state.model_bundle
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    # Convert all customers into one DataFrame.
    customer_records = [
        customer.model_dump() for customer in customers
    ]

    X_customers = pd.DataFrame(
        customer_records,
        columns=feature_names,
    )

    try:
        # Predict all customers in one batch.
        predicted_classes = model.predict(X_customers)
        probabilities = model.predict_proba(X_customers)

        # Find the probability column corresponding to class 1.
        class_index = list(model.classes_).index(1)

        results = [
            PredictionResponse(
                model_name=bundle["model_name"],
                predicted_class=int(predicted_classes[i]),
                churn_probability=float(
                    probabilities[i][class_index]
                ),
            )
            for i in range(len(customers))
        ]

    except (ValueError, IndexError, KeyError) as exc:
        raise HTTPException(
            status_code=500,
            detail="The model could not generate batch predictions.",
        ) from exc

    return BatchPredictionResponse(
        model_name=bundle["model_name"],
        total_customers=len(customers),
        predictions=results,
    )



@router.post(
    "/predict/customer/{customer_id}",
    response_model=PredictionResponse,
)
def predict_existing_customer(customer_id: int, request: Request):
    """Predict churn for an existing customer using stored database features."""

    if customer_id <= 0:
        raise HTTPException(
            status_code=400,
            detail="Customer ID must be a positive integer.",
        )

    try:
        # Step 1: Fetch the customer's features from PostgreSQL.
        customer_data = get_customer_features(customer_id)

        if customer_data is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Customer features not found in PostgreSQL "
                    "for the requested customer ID."
                ),
            )

        # Step 2: Get the already-loaded model.
        bundle = request.app.state.model_bundle
        model = bundle["model"]
        feature_names = bundle["feature_names"]

        # Step 3: Prepare the features in the model's expected order.
        X_customer = pd.DataFrame(
            [[customer_data[name] for name in feature_names]],
            columns=feature_names,
        )

        # Step 4: Generate the churn prediction.
        predicted_class = int(model.predict(X_customer)[0])

        probabilities = model.predict_proba(X_customer)[0]
        class_index = list(model.classes_).index(1)
        churn_probability = float(probabilities[class_index])

        # Step 5: Save the prediction in PostgreSQL.
        save_prediction(
            customer_id=customer_id,
            model_name=bundle["model_name"],
            predicted_class=predicted_class,
            churn_probability=churn_probability,
        )

        # Step 6: Return the prediction.
        return PredictionResponse(
            model_name=bundle["model_name"],
            predicted_class=predicted_class,
            churn_probability=churn_probability,
        )

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Customer prediction or database persistence failed.",
        ) from exc




@router.post("/predict/batch/csv")
async def predict_batch_csv(
    request: Request,
    file: UploadFile = File(...),
):
    """Upload a CSV of customer features and download churn predictions."""

    if not file.filename or Path(file.filename).suffix.lower() != ".csv":
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file.",
        )

    try:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "customer_features.csv"
            output_path = temp_path / "churn_predictions.csv"

            # Save the uploaded CSV to a temporary file.
            with input_path.open("wb") as destination:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    destination.write(chunk)

            # Generate predictions using the existing loaded model.
            predict_batch_from_csv(
                csv_path=input_path,
                model_bundle=request.app.state.model_bundle,
                output_path=output_path,
            )

            # Read the report before the temporary directory is removed.
            csv_content = output_path.read_bytes()

        return StreamingResponse(
            BytesIO(csv_content),
            media_type="text/csv",
            headers={
                "Content-Disposition":
                    'attachment; filename="churn_predictions.csv"'
            },
        )

    except (ValueError, FileNotFoundError, pd.errors.ParserError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="CSV batch prediction failed.",
        ) from exc

    finally:
        await file.close()
