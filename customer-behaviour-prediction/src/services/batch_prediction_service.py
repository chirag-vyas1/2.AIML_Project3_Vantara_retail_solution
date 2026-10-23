
from pathlib import Path
import pandas as pd


def predict_batch_from_csv(
    csv_path: str | Path,
    model_bundle: dict,
    output_path: str | Path,
) -> pd.DataFrame:
    """
    Load a CSV of customer features, predict churn, and save a report.

    The input CSV must contain the same 27 features used by the model.
    """

    csv_path = Path(csv_path)
    output_path = Path(output_path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    # Load the customer data.
    input_df = pd.read_csv(csv_path)

    if input_df.empty:
        raise ValueError("The uploaded CSV contains no customer records.")

    # Get the model's expected feature columns.
    feature_names = model_bundle["feature_names"]
    model = model_bundle["model"]

    # Check required columns before prediction.
    missing_columns = [
        name for name in feature_names
        if name not in input_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "CSV is missing required model features: "
            + ", ".join(missing_columns)
        )

    # Select columns in exactly the order expected by the model.
    X = input_df.loc[:, feature_names].copy()

    # Validate numeric values.
    for column in feature_names:
        X[column] = pd.to_numeric(X[column], errors="raise")

    if X.isna().any().any():
        raise ValueError(
            "The CSV contains missing values in model feature columns."
        )

    if not X.map(lambda value: pd.notna(value) and abs(value) != float("inf")).all().all():
        raise ValueError(
            "The CSV contains infinite or invalid feature values."
        )

    # Generate churn predictions.
    predicted_classes = model.predict(X).astype(int)
    probabilities = model.predict_proba(X)
    class_index = list(model.classes_).index(1)
    churn_probabilities = probabilities[:, class_index]

    # Preserve original columns and append prediction results.
    report_df = input_df.copy()
    report_df["Predicted_Class"] = predicted_classes
    report_df["Churn_Probability"] = churn_probabilities
    report_df["Model_Name"] = model_bundle["model_name"]

    # Create the output directory if required.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the downloadable CSV report.
    report_df.to_csv(output_path, index=False)

    return report_df
