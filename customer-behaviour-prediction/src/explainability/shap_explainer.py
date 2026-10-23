
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap



def run_shap_analysis(
    model,
    explainer,
    X_validation: pd.DataFrame,
    feature_names,
    y_validation=None,
    customer_idx=None,
) -> dict:
    """
    Validate features, calculate SHAP values, predict churn,
    and display selected customer details in a readable format.

    Returns
    -------
    dict
        Validated features, SHAP values, predictions,
        and selected customer details.
    """

    # 1. Validate input
    if not isinstance(X_validation, pd.DataFrame):
        raise TypeError("X_validation must be a DataFrame.")

    if X_validation.empty:
        raise ValueError("X_validation is empty.")

    expected_features = list(feature_names)

    if len(expected_features) != len(set(expected_features)):
        raise ValueError("Expected feature names contain duplicates.")

    missing_features = [
        feature for feature in expected_features
        if feature not in X_validation.columns
    ]

    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    # Keep features in the exact model order
    X_explain = X_validation[expected_features].copy()

    if X_explain.isna().any().any():
        raise ValueError("Validation features contain missing values.")

    try:
        feature_array = X_explain.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Validation features must contain numeric values."
        ) from exc

    if not np.isfinite(feature_array).all():
        raise ValueError(
            "Validation features contain infinite or invalid values."
        )

    if (
        hasattr(model, "n_features_in_")
        and X_explain.shape[1] != model.n_features_in_
    ):
        raise ValueError(
            f"Model expects {model.n_features_in_} features, "
            f"but received {X_explain.shape[1]}."
        )

    if y_validation is not None:
        if len(y_validation) != len(X_explain):
            raise ValueError(
                "y_validation and X_validation lengths differ."
            )

    print("=" * 65)
    print("SHAP ANALYSIS VALIDATION")
    print("=" * 65)
    print(f"Validation data shape : {X_explain.shape}")
    print(f"Expected feature count: {len(expected_features)}")
    print("Feature validation    : PASSED")

    # 2. Calculate SHAP values
    shap_explanation = explainer(X_explain)

    print("\nSHAP calculation      : COMPLETED")
    print("SHAP values shape     :", shap_explanation.values.shape)

    # 3. Predict all validation customers
    predictions = model.predict(X_explain)

    predicted_churn_count = int(np.sum(predictions == 1))

    print("\nPREDICTION SUMMARY")
    print("-" * 65)
    print(f"Total customers       : {len(predictions)}")
    print(f"Predicted churn       : {predicted_churn_count}")
    print(
        f"Predicted no churn    : "
        f"{int(np.sum(predictions == 0))}"
    )

    # 4. Select a customer
    if customer_idx is None:
        churn_indices = np.flatnonzero(predictions == 1)

        if len(churn_indices) == 0:
            print("\nNo customers were predicted as churn.")

            return {
                "X_explain": X_explain,
                "shap_values": shap_explanation,
                "predictions": predictions,
                "customer_idx": None,
                "customer_explanation": None,
            }

        customer_idx = int(churn_indices[0])

    if isinstance(customer_idx, (bool, np.bool_)) or not isinstance(
        customer_idx, (int, np.integer)
    ):
        raise TypeError("customer_idx must be an integer.")

    if not 0 <= customer_idx < len(X_explain):
        raise IndexError(
            "customer_idx is outside the validation data."
        )

    # 5. Get selected customer's prediction and features
    customer_idx = int(customer_idx)
    customer_row = X_explain.iloc[[customer_idx]]
    customer_prediction = int(predictions[customer_idx])

    prediction_label = (
        "Churn" if customer_prediction == 1 else "No Churn"
    )

    # Convert the row into a readable feature table
    customer_feature_table = pd.DataFrame({
        "Feature": customer_row.columns,
        "Value": customer_row.iloc[0].to_numpy(),
    })

    # 6. Get class probabilities
    class_probabilities = {}

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(customer_row)[0]

        class_probabilities = {
            str(class_label): float(probability)
            for class_label, probability
            in zip(model.classes_, probabilities)
        }

    # 7. Get actual churn label
    actual_churn = None

    if y_validation is not None:
        if isinstance(y_validation, pd.DataFrame):
            if y_validation.shape[1] != 1:
                raise ValueError(
                    "y_validation DataFrame must have one column."
                )
            actual_churn = y_validation.iloc[customer_idx, 0]

        elif isinstance(y_validation, pd.Series):
            actual_churn = y_validation.iloc[customer_idx]

        else:
            actual_churn = y_validation[customer_idx]

        actual_churn = int(actual_churn)

    # 8. Display customer summary
    print("\n" + "=" * 65)
    print(f"SELECTED CUSTOMER: {customer_idx}")
    print("=" * 65)

    print(f"Predicted class : {customer_prediction}")
    print(f"Prediction      : {prediction_label}")

    if class_probabilities:
        print("\nCLASS PROBABILITIES")
        print("-" * 35)

        for class_label, probability in class_probabilities.items():
            label = "Churn" if class_label == "1" else "No Churn"
            print(f"{label:<12}: {probability:.2%}")

    if actual_churn is not None:
        actual_label = (
            "Churn" if actual_churn == 1 else "No Churn"
        )

        print(f"\nActual outcome : {actual_label}")
        print(
            "Prediction correct:",
            bool(customer_prediction == actual_churn),
        )

    print("\nCUSTOMER FEATURE DETAILS")
    print("-" * 65)
    print(
        customer_feature_table.to_string(
            index=False,
            formatters={
                "Value": lambda value: f"{value:,.4f}"
                if isinstance(value, (int, float, np.number))
                else str(value)
            },
        )
    )

    # 9. Prepare structured results for later use
    customer_result = {
        "customer_idx": customer_idx,
        "predicted_class": customer_prediction,
        "prediction_label": prediction_label,
        "features": customer_feature_table,
        "class_probabilities": class_probabilities,
        "actual_churn": actual_churn,
    }

    print("=" * 65)

    return {
        "X_explain": X_explain,
        "shap_values": shap_explanation,
        "predictions": predictions,
        "customer_idx": customer_idx,
        "customer_explanation": customer_result,
    }

def explain_customer_waterfall(
    model,
    explainer,
    X: pd.DataFrame,
    customer_idx: int,
    output_dir: Path,
    max_display: int = 15,
    show: bool = False,
) -> dict:
    """
    Generate and save a SHAP waterfall plot for one customer.

    Parameters
    ----------
    model:
        Fitted classification model.

    explainer:
        Initialized SHAP explainer for the model.

    X:
        DataFrame containing the model's expected input features.

    customer_idx:
        Zero-based positional index of the customer in X.

    output_dir:
        Directory where the plot will be saved.

    max_display:
        Maximum number of features displayed in the plot.

    show:
        Whether to display the plot.

    Returns
    -------
    dict
        Contains the plot path, predicted class, and
        predicted class probabilities when available.
    """

    # 1. Validate inputs
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if X.empty:
        raise ValueError("X is empty.")

    if not isinstance(customer_idx, int):
        raise TypeError("customer_idx must be an integer.")

    if not 0 <= customer_idx < len(X):
        raise IndexError(
            f"customer_idx must be between 0 and {len(X) - 1}."
        )

    if max_display < 1:
        raise ValueError("max_display must be at least 1.")

    # 2. Select one customer
    customer = X.iloc[[customer_idx]].copy()

    if customer.isna().any().any():
        raise ValueError("Customer features contain missing values.")

    if not np.isfinite(customer.to_numpy(dtype=float)).all():
        raise ValueError(
            "Customer features contain infinite or invalid values."
        )

    # 3. Generate SHAP explanation
    explanation = explainer(customer)

    # 4. Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_path = (
        output_dir
        / f"customer_{customer_idx}_shap_waterfall.png"
    )

    # 5. Generate and save waterfall plot
    try:
        shap.plots.waterfall(
            explanation[0],
            max_display=max_display,
            show=False,
        )

        fig = plt.gcf()
        fig.set_size_inches(12, 8)
        fig.tight_layout()

        fig.savefig(
            plot_path,
            dpi=300,
            bbox_inches="tight",
        )

        if show:
            plt.show()

    finally:
        plt.close("all")

    # 6. Get model prediction
    prediction = model.predict(customer)[0]

    result = {
        "customer_idx": customer_idx,
        "predicted_class": prediction,
        "plot_path": plot_path,
        "shap_values": explanation,
    }

    # Include probabilities for classifiers that support them
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(customer)[0]

        result["class_probabilities"] = {
            str(class_label): float(probability)
            for class_label, probability
            in zip(model.classes_, probabilities)
        }

    print(f"SHAP waterfall saved to: {plot_path}")
    print(f"Predicted class: {prediction}")

    if "class_probabilities" in result:
        print("Class probabilities:", result["class_probabilities"])

    return result

def explain_global_summary(
    explainer,
    X: pd.DataFrame,
    output_dir: Path,
    max_display: int = 15,
    plot_type: str = "dot",
    show: bool = False,
) -> dict:
    """
    Generate and save a global SHAP summary plot.

    Parameters
    ----------
    explainer:
        Initialized SHAP explainer for the trained model.

    X:
        DataFrame containing the model's input features.

    output_dir:
        Directory where plots will be saved.

    max_display:
        Maximum number of features displayed.

    plot_type:
        "dot" for the SHAP beeswarm summary plot,
        or "bar" for mean absolute SHAP importance.

    show:
        Whether to display the plot.

    Returns
    -------
    dict
        Contains the saved plot path and SHAP values.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if X.empty:
        raise ValueError("X is empty.")

    if plot_type not in {"dot", "bar"}:
        raise ValueError(
            "plot_type must be either 'dot' or 'bar'."
        )

    if max_display < 1:
        raise ValueError("max_display must be at least 1.")

    if X.isna().any().any():
        raise ValueError("X contains missing values.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("X contains infinite or invalid values.")

    # Calculate SHAP values for all supplied customers
    shap_values = explainer(X)

    # Prepare output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        "shap_summary_dot.png"
        if plot_type == "dot"
        else "shap_summary_bar.png"
    )
    plot_path = output_dir / filename

    try:
        if plot_type == "dot":
            shap.plots.beeswarm(
                shap_values,
                max_display=max_display,
                show=False,
            )
        else:
            shap.plots.bar(
                shap_values,
                max_display=max_display,
                show=False,
            )

        fig = plt.gcf()
        fig.set_size_inches(12, 8)
        fig.tight_layout()

        fig.savefig(
            plot_path,
            dpi=300,
            bbox_inches="tight",
        )

        if show:
            plt.show()

    finally:
        plt.close("all")

    print(f"SHAP summary plot saved to: {plot_path}")

    return {
        "plot_path": plot_path,
        "shap_values": shap_values,
        "plot_type": plot_type,
    }

def plot_feature_importance(
    model,
    feature_names,
    output_dir: Path,
    top_n: int = 10,
    shap_values=None,
    show: bool = False,
) -> dict:
    """
    Create a feature-importance table and bar chart.

    If shap_values are provided, calculate mean absolute SHAP
    importance. Otherwise, use the model's built-in importance.

    Parameters
    ----------
    model:
        Fitted XGBoost model.

    feature_names:
        Names of the model's input features.

    output_dir:
        Directory for saving the table and plot.

    top_n:
        Number of top features to display.

    shap_values:
        Optional SHAP Explanation object or SHAP value array.

    show:
        Whether to display the plot.

    Returns
    -------
    dict
        Importance table, CSV path, and plot path.
    """

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    feature_names = list(feature_names)

    if not feature_names:
        raise ValueError("feature_names cannot be empty.")

    if shap_values is None:
        # XGBoost built-in importance
        importance_values = model.feature_importances_
        importance_column = "XGBoost_Importance"
        filename_prefix = "xgboost_feature_importance"

    else:
        # Actual global SHAP importance: mean absolute SHAP value
        values = (
            shap_values.values
            if isinstance(shap_values, shap.Explanation)
            else np.asarray(shap_values)
        )

        if values.ndim != 2:
            raise ValueError(
                "Expected SHAP values with shape "
                "(samples, features)."
            )

        if values.shape[1] != len(feature_names):
            raise ValueError(
                "SHAP feature count does not match feature_names."
            )

        importance_values = np.abs(values).mean(axis=0)
        importance_column = "Mean_Absolute_SHAP"
        filename_prefix = "shap_feature_importance"

    if len(importance_values) != len(feature_names):
        raise ValueError(
            "Importance count does not match feature_names."
        )

    # Create and sort the importance table
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        importance_column: importance_values,
    })

    importance_df = importance_df.sort_values(
        by=importance_column,
        ascending=False,
    ).reset_index(drop=True)

    top_features = importance_df.head(top_n).copy()

    # Prepare output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{filename_prefix}.csv"
    plot_path = output_dir / f"{filename_prefix}.png"

    # Save the complete sorted table
    importance_df.to_csv(csv_path, index=False)

    # Plot the top features
    plot_data = top_features.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(
        plot_data["Feature"],
        plot_data[importance_column],
    )

    ax.set_xlabel(importance_column.replace("_", " "))
    ax.set_ylabel("Feature")
    ax.set_title(f"Top {min(top_n, len(importance_df))} Features")
    fig.tight_layout()

    try:
        fig.savefig(
            plot_path,
            dpi=300,
            bbox_inches="tight",
        )

        if show:
            plt.show()

    finally:
        plt.close(fig)

    print("\nFeature importance table:")
    print(top_features.to_string(index=False))
    print(f"\nFull table saved to: {csv_path}")
    print(f"Plot saved to: {plot_path}")

    return {
        "importance_table": importance_df,
        "top_features": top_features,
        "csv_path": csv_path,
        "plot_path": plot_path,
        "importance_type": importance_column,
    }


import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


def explain_customer_for_dashboard(
    model,
    explainer,
    customer_features: pd.DataFrame,
    max_display: int = 15,
):
    """
    Explain one customer using SHAP.

    customer_features must contain exactly one row of
    engineered features in the model's expected feature order.

    Returns prediction, probabilities, feature contributions,
    and a Matplotlib figure for a dashboard.
    """

    if not isinstance(customer_features, pd.DataFrame):
        raise TypeError("customer_features must be a DataFrame.")

    if len(customer_features) != 1:
        raise ValueError(
            "Provide exactly one customer row."
        )

    if customer_features.isna().any().any():
        raise ValueError("Customer features contain missing values.")

    if not np.isfinite(
        customer_features.to_numpy(dtype=float)
    ).all():
        raise ValueError("Customer features contain invalid values.")

    # Calculate the SHAP explanation
    explanation = explainer(customer_features)

    # Predict churn
    prediction = int(model.predict(customer_features)[0])

    probabilities = {}
    if hasattr(model, "predict_proba"):
        raw_probabilities = model.predict_proba(
            customer_features
        )[0]

        probabilities = {
            str(label): float(probability)
            for label, probability in zip(
                model.classes_, raw_probabilities
            )
        }

    # Prepare feature contribution table
    contributions = pd.DataFrame({
        "Feature": customer_features.columns,
        "Value": customer_features.iloc[0].values,
        "SHAP_Value": explanation.values[0],
    })

    contributions["Absolute_SHAP"] = (
        contributions["SHAP_Value"].abs()
    )

    contributions = contributions.sort_values(
        "Absolute_SHAP",
        ascending=False,
    ).reset_index(drop=True)

    # Create waterfall plot without forcing it to display
    shap.plots.waterfall(
        explanation[0],
        max_display=max_display,
        show=False,
    )

    figure = plt.gcf()

    return {
        "prediction": prediction,
        "prediction_label": (
            "Churn" if prediction == 1 else "No Churn"
        ),
        "probabilities": probabilities,
        "contributions": contributions,
        "shap_explanation": explanation,
        "figure": figure,
    }


"""
def explain_global_for_dashboard(
    explainer,
    X: pd.DataFrame,
    max_display: int = 15,
):
    '''
    Generate global SHAP summary and feature importance.

    Returns a summary figure and a ranked importance table.
    '''

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a DataFrame.")

    if X.empty:
        raise ValueError("X is empty.")

    if X.isna().any().any():
        raise ValueError("X contains missing values.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("X contains invalid values.")

    # Calculate SHAP values for all supplied customers
    explanation = explainer(X)

    # Global feature importance
    importance = pd.DataFrame({
        "Feature": X.columns,
        "Mean_Absolute_SHAP": np.abs(
            explanation.values
        ).mean(axis=0),
    }).sort_values(
        "Mean_Absolute_SHAP",
        ascending=False,
    ).reset_index(drop=True)

    # Global beeswarm summary plot
    shap.plots.beeswarm(
        explanation,
        max_display=max_display,
        show=False,
    )

    figure = plt.gcf()

    return {
        "importance": importance,
        "shap_explanation": explanation,
        "figure": figure,
    }"""


def explain_global_for_dashboard(
    explainer,
    X: pd.DataFrame,
    max_display: int = 15,
):
    """
    Generate global SHAP summary and feature importance.

    Returns a summary figure and a ranked importance table.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a DataFrame.")

    if X.empty:
        raise ValueError("X is empty.")

    if X.shape[1] == 0:
        raise ValueError("X contains no feature columns.")

    if X.isna().any().any():
        raise ValueError("X contains missing values.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("X contains invalid values.")

    # Calculate SHAP values for all supplied customers
    explanation = explainer(X)

    # Global feature importance
    importance = pd.DataFrame({
        "Feature": X.columns,
        "Mean_Absolute_SHAP": np.abs(
            explanation.values
        ).mean(axis=0),
    }).sort_values(
        "Mean_Absolute_SHAP",
        ascending=False,
    ).reset_index(drop=True)

    # Global beeswarm summary plot
    shap.plots.beeswarm(
        explanation,
        max_display=max_display,
        show=False,
    )

    figure = plt.gcf()

    return {
        "importance": importance,
        "shap_explanation": explanation,
        "figure": figure,
    }
