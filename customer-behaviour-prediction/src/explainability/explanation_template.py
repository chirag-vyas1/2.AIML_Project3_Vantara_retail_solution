
import pandas as pd


def generate_churn_explanation(
    prediction,
    probabilities,
    contributions,
    customer_idx=None,
    top_n=3,
):
    """
    Generate a plain-language explanation of a churn prediction.

    Parameters
    ----------
    prediction:
        Predicted class: 0 (no churn) or 1 (churn).

    probabilities:
        Dictionary of class probabilities, for example
        {"0": 0.15, "1": 0.85}.

    contributions:
        DataFrame containing either:
        - Feature and SHAP_Value, or
        - Feature_Condition and LIME_Weight.

    customer_idx:
        Optional customer row index for reporting.

    top_n:
        Number of leading contributing features to include.

    Returns
    -------
    dict
        Summary, feature explanations, and business recommendation.
    """

    if prediction not in (0, 1):
        raise ValueError("prediction must be 0 or 1.")

    if not isinstance(probabilities, dict):
        raise TypeError("probabilities must be a dictionary.")

    if not isinstance(contributions, pd.DataFrame):
        raise TypeError("contributions must be a DataFrame.")

    if contributions.empty:
        raise ValueError("contributions cannot be empty.")

    if top_n < 1:
        raise ValueError("top_n must be positive.")

    # Determine the predicted class probability
    churn_probability = float(probabilities.get("1", 0.0))

    if not 0 <= churn_probability <= 1:
        raise ValueError("Churn probability must be between 0 and 1.")

    prediction_label = "Churn" if prediction == 1 else "No Churn"

    # Identify the contribution column
    if "SHAP_Value" in contributions.columns:
        contribution_column = "SHAP_Value"
        feature_column = "Feature"
        method_name = "SHAP"

    elif "LIME_Weight" in contributions.columns:
        contribution_column = "LIME_Weight"

        feature_column = (
            "Feature_Condition"
            if "Feature_Condition" in contributions.columns
            else "Feature"
        )
        method_name = "LIME"

    else:
        raise ValueError(
            "Contributions must contain SHAP_Value or LIME_Weight."
        )

    if feature_column not in contributions.columns:
        raise ValueError(
            f"Missing feature column: {feature_column}"
        )

    # Rank the strongest positive and negative contributions
    table = contributions.copy()
    table[contribution_column] = pd.to_numeric(
        table[contribution_column],
        errors="raise",
    )

    if not table[contribution_column].map(
        lambda value: pd.notna(value) and abs(value) != float("inf")
    ).all():
        raise ValueError("Contribution values must be finite.")

    positive = table[table[contribution_column] > 0].sort_values(
        contribution_column,
        ascending=False,
    )

    negative = table[table[contribution_column] < 0].sort_values(
        contribution_column,
        ascending=True,
    )

    # For binary churn classification, positive contributions
    # support class 1 and negative contributions oppose class 1.
    if prediction == 1:
        supporting = positive.head(top_n)
        opposing = negative.head(top_n)
    else:
        supporting = negative.head(top_n)
        opposing = positive.head(top_n)

    supporting_features = supporting[feature_column].tolist()
    opposing_features = opposing[feature_column].tolist()

    if prediction == 1:
        if churn_probability >= 0.70:
            risk_level = "High"
        elif churn_probability >= 0.40:
            risk_level = "Moderate"
        else:
            risk_level = "Lower"

        summary = (
            f"The model predicts that this customer may churn. "
            f"The estimated churn probability is "
            f"{churn_probability:.1%}, classified as {risk_level} "
            f"risk using this template's thresholds."
        )

        recommendation = (
            "Consider reviewing this customer's purchase history "
            "and engagement before deciding whether a retention "
            "campaign is appropriate."
        )

    else:
        risk_level = (
            "Lower"
            if churn_probability < 0.40
            else "Elevated"
        )

        summary = (
            f"The model predicts that this customer will not churn. "
            f"The estimated churn probability is "
            f"{churn_probability:.1%}."
        )

        recommendation = (
            "Continue monitoring customer behaviour and reassess "
            "if their purchase or engagement patterns change."
        )

    # Convert feature names into readable text
    def format_features(feature_list):
        return ", ".join(
            str(name).replace("_", " ")
            for name in feature_list
        ) or "No clear features identified by this explanation."

    explanation_text = (
        summary
        + f"\n\nExplanation method: {method_name}."
        + f"\nFeatures supporting the predicted class: "
        f"{format_features(supporting_features)}."
        + f"\nFeatures opposing the predicted class: "
        f"{format_features(opposing_features)}."
        + f"\n\nSuggested next step: {recommendation}"
    )

    return {
        "customer_idx": customer_idx,
        "prediction": int(prediction),
        "prediction_label": prediction_label,
        "churn_probability": churn_probability,
        "risk_level": risk_level,
        "explanation_method": method_name,
        "supporting_features": supporting_features,
        "opposing_features": opposing_features,
        "summary": summary,
        "recommendation": recommendation,
        "explanation_text": explanation_text,
    }
