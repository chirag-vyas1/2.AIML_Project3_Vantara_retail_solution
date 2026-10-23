
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import PartialDependenceDisplay


def generate_pdp_analysis(
    model,
    X: pd.DataFrame,
    features=None,
    feature_names=None,
    output_dir=None,
    target_class=1,
    grid_resolution=20,
    max_samples=1000,
    show=False,
):
    """
    Generate Partial Dependence Plots (PDPs) for a fitted
    classification model.

    Parameters
    ----------
    model:
        Fitted classifier supporting predict_proba().

    X:
        Representative engineered feature DataFrame.

    features:
        Feature names to analyse. Defaults to Recency,
        Engagement_Score and Frequency when available.

    feature_names:
        Expected model feature names and order.

    output_dir:
        Optional directory for saving plots and CSV files.

    target_class:
        Class whose predicted probability is analysed.

    grid_resolution:
        Number of grid points per feature.

    max_samples:
        Maximum number of samples used for PDP calculation.
        Set to None to use all rows.

    show:
        Whether to display plots.

    Returns
    -------
    dict
        Target class, selected features, figures, plot paths,
        CSV paths and computed PDP tables.
    """

    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.inspection import PartialDependenceDisplay

    # --------------------------------------------------
    # 1. Validate input data
    # --------------------------------------------------
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if X.empty:
        raise ValueError("X is empty.")

    if X.columns.duplicated().any():
        raise ValueError("X contains duplicate column names.")

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "The model must support predict_proba()."
        )

    expected_features = (
        list(feature_names)
        if feature_names is not None
        else list(X.columns)
    )

    if not expected_features:
        raise ValueError("No model features were provided.")

    if len(expected_features) != len(set(expected_features)):
        raise ValueError(
            "Expected feature names contain duplicates."
        )

    missing_features = [
        name for name in expected_features
        if name not in X.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    # Preserve the exact feature order expected by the model.
    X_pdp = X.loc[:, expected_features].copy()

    # PDP compatibility: convert numeric columns to float.
    try:
        X_pdp = X_pdp.astype("float64")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "All PDP input features must be numeric."
        ) from exc

    values = X_pdp.to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError(
            "PDP input contains missing or infinite values."
        )

    if (
        hasattr(model, "n_features_in_")
        and X_pdp.shape[1] != model.n_features_in_
    ):
        raise ValueError(
            f"Model expects {model.n_features_in_} features, "
            f"but received {X_pdp.shape[1]}."
        )

    if grid_resolution < 2:
        raise ValueError(
            "grid_resolution must be at least 2."
        )

    if max_samples is not None:
        if not isinstance(max_samples, (int, np.integer)):
            raise TypeError(
                "max_samples must be an integer or None."
            )
        if max_samples < 1:
            raise ValueError(
                "max_samples must be positive or None."
            )

    # --------------------------------------------------
    # 2. Validate target class
    # --------------------------------------------------
    if not hasattr(model, "classes_"):
        raise ValueError(
            "The fitted model must expose classes_."
        )

    class_labels = list(model.classes_)

    matching_classes = [
        index
        for index, label in enumerate(class_labels)
        if label == target_class
    ]

    if not matching_classes:
        raise ValueError(
            f"Target class {target_class!r} is not in "
            f"model classes {class_labels}."
        )

    # sklearn expects the index of the desired probability
    # column, not the class label itself.
    target_class_index = matching_classes[0]

    # --------------------------------------------------
    # 3. Select features
    # --------------------------------------------------
    if features is None:
        defaults = [
            "Recency",
            "Engagement_Score",
            "Frequency",
        ]

        features = [
            name for name in defaults
            if name in X_pdp.columns
        ]
    else:
        features = list(features)

    if not features:
        raise ValueError(
            "No valid PDP features were selected."
        )

    if len(features) != len(set(features)):
        raise ValueError(
            "The requested PDP features contain duplicates."
        )

    missing_pdp_features = [
        name for name in features
        if name not in X_pdp.columns
    ]

    if missing_pdp_features:
        raise ValueError(
            f"Requested PDP features are missing: "
            f"{missing_pdp_features}"
        )

    # --------------------------------------------------
    # 4. Prepare output directory
    # --------------------------------------------------
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # --------------------------------------------------
    # 5. Generate PDP for each feature
    # --------------------------------------------------
    results = {
        "target_class": target_class,
        "features": features,
        "plots": {},
        "pdp_data": {},
    }

    for feature in features:
        fig, ax = plt.subplots(figsize=(8, 5))

        try:
            display = PartialDependenceDisplay.from_estimator(
                estimator=model,
                X=X_pdp,
                features=[feature],
                kind="average",
                response_method="predict_proba",
                target=target_class_index,
                grid_resolution=grid_resolution,
                percentiles=(0.05, 0.95),
                subsample=max_samples,
                random_state=42,
                ax=ax,
            )

            # Extract the computed PDP grid and average values.
            pdp_result = display.pd_results[0]

            grid_values = np.asarray(
                pdp_result["grid_values"][0]
            ).reshape(-1)

            # The result is already restricted to the requested
            # target class. Do not index it again by class index.
            average_values = np.asarray(
                pdp_result["average"]
            ).reshape(-1)

            if len(grid_values) != len(average_values):
                raise ValueError(
                    f"PDP grid/value length mismatch for {feature}."
                )

            if not np.isfinite(average_values).all():
                raise ValueError(
                    f"PDP values for {feature} are not finite."
                )

            pdp_table = pd.DataFrame({
                feature: grid_values,
                "Average_Predicted_Probability": average_values,
            })

            ax.set_title(
                f"Partial Dependence: {feature}"
            )
            ax.set_xlabel(feature)
            ax.set_ylabel(
                f"Average predicted probability "
                f"(class {target_class})"
            )
            ax.set_ylim(0, 1)

            fig.tight_layout()

            # ------------------------------------------
            # 6. Save artifacts
            # ------------------------------------------
            plot_path = None
            csv_path = None

            if output_dir is not None:
                safe_feature_name = "".join(
                    char
                    if char.isalnum() or char in "_-"
                    else "_"
                    for char in feature
                )

                plot_path = (
                    output_dir
                    / f"pdp_{safe_feature_name}.png"
                )

                csv_path = (
                    output_dir
                    / f"pdp_{safe_feature_name}.csv"
                )

                fig.savefig(
                    plot_path,
                    dpi=150,
                    bbox_inches="tight",
                )

                pdp_table.to_csv(
                    csv_path,
                    index=False,
                )

            results["pdp_data"][feature] = pdp_table

            results["plots"][feature] = {
                "figure": fig,
                "plot_path": plot_path,
                "csv_path": csv_path,
            }

            if show:
                plt.show()
            else:
                plt.close(fig)

        except Exception:
            plt.close(fig)
            raise

    print("PDP analysis completed successfully.")
    print("Target class:", target_class)
    print("Features analysed:", features)

    if output_dir is not None:
        print("Output directory:", output_dir)

    return results
