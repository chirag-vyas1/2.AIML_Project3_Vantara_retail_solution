from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

################################################################
#---- for classification model  -----------------
################################################################
def display_model_registry(
    model_config: dict,
    title: str = "MODEL REGISTRY",
) -> None:
    """
    Display model configuration in a clean, notebook-friendly format.

    Parameters
    ----------
    model_config : dict
        Model configuration loaded from config.yaml.
        Expected to contain model names with their configuration.

    title : str, default="MODEL REGISTRY"
        Heading displayed above the configuration.
    """

    print("=" * 90)
    print(title)
    print("=" * 90)

    for model_name, config in model_config.items():

        display_name = model_name.replace(
            "_", " "
        ).title()

        enabled = config.get(
            "enabled",
            False
        )

        print(f"\n{display_name}")
        print("-" * 90)

        print(f"Enabled: {enabled}")

        base_parameters = config.get(
            "base_parameters",
            {}
        )

        print("\nBase Parameters")

        if base_parameters:
            for parameter, value in base_parameters.items():
                print(
                    f"  {parameter:<30}: {value}"
                )
        else:
            print("  None")

        hyperparameter_space = config.get(
            "hyperparameter_space",
            {}
        )

        print("\nHyperparameter Space")

        if hyperparameter_space:
            for parameter, values in hyperparameter_space.items():
                print(
                    f"  {parameter:<30}: {values}"
                )
        else:
            print("  Not configured")

    print("\n" + "=" * 90)

def display_baseline_training_results(
    baseline_models: dict,
    title: str = "BASELINE MODEL TRAINING RESULTS",
) -> None:
    """
    Display baseline classification model training results
    in a clean Jupyter Notebook table.

    Parameters
    ----------
    baseline_models : dict
        Dictionary containing fitted models, training times,
        and feature scaling information.

    title : str, default="BASELINE MODEL TRAINING RESULTS"
        Heading displayed above the results.
    """

    results = []

    for model_name, result in baseline_models.items():

        model = result["model"]

        display_name = model_name.replace(
            "_", " "
        ).title()

        results.append(
            {
                "Model": display_name,
                "Estimator": model.__class__.__name__,
                "Feature Scaling": (
                    "Yes"
                    if result["feature_scaling"]
                    else "No"
                ),
                "Training Time (sec)": round(
                    result["training_time"],
                    4,
                ),
            }
        )

    results_df = pd.DataFrame(results)

    print("=" * 90)
    print(title)
    print("=" * 90)

    display(
        results_df.style
        .format(
            {
                "Training Time (sec)": "{:.4f}"
            }
        )
        .set_properties(
            **{
                "text-align": "center",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("text-align", "center"),
                        ("font-weight", "bold"),
                    ],
                }
            ]
        )
    )

def display_classification_evaluation(
    evaluation_results: dict,
    title: str = "BASELINE CLASSIFICATION EVALUATION",
) -> None:
    """
    Display classification evaluation results in a clean,
    ranked Jupyter Notebook table.

    Models are ranked by ROC-AUC, with Recall used as
    the secondary ranking criterion.

    Parameters
    ----------
    evaluation_results : dict
        Dictionary containing evaluation metrics for each model.

    title : str, default="BASELINE CLASSIFICATION EVALUATION"
        Heading displayed above the results.
    """

    results = []

    for model_name, metrics in evaluation_results.items():

        display_name = model_name.replace(
            "_", " "
        ).title()

        results.append(
            {
                "Model": display_name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"],
                "ROC-AUC": metrics["roc_auc"],
            }
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by=["ROC-AUC", "Recall"],
        ascending=False,
    ).reset_index(drop=True)

    print("=" * 100)
    print(title)
    print("=" * 100)

    display(
        results_df.style
        .format(
            {
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1": "{:.4f}",
                "ROC-AUC": "{:.4f}",
            }
        )
        .background_gradient(
            subset=[
                "Recall",
                "ROC-AUC",
            ],
        )
        .set_properties(
            **{
                "text-align": "center",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("text-align", "center"),
                        ("font-weight", "bold"),
                    ],
                }
            ]
        )
    )


def display_classification_evaluation_visuals(
    evaluation_results: dict,
    model_name: str = "Classification Model",
) -> None:
    """
    Display confusion matrix and classification metrics
    for a fitted classification model.

    Parameters
    ----------
    evaluation_results : dict
        Evaluation results returned by
        evaluate_classification_model().

    model_name : str
        Name of the model displayed in the plots.
    """

    confusion = evaluation_results["confusion_matrix"]

    metrics = {
        "Accuracy": evaluation_results["accuracy"],
        "Precision": evaluation_results["precision"],
        "Recall": evaluation_results["recall"],
        "F1": evaluation_results["f1"],
        "ROC-AUC": evaluation_results["roc_auc"],
    }

    # ---------------------------------------------------------
    # Confusion Matrix
    # ---------------------------------------------------------

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted Non-Churn", "Predicted Churn"],
        yticklabels=["Actual Non-Churn", "Actual Churn"],
    )

    plt.title(
        f"{model_name} - Confusion Matrix"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # Classification Metrics
    # ---------------------------------------------------------

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        metrics.keys(),
        metrics.values(),
    )

    plt.ylim(0, 1)

    plt.title(
        f"{model_name} - Classification Metrics"
    )

    plt.ylabel("Score")

    plt.xticks(rotation=0)

    for bar, value in zip(
        bars,
        metrics.values(),
    ):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()


############################################
# --- for regression model --------
############################################

def display_clv_validation(validation_results: dict) -> None:
    """
    Display CLV regression dataset validation results.
    """

    print("CLV REGRESSION DATASET VALIDATION")
    print("=" * 50)

    print(
        f"Rows: {validation_results['row_count']}"
    )

    print(
        f"Columns: {validation_results['column_count']}"
    )

    print(
        f"Duplicate Customer IDs: "
        f"{validation_results['duplicate_customer_id']}"
    )

    print(
        f"Missing Customer IDs: "
        f"{validation_results['missing_customer_id']}"
    )

    print(
        f"Missing Target: "
        f"{validation_results['missing_target']}"
    )

    print(
        f"Negative Target: "
        f"{validation_results['negative_target']}"
    )

    print(
        f"Missing Predictor Values: "
        f"{validation_results['missing_predictor_values']}"
    )

    print("\nTarget Statistics")
    print("-" * 50)

    print(
        f"Target: "
        f"{validation_results['target_column']}"
    )

    print(
        f"Mean: "
        f"{validation_results['target_mean']:.2f}"
    )

    print(
        f"Median: "
        f"{validation_results['target_median']:.2f}"
    )

    print(
        f"Std: "
        f"{validation_results['target_std']:.2f}"
    )

    print(
        f"Minimum: "
        f"{validation_results['target_min']:.2f}"
    )

    print(
        f"Maximum: "
        f"{validation_results['target_max']:.2f}"
    )

    print(
        f"Zero CLV: "
        f"{validation_results['zero_clv_count']} "
        f"({validation_results['zero_clv_percentage']:.2f}%)"
    )

    print(
        f"Positive CLV: "
        f"{validation_results['positive_clv_count']} "
        f"({validation_results['positive_clv_percentage']:.2f}%)"
    )

    print("\nValidation Status")
    print("-" * 50)

    if validation_results["passed"]:
        print("PASSED")
    else:
        print("FAILED")

def display_clv_analysis(
    clv_analysis: dict,
) -> None:
    """
    Display the statistical and structural analysis
    of the customer-level CLV regression dataset.

    Parameters
    ----------
    clv_analysis : dict
        Analysis results returned by analyze_clv_dataset().
    """

    # ============================================================
    # 1. Basic structure
    # ============================================================

    print("=" * 70)
    print("CLV REGRESSION DATASET ANALYSIS")
    print("=" * 70)

    print("\nDataset Structure")
    print("-" * 70)

    print(
        f"Shape: {clv_analysis['shape']}"
    )

    print(
        f"Rows: {clv_analysis['row_count']}"
    )

    print(
        f"Columns: {clv_analysis['column_count']}"
    )

    # ============================================================
    # 2. Duplicate analysis
    # ============================================================

    print("\nDuplicate Analysis")
    print("-" * 70)

    print(
        f"Duplicate rows: "
        f"{clv_analysis['duplicate_rows']}"
    )

    print(
        f"Duplicate Customer IDs: "
        f"{clv_analysis['duplicate_customer_ids']}"
    )

    # ============================================================
    # 3. Missing-value analysis
    # ============================================================

    print("\nMissing-Value Analysis")
    print("-" * 70)

    print(
        f"Total missing values: "
        f"{clv_analysis['total_missing_values']}"
    )

    if clv_analysis["missing_columns"]:

        print("\nColumns with missing values:")

        for column, count in (
            clv_analysis["missing_columns"].items()
        ):
            print(
                f"  {column}: {count}"
            )

    else:
        print(
            "No columns contain missing values."
        )

    # ============================================================
    # 4. CLV target statistics
    # ============================================================

    clv_statistics = (
        clv_analysis["clv_statistics"]
    )

    print("\nCLV Target Analysis")
    print("-" * 70)

    print(
        f"Target column: "
        f"{clv_statistics['target_column']}"
    )

    print(
        f"Prediction window: "
        f"{clv_statistics['prediction_window_days']} days"
    )

    print(
        f"Zero CLV customers: "
        f"{clv_statistics['zero_clv_count']} "
        f"({clv_statistics['zero_clv_percentage']:.2f}%)"
    )

    print(
        f"Positive CLV customers: "
        f"{clv_statistics['positive_clv_count']} "
        f"({clv_statistics['positive_clv_percentage']:.2f}%)"
    )

    print(
        f"Negative CLV customers: "
        f"{clv_statistics['negative_clv_count']} "
        f"({clv_statistics['negative_clv_percentage']:.2f}%)"
    )

    print("\nCLV Distribution")
    print("-" * 70)

    print(
        f"Mean: "
        f"{clv_statistics['mean_clv']:.2f}"
    )

    print(
        f"Median: "
        f"{clv_statistics['median_clv']:.2f}"
    )

    print(
        f"Standard deviation: "
        f"{clv_statistics['standard_deviation']:.2f}"
    )

    print(
        f"Minimum: "
        f"{clv_statistics['minimum_clv']:.2f}"
    )

    print(
        f"25th percentile: "
        f"{clv_statistics['percentile_25']:.2f}"
    )

    print(
        f"75th percentile: "
        f"{clv_statistics['percentile_75']:.2f}"
    )

    print(
        f"90th percentile: "
        f"{clv_statistics['percentile_90']:.2f}"
    )

    print(
        f"95th percentile: "
        f"{clv_statistics['percentile_95']:.2f}"
    )

    print(
        f"99th percentile: "
        f"{clv_statistics['percentile_99']:.2f}"
    )

    print(
        f"Maximum: "
        f"{clv_statistics['maximum_clv']:.2f}"
    )

    # ============================================================
    # 5. Quality checks
    # ============================================================

    print("\nQuality Checks")
    print("-" * 70)

    for check_name, result in (
        clv_analysis["quality_checks"].items()
    ):

        status = "PASS" if result else "FAIL"

        formatted_name = (
            check_name
            .replace("_", " ")
            .title()
        )

        print(
            f"{formatted_name}: {status}"
        )

    # ============================================================
    # 6. Overall quality status
    # ============================================================

    print("\nOverall Quality Status")
    print("-" * 70)

    if clv_analysis["quality_status"]:
        print("PASSED")
    else:
        print("FAILED")

    print("=" * 70)

def display_regression_evaluation(
    evaluation_results: dict,
    title: str = "BASELINE CLV REGRESSION EVALUATION",
) -> None:
    """
    Display regression evaluation results in a clean,
    ranked Jupyter Notebook table.

    Models are ranked by RMSE (ascending), with MAE
    used as the secondary ranking criterion.

    Parameters
    ----------
    evaluation_results : dict
        Dictionary containing regression evaluation
        metrics for each model.

    title : str, default="BASELINE CLV REGRESSION EVALUATION"
        Heading displayed above the results.
    """

    results = []

    for model_name, metrics in evaluation_results.items():

        display_name = model_name.replace(
            "_", " "
        ).title()

        results.append(
            {
                "Model": display_name,
                "MAE": metrics["mae"],
                "RMSE": metrics["rmse"],
                "R²": metrics["r2"],
            }
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by=["RMSE", "MAE"],
        ascending=True,
    ).reset_index(drop=True)

    print("=" * 100)
    print(title)
    print("=" * 100)

    display(
        results_df.style
        .format(
            {
                "MAE": "{:.4f}",
                "RMSE": "{:.4f}",
                "R²": "{:.4f}",
            }
        )
        .background_gradient(
            subset=["RMSE", "MAE"],
            cmap="RdYlGn_r",
        )
        .background_gradient(
            subset=["R²"],
            cmap="RdYlGn",
        )
        .set_properties(
            **{
                "text-align": "center",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("text-align", "center"),
                        ("font-weight", "bold"),
                    ],
                }
            ]
        )
    )


def display_regression_evaluation_visuals(
    evaluation_results: dict,
    model_name: str = "Regression Model",
) -> None:
    """
    Display actual vs. predicted values and regression metrics
    for a fitted regression model.

    Parameters
    ----------
    evaluation_results : dict
        Evaluation results returned by
        evaluate_regression_model().

    model_name : str
        Name of the model displayed in the plots.
    """

    metrics = {
        "MAE": evaluation_results["mae"],
        "RMSE": evaluation_results["rmse"],
        "R²": evaluation_results["r2"],
    }

    # ---------------------------------------------------------
    # Regression Metrics
    # ---------------------------------------------------------

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        metrics.keys(),
        metrics.values(),
    )

    plt.title(
        f"{model_name} - Regression Metrics"
    )

    plt.ylabel("Metric Value")

    # Display metric values above the bars.
    for bar, value in zip(
        bars,
        metrics.values(),
    ):
        offset = max(abs(value) * 0.02, 0.01)

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    plt.axhline(
        y=0,
        color="black",
        linewidth=0.8,
    )

    plt.tight_layout()
    plt.show()


def display_actual_vs_predicted(
    model,
    X_data,
    y_data,
    model_name: str = "Regression Model",
) -> None:
    """
    Display actual vs. predicted values with a reference line.
    """

    y_pred = model.predict(X_data)

    plt.figure(figsize=(7, 6))

    plt.scatter(
        y_data,
        y_pred,
        alpha=0.5,
    )

    lower = min(y_data.min(), y_pred.min())
    upper = max(y_data.max(), y_pred.max())

    plt.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        label="Perfect Prediction",
    )

    plt.title(
        f"{model_name} - Actual vs. Predicted"
    )

    plt.xlabel("Actual Future CLV")
    plt.ylabel("Predicted Future CLV")
    plt.legend()
    plt.tight_layout()
    plt.show()
