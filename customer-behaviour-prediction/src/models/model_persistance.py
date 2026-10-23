from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[2]

"""def save_model_artifact(
    model,
    model_name,
    model_type,
    config,
    best_params=None,
    evaluation_results=None,
    training_time=None,
    preprocessing=None,
    metadata=None,
):
    """"""
    Save a trained model and its associated experiment information.

    Parameters
    ----------
    model : object
        Trained model.

    model_name : str
        Name of the model artifact.

    model_type : str
        High-level model category, for example:
        'classification', 'regression', 'deep_learning'.

    config : dict
        Project configuration.

    best_params : dict, optional
        Best hyperparameters obtained during tuning.

    evaluation_results : dict, optional
        Model evaluation metrics.

    training_time : float, optional
        Training or tuning time in seconds.

    preprocessing : object, optional
        Preprocessing information.

    metadata : dict, optional
        Additional model information.

    Returns
    -------
    str
        Path of the saved model artifact.
    """"""

    project_root = get_project_root()

    model_dir = (
        project_root
        / config["model_artifacts"]["model_dir"]
        / model_type
        / model_name
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact = {
        "model": model,
        "model_name": model_name,
        "model_type": model_type,
        "best_params": best_params,
        "evaluation_results": evaluation_results,
        "training_time": training_time,
        "preprocessing": preprocessing,
        "metadata": metadata,
    }

    model_path = (
        model_dir
        / f"{model_name}_artifact.joblib"
    )

    joblib.dump(
        artifact,
        model_path,
    )

    return str(model_path)
"""

def save_model_artifact(
    model,
    model_name,
    model_type,
    config,
    best_params=None,
    evaluation_results=None,
    training_time=None,
    preprocessing=None,
    metadata=None,
):
    """Save model and experiment artifacts."""

    project_root = get_project_root()

    model_dir = (
        project_root
        / config["model_artifacts"]["model_dir"]
        / model_type
        / model_name
    )
    model_dir.mkdir(parents=True, exist_ok=True)

    # Use native Keras format for TensorFlow/Keras models.
    if hasattr(model, "save") and hasattr(model, "trainable_variables"):
        model_path = model_dir / f"{model_name}.keras"
        model.save(model_path)
    else:
        model_path = model_dir / f"{model_name}_artifact.joblib"

    # Keep the existing Joblib artifact format for traditional ML models.
    if model_path.suffix == ".joblib":
        artifact = {
            "model": model,
            "model_name": model_name,
            "model_type": model_type,
            "best_params": best_params,
            "evaluation_results": evaluation_results,
            "training_time": training_time,
            "preprocessing": preprocessing,
            "metadata": metadata,
        }
        joblib.dump(artifact, model_path)

    # Save preprocessing and experiment details for either model type.
    preprocessing_path = None
    if preprocessing is not None:
        preprocessing_path = (
            model_dir / f"{model_name}_preprocessing.joblib"
        )
        joblib.dump(preprocessing, preprocessing_path)

    metadata_path = model_dir / f"{model_name}_metadata.joblib"
    joblib.dump(
        {
            "model_name": model_name,
            "model_type": model_type,
            "best_params": best_params,
            "evaluation_results": evaluation_results,
            "training_time": training_time,
            "metadata": metadata,
        },
        metadata_path,
    )

    return {
        "model_path": str(model_path),
        "preprocessing_path": (
            str(preprocessing_path) if preprocessing_path else None
        ),
        "metadata_path": str(metadata_path),
    }


def load_model_artifacts(
    config,
    get_project_root,
    model_key,
    model_type="classification",
    X_validation=None,
):
    """Load and validate a model using artifact settings from YAML."""

    from pathlib import Path
    import joblib
    import numpy as np
    import pandas as pd

    # 1. Read artifact configuration from YAML
    try:
        artifact_config = config["model"][model_type][model_key]
        model_root = config["model_artifacts"]["model_dir"]

        model_name = artifact_config["model_name"]
        expected_estimator = artifact_config["estimator_class"]

    except KeyError as exc:
        raise KeyError(
            f"Missing model artifact configuration: {exc}"
        ) from exc

    model_name = artifact_config["model_name"]
    expected_estimator = artifact_config["estimator_class"]
    artifact_format = artifact_config.get("artifact_format", "joblib")

    if artifact_format != "joblib":
        raise ValueError(
            f"Unsupported artifact format: {artifact_format}"
        )

    # 2. Resolve artifact path
    project_root = Path(get_project_root()).resolve()

    artifact_dir = (
        project_root / model_root / model_type / model_name
    )
    artifact_path = artifact_dir / f"{model_name}_artifact.joblib"

    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {artifact_path}"
        )

    # 3. Load artifact
    artifact = joblib.load(artifact_path)

    if not isinstance(artifact, dict):
        raise TypeError("The saved artifact must be a dictionary.")

    required_keys = {
        "model", "model_name", "model_type", "best_params",
        "evaluation_results", "preprocessing", "metadata",
    }

    missing_keys = required_keys - set(artifact)
    if missing_keys:
        raise ValueError(
            f"Artifact is missing keys: {sorted(missing_keys)}"
        )

    model = artifact["model"]

    if type(model).__name__ != expected_estimator:
        raise TypeError(
            f"Expected {expected_estimator}, "
            f"found {type(model).__name__}."
        )

    if artifact["model_name"] != model_name:
        raise ValueError("Artifact model name does not match YAML.")

    if artifact["model_type"] != model_type:
        raise ValueError("Artifact model type does not match YAML.")

    # 4. Validate fitted model and feature metadata
    if not hasattr(model, "n_features_in_"):
        raise ValueError("The model does not appear to be fitted.")

    n_features = int(model.n_features_in_)
    feature_names = getattr(model, "feature_names_in_", None)

    if feature_names is None and hasattr(model, "get_booster"):
        feature_names = model.get_booster().feature_names

    if feature_names is not None:
        feature_names = list(feature_names)

        if len(feature_names) != n_features:
            raise ValueError("Feature count does not match feature names.")

        if len(set(feature_names)) != len(feature_names):
            raise ValueError("Duplicate feature names were found.")

    # 5. Optional prediction smoke test
    prediction_test_passed = None

    if X_validation is not None:
        if not isinstance(X_validation, pd.DataFrame):
            raise TypeError("X_validation must be a pandas DataFrame.")

        if X_validation.empty:
            raise ValueError("X_validation must not be empty.")

        if feature_names is None:
            raise ValueError(
                "Model feature names are unavailable for validation."
            )

        missing = [
            name for name in feature_names
            if name not in X_validation.columns
        ]
        if missing:
            raise ValueError(f"Missing model features: {missing}")

        X_check = X_validation.loc[:, feature_names].copy()

        if X_check.isnull().any().any():
            raise ValueError("Validation data contains missing values.")

        if not all(
            pd.api.types.is_numeric_dtype(dtype)
            for dtype in X_check.dtypes
        ):
            raise TypeError("All model features must be numeric.")

        if not np.isfinite(X_check.to_numpy(dtype=float)).all():
            raise ValueError("Validation data contains non-finite values.")

        predictions = model.predict(X_check)

        if len(predictions) != len(X_check):
            raise ValueError("Unexpected prediction output length.")

        prediction_test_passed = True

    # 6. Return loaded model and validation information
    return {
        "model": model,
        "artifact": artifact,
        "model_name": model_name,
        "model_type": model_type,
        "artifact_path": str(artifact_path),
        "feature_names": feature_names,
        "n_features": n_features,
        "best_params": artifact["best_params"],
        "model_parameters": (
            model.get_params(deep=False)
            if hasattr(model, "get_params")
            else None
        ),
        "preprocessing": artifact["preprocessing"],
        "evaluation_results": artifact["evaluation_results"],
        "metadata": artifact["metadata"],
        "validation_report": {
            "artifact_exists": True,
            "artifact_loaded": True,
            "estimator_class": type(model).__name__,
            "feature_count": n_features,
            "feature_names_available": feature_names is not None,
            "preprocessing_available": artifact["preprocessing"] is not None,
            "prediction_smoke_test_passed": prediction_test_passed,
        },
    }
