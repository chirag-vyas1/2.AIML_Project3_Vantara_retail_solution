
import joblib
import pytest
from pathlib import Path
from unittest.mock import patch
from sklearn.linear_model import LogisticRegression
from src.models.model_persistance import save_model_artifact


def test_save_model_artifact(tmp_path):
    """Test saving a trained model and its associated metadata."""

    model = LogisticRegression()
    model_name = "test_logistic_regression"
    model_type = "classification"

    config = {
        "model_artifacts": {
            "model_dir": "model_artifacts",
        }
    }

    best_params = {"C": 1.0}
    evaluation_results = {"accuracy": 0.95}
    preprocessing = {"scaler": "standard"}
    metadata = {"purpose": "unit test"}

    # Redirect artifact saving to a temporary directory.
    with patch(
        "src.models.model_persistance.get_project_root",
        return_value=tmp_path,
    ):
        saved_paths = save_model_artifact(
            model=model,
            model_name=model_name,
            model_type=model_type,
            config=config,
            best_params=best_params,
            evaluation_results=evaluation_results,
            training_time=2.5,
            preprocessing=preprocessing,
            metadata=metadata,
        )

    # Verify the returned paths.
    assert Path(saved_paths["model_path"]).is_file()
    assert Path(saved_paths["preprocessing_path"]).is_file()
    assert Path(saved_paths["metadata_path"]).is_file()

    # Verify the saved model artifact.
    artifact = joblib.load(saved_paths["model_path"])

    assert isinstance(artifact["model"], LogisticRegression)
    assert artifact["model_name"] == model_name
    assert artifact["model_type"] == model_type
    assert artifact["best_params"] == best_params
    assert artifact["evaluation_results"] == evaluation_results
    assert artifact["training_time"] == 2.5
    assert artifact["preprocessing"] == preprocessing
    assert artifact["metadata"] == metadata

    # Verify the preprocessing artifact.
    saved_preprocessing = joblib.load(
        saved_paths["preprocessing_path"]
    )
    assert saved_preprocessing == preprocessing

    # Verify the metadata artifact.
    saved_metadata = joblib.load(saved_paths["metadata_path"])

    assert saved_metadata["model_name"] == model_name
    assert saved_metadata["model_type"] == model_type
    assert saved_metadata["best_params"] == best_params
    assert saved_metadata["evaluation_results"] == evaluation_results
    assert saved_metadata["training_time"] == 2.5
    assert saved_metadata["metadata"] == metadata

def test_load_model_artifacts(tmp_path):
    """Test loading and validating a saved model artifact."""
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    from src.models.model_persistance import (
        save_model_artifact,
        load_model_artifacts,
    )

    # 1. Train a model with named features.
    X = pd.DataFrame(
        {
            "feature_a": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_b": [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        }
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    model = LogisticRegression()
    model.fit(X, y)

    # 2. Save the model into a temporary directory.
    config = {
        "model_artifacts": {
            "model_dir": "model_artifacts",
        }
    }

    from unittest.mock import patch

    with patch(
         "src.models.model_persistance.get_project_root",
          return_value=tmp_path,):
        
        saved_paths = save_model_artifact(
            model=model,
            model_name="test_logistic_regression",
            model_type="classification",
            config=config,
            best_params={"C": 1.0},
            evaluation_results={"accuracy": 1.0},
            preprocessing=None,
            metadata={"purpose": "unit test"},
        )

    # 3. Configure the model loader.
    load_config = {
        "model": {
            "classification": {
                "test_model": {
                    "model_name": "test_logistic_regression",
                    "estimator_class": "LogisticRegression",
                    "artifact_format": "joblib",
                }
            }
        },
        "model_artifacts": {
            "model_dir": "model_artifacts",
        },
    }

    # 4. Load the model and run the prediction smoke test.
    loaded = load_model_artifacts(
        config=load_config,
        get_project_root=lambda: tmp_path,
        model_key="test_model",
        model_type="classification",
        X_validation=X,
    )

    # 5. Verify the loaded model and validation report.
    assert isinstance(loaded["model"], LogisticRegression)
    assert loaded["model_name"] == "test_logistic_regression"
    assert loaded["model_type"] == "classification"
    assert loaded["n_features"] == 2
    assert loaded["feature_names"] == ["feature_a", "feature_b"]

    assert loaded["validation_report"]["artifact_exists"] is True
    assert loaded["validation_report"]["artifact_loaded"] is True
    assert (
        loaded["validation_report"]["prediction_smoke_test_passed"]
        is True
    )
    assert loaded["preprocessing"] is None
    assert loaded["metadata"] == {"purpose": "unit test"}


def test_load_model_artifacts_file_not_found(tmp_path):
    """Verify a missing model artifact raises FileNotFoundError."""
    from src.models.model_persistance import load_model_artifacts

    config = {
        "model": {
            "classification": {
                "missing_model": {
                    "model_name": "model_that_does_not_exist",
                    "estimator_class": "LogisticRegression",
                }
            }
        },
        "model_artifacts": {
            "model_dir": "model_artifacts",
        },
    }

    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        load_model_artifacts(
            config=config,
            get_project_root=lambda: tmp_path,
            model_key="missing_model",
            model_type="classification",
        )
