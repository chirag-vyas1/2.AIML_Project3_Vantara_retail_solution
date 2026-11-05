
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import Mock

from src.services.batch_prediction_service import predict_batch_from_csv


@pytest.fixture
def sample_batch_csv(tmp_path):
    """Create a small CSV containing customer features."""
    csv_path = tmp_path / "customers.csv"

    pd.DataFrame(
        {
            "feature_a": [10, 20, 30],
            "feature_b": [1.5, 2.5, 3.5],
            "Customer_ID": ["C001", "C002", "C003"],
        }
    ).to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def mock_model_bundle():
    """Create a mock model bundle with predictable predictions."""
    model = Mock()
    model.classes_ = np.array([0, 1])
    model.predict.return_value = np.array([0, 1, 1])
    model.predict_proba.return_value = np.array(
        [
            [0.90, 0.10],
            [0.25, 0.75],
            [0.10, 0.90],
        ]
    )

    return {
        "model": model,
        "model_name": "test_churn_model",
        "feature_names": ["feature_a", "feature_b"],
    }


def test_predict_batch_from_csv_success(
    tmp_path, sample_batch_csv, mock_model_bundle
):
    """Test successful batch predictions and report creation."""
    output_path = tmp_path / "reports" / "predictions.csv"

    result = predict_batch_from_csv(
        csv_path=sample_batch_csv,
        model_bundle=mock_model_bundle,
        output_path=output_path,
    )

    # Verify prediction output.
    assert len(result) == 3
    assert result["Predicted_Class"].tolist() == [0, 1, 1]
    assert result["Churn_Probability"].tolist() == [0.10, 0.75, 0.90]
    assert result["Model_Name"].tolist() == ["test_churn_model"] * 3

    # Verify original columns are preserved.
    assert result["Customer_ID"].tolist() == ["C001", "C002", "C003"]

    # Verify report is saved and can be read back.
    assert output_path.is_file()
    saved_report = pd.read_csv(output_path)
    assert len(saved_report) == 3
    assert saved_report["Predicted_Class"].tolist() == [0, 1, 1]

    # Verify features are selected in the expected order.
    actual_features = mock_model_bundle["model"].predict.call_args.args[0]
    assert actual_features.columns.tolist() == ["feature_a", "feature_b"]

    mock_model_bundle["model"].predict.assert_called_once()
    mock_model_bundle["model"].predict_proba.assert_called_once()


def test_predict_batch_from_csv_missing_input_file(
    tmp_path, mock_model_bundle
):
    """Test that a nonexistent input CSV raises FileNotFoundError."""
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Input CSV not found"):
        predict_batch_from_csv(
            csv_path=missing_path,
            model_bundle=mock_model_bundle,
            output_path=tmp_path / "output.csv",
        )


def test_predict_batch_from_csv_empty_file(
    tmp_path, mock_model_bundle
):
    """Test that a CSV with headers but no records raises ValueError."""
    csv_path = tmp_path / "empty.csv"
    pd.DataFrame(columns=["feature_a", "feature_b"]).to_csv(
        csv_path, index=False
    )

    with pytest.raises(
        ValueError,
        match="contains no customer records",
    ):
        predict_batch_from_csv(
            csv_path=csv_path,
            model_bundle=mock_model_bundle,
            output_path=tmp_path / "output.csv",
        )


def test_predict_batch_from_csv_missing_features(
    tmp_path, mock_model_bundle
):
    """Test that missing model features raise ValueError."""
    csv_path = tmp_path / "incomplete.csv"
    pd.DataFrame({"feature_a": [10, 20]}).to_csv(
        csv_path, index=False
    )

    with pytest.raises(
        ValueError,
        match="CSV is missing required model features",
    ):
        predict_batch_from_csv(
            csv_path=csv_path,
            model_bundle=mock_model_bundle,
            output_path=tmp_path / "output.csv",
        )


def test_predict_batch_from_csv_non_numeric_features(
    tmp_path, mock_model_bundle
):
    """Test that non-numeric feature values raise ValueError."""
    csv_path = tmp_path / "invalid.csv"
    pd.DataFrame(
        {
            "feature_a": ["ten", "twenty"],
            "feature_b": [1.5, 2.5],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        predict_batch_from_csv(
            csv_path=csv_path,
            model_bundle=mock_model_bundle,
            output_path=tmp_path / "output.csv",
        )


def test_predict_batch_from_csv_missing_values(
    tmp_path, mock_model_bundle
):
    """Test that missing feature values raise ValueError."""
    csv_path = tmp_path / "missing_values.csv"
    pd.DataFrame(
        {
            "feature_a": [10, np.nan],
            "feature_b": [1.5, 2.5],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(
        ValueError,
        match="contains missing values",
    ):
        predict_batch_from_csv(
            csv_path=csv_path,
            model_bundle=mock_model_bundle,
            output_path=tmp_path / "output.csv",
        )


@pytest.mark.parametrize(
    "invalid_value",
    [np.inf, -np.inf],
)
def test_predict_batch_from_csv_infinite_values(
    tmp_path, mock_model_bundle, invalid_value
):
    """Test that infinite feature values raise ValueError."""
    csv_path = tmp_path / "infinite_values.csv"
    pd.DataFrame(
        {
            "feature_a": [10, invalid_value],
            "feature_b": [1.5, 2.5],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(
        ValueError,
        match="infinite or invalid feature values",
    ):
        predict_batch_from_csv(
            csv_path=csv_path,
            model_bundle=mock_model_bundle,
            output_path=tmp_path / "output.csv",
        )
