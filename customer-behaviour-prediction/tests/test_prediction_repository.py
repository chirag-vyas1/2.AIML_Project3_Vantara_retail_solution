
from unittest.mock import MagicMock, patch

import pytest

from src.utils.prediction_repository import (
    get_customer_features,
    save_prediction,
)

MODULE = "src.utils.prediction_repository"

def test_get_customer_features_success():
    """Return customer features as a dictionary when found."""
    expected_features = {
        "Recency": 10,
        "Frequency": 5,
        "Total_Monetary_Value": 1250.50,
        "Engagement_Score": 0.85,
    }

    engine = MagicMock()
    connection = engine.connect.return_value.__enter__.return_value

    connection.execute.return_value.mappings.return_value.first.return_value = (
        expected_features
    )

    with patch(f"{MODULE}.get_database_engine", return_value=engine):
        result = get_customer_features(101)

    assert result == expected_features
    connection.execute.assert_called_once()
    engine.dispose.assert_called_once()

def test_get_customer_features_customer_not_found():
    """Return None when the customer has no feature record."""
    engine = MagicMock()
    connection = engine.connect.return_value.__enter__.return_value
    connection.execute.return_value.mappings.return_value.first.return_value = None

    with patch(f"{MODULE}.get_database_engine", return_value=engine):
        result = get_customer_features(999)

    assert result is None
    engine.dispose.assert_called_once()

def test_get_customer_features_disposes_engine_on_error():
    """Dispose of the engine even if the database query fails."""
    engine = MagicMock()
    connection = engine.connect.return_value.__enter__.return_value
    connection.execute.side_effect = RuntimeError("Database query failed")

    with patch(f"{MODULE}.get_database_engine", return_value=engine):
        with pytest.raises(RuntimeError, match="Database query failed"):
            get_customer_features(101)

    engine.dispose.assert_called_once()

def test_save_prediction_success():
    """Save a prediction when the customer and features exist."""
    engine = MagicMock()
    connection = engine.begin.return_value.__enter__.return_value
    connection.execute.return_value.rowcount = 1

    with patch(f"{MODULE}.get_database_engine", return_value=engine):
        result = save_prediction(
            customer_id=101,
            model_name="lightgbm",
            predicted_class=1,
            churn_probability=0.92,
        )

    assert result is None
    connection.execute.assert_called_once()
    engine.dispose.assert_called_once()

def test_save_prediction_customer_or_features_missing():
    """Raise ValueError if the prediction insert affects no rows."""
    engine = MagicMock()
    connection = engine.begin.return_value.__enter__.return_value
    connection.execute.return_value.rowcount = 0

    with patch(f"{MODULE}.get_database_engine", return_value=engine):
        with pytest.raises(
            ValueError,
            match="Customer 999 does not have both",
        ):
            save_prediction(
                customer_id=999,
                model_name="lightgbm",
                predicted_class=1,
                churn_probability=0.92,
            )

    engine.dispose.assert_called_once()

def test_save_prediction_disposes_engine_on_database_error():
    """Dispose of the engine if inserting the prediction fails."""
    engine = MagicMock()
    connection = engine.begin.return_value.__enter__.return_value
    connection.execute.side_effect = RuntimeError("Insert failed")

    with patch(f"{MODULE}.get_database_engine", return_value=engine):
        with pytest.raises(RuntimeError, match="Insert failed"):
            save_prediction(
                customer_id=101,
                model_name="lightgbm",
                predicted_class=0,
                churn_probability=0.15,
            )

    engine.dispose.assert_called_once()


#================================================================================================================
import pandas as pd
from unittest.mock import MagicMock, patch

from src.utils.import_customer_features import import_customer_features


import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.utils.import_customer_features import import_customer_features


IMPORT_MODULE = "src.utils.import_customer_features"


def make_customer_features_csv(tmp_path, rows=None):
    """Create a temporary CSV with all required columns."""
    row = {
        "Customer ID": 101,
        "Recency": 10,
        "Frequency": 5,
        "Total_Monetary_Value": 1250.50,
        "Average_Order_Spend": 250.10,
        "Historical_CLV": 1500.0,
        "Customer_Lifetime_Days": 180,
        "Historical_Average_Order_Value": 250.10,
        "Historical_Invoice_Count": 5,
        "First_Purchase_Date": "2024-01-01",
        "Last_Purchase_Date": "2024-06-01",
        "Average_Basket_Size": 4.0,
        "Purchase_Frequency_Trend": 0.1,
        "Interpurchase_Variance": 2.0,
        "Seasonal_Purchase_Concentration": 0.3,
        "Home_Fragrance_Affinity": 0.1,
        "Lighting_Candles_Affinity": 0.2,
        "Stationery_Paper_Affinity": 0.1,
        "Jewelry_Fashion_Accessories_Affinity": 0.1,
        "Bags_Travel_Accessories_Affinity": 0.1,
        "Kitchen_Dining_Affinity": 0.1,
        "Storage_Organization_Affinity": 0.1,
        "Garden_Outdoor_Affinity": 0.1,
        "Bath_Personal_Care_Affinity": 0.1,
        "Toys_Games_Affinity": 0.1,
        "Party_Celebration_Affinity": 0.1,
        "Gifts_Seasonal_Affinity": 0.1,
        "Food_Consumables_Affinity": 0.1,
        "Home_Decor_Affinity": 0.1,
        "Other___Unclassified_Affinity": 0.1,
        "Return_Rate": 0.05,
        "Discount_Markdown_Sensitivity": 0.2,
        "Engagement_Score": 0.85,
        "Engagement_Band": "High",
    }

    data = rows if rows is not None else [row]

    project_root = tmp_path
    csv_path = (
        project_root / "data" / "processed" / "customer_features.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(csv_path, index=False)

    return csv_path, row


def test_import_customer_features_file_not_found(tmp_path):
    """Raise FileNotFoundError when the CSV is missing."""
    with patch(
        f"{IMPORT_MODULE}.get_project_root",
        return_value=tmp_path,
    ):
        with pytest.raises(
            FileNotFoundError,
            match="Customer features CSV not found",
        ):
            import_customer_features()


def test_import_customer_features_missing_columns(tmp_path):
    """Reject CSV files that lack required columns."""
    csv_path, row = make_customer_features_csv(tmp_path)
    pd.DataFrame([{"Customer ID": row["Customer ID"]}]).to_csv(
        csv_path, index=False
    )

    with patch(
        f"{IMPORT_MODULE}.get_project_root",
        return_value=tmp_path,
    ):
        with pytest.raises(ValueError, match="Missing CSV columns"):
            import_customer_features()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"Customer ID": None}, "Customer ID contains missing values"),
        ({"Customer ID": 101}, "Customer ID contains duplicates"),
    ],
)
def test_import_customer_features_invalid_customer_ids(
    tmp_path, change, message
):
    """Reject missing or duplicated customer IDs."""
    _, row = make_customer_features_csv(tmp_path)
    first = row.copy()
    second = row.copy()
    second["Customer ID"] = 102

    if change["Customer ID"] is None:
        first["Customer ID"] = None
    else:
        second["Customer ID"] = change["Customer ID"]

    make_customer_features_csv(tmp_path, rows=[first, second])

    with patch(
        f"{IMPORT_MODULE}.get_project_root",
        return_value=tmp_path,
    ):
        with pytest.raises(ValueError, match=message):
            import_customer_features()


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        (
            "First_Purchase_Date",
            "not-a-date",
             "Unknown datetime string format",
        ),
        (
            "Frequency",
            2.5,
            "Frequency contains non-integer values",
        ),
        (
            "Historical_Invoice_Count",
            3.5,
            "Historical_Invoice_Count contains non-integer values",
        ),
    ],
)
def test_import_customer_features_invalid_values(
    tmp_path, column, value, message
):
    """Reject invalid dates and non-integer count fields."""
    _, row = make_customer_features_csv(tmp_path)
    row[column] = value
    make_customer_features_csv(tmp_path, rows=[row])

    with patch(
        f"{IMPORT_MODULE}.get_project_root",
        return_value=tmp_path,
    ):
        with pytest.raises(ValueError, match=message):
            import_customer_features()


def test_import_customer_features_success(tmp_path, capsys):
    """Insert customer and feature records and report row counts."""
    make_customer_features_csv(tmp_path)

    engine = MagicMock()
    connection = engine.begin.return_value.__enter__.return_value
    connection.execute.return_value.scalar_one.side_effect = [1, 1]

    with patch(
        f"{IMPORT_MODULE}.get_project_root",
        return_value=tmp_path,
    ), patch(
        f"{IMPORT_MODULE}.get_database_engine",
        return_value=engine,
    ):
        result = import_customer_features()

    assert result is None
    assert connection.execute.call_count == 4
    engine.dispose.assert_called_once()

    output = capsys.readouterr().out
    assert "CSV records processed: 1" in output
    assert "Customers in database: 1" in output
    assert "Customer feature records: 1" in output


def test_import_customer_features_disposes_engine_on_error(tmp_path):
    """Dispose of the engine if a database insert fails."""
    make_customer_features_csv(tmp_path)

    engine = MagicMock()
    connection = engine.begin.return_value.__enter__.return_value
    connection.execute.side_effect = RuntimeError("Database insert failed")

    with patch(
        f"{IMPORT_MODULE}.get_project_root",
        return_value=tmp_path,
    ), patch(
        f"{IMPORT_MODULE}.get_database_engine",
        return_value=engine,
    ):
        with pytest.raises(RuntimeError, match="Database insert failed"):
            import_customer_features()

    engine.dispose.assert_called_once()
