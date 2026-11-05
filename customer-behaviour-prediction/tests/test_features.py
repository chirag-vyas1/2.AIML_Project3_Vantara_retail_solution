import warnings
import pandas as pd
import pytest
import numpy as np
from src.features.features import create_rfm_features
from src.features.features import validate_rfm_features
from src.features.features import create_rfm_features
from src.features.features import create_lstm_sequences
from src.features.features import create_lstm_sequences
from src.features.features import select_autoencoder_features


@pytest.fixture
def sample_transactions():
    """Create a small, predictable transaction dataset."""
    return pd.DataFrame(
        {
            "Customer ID": ["C001", "C001", "C001", "C002", "C002"],
            "Invoice": ["I001", "I002", "I002", "I003", "I004"],
            "InvoiceDate": [
                "2026-01-01",
                "2026-01-05",
                "2026-01-05",
                "2026-01-03",
                "2026-01-20",
            ],
            "Transaction Type": ["Purchase"] * 5,
            "Customer ID Missing Flag": [0] * 5,
            "Include In Sales Analysis": [True] * 5,
            "Analytical Transaction Value": [
                100.0, 200.0, 50.0, 300.0, 500.0
            ],
        }
    )

#=====================================================================
#---------------------- 1.testing of rmf features --------------------
#=====================================================================
def test_create_rfm_features(sample_transactions):
    """Test RFM calculations, filtering, input preservation, and errors."""
    cutoff = pd.Timestamp("2026-01-10")
    original = sample_transactions.copy(deep=True)

    # 1. Verify RFM calculations and cutoff-date filtering.
    result = create_rfm_features(
        sample_transactions, cutoff_date=cutoff
    )
    result = result.set_index("Customer ID")

    assert result.loc["C001", "Recency"] == 5.0
    assert result.loc["C001", "Frequency"] == 2
    assert result.loc["C001", "Total_Monetary_Value"] == 350.0
    assert result.loc["C001", "Average_Order_Spend"] == 175.0

    assert result.loc["C002", "Recency"] == 7.0
    assert result.loc["C002", "Frequency"] == 1
    assert result.loc["C002", "Total_Monetary_Value"] == 300.0
    assert result.loc["C002", "Average_Order_Spend"] == 300.0

    # 2. Verify one output row per customer.
    assert result.index.is_unique
    assert set(result.index) == {"C001", "C002"}

    # 3. Verify that the input DataFrame remains unchanged.
    pd.testing.assert_frame_equal(sample_transactions, original)

    # 4. Verify that missing required columns raise ValueError.
    incomplete = sample_transactions.drop(columns=["Invoice"])

    with pytest.raises(ValueError, match="Missing required columns"):
        create_rfm_features(incomplete, cutoff_date=cutoff)

    # 5. Verify that no qualifying purchases raise ValueError.
    future_transactions = sample_transactions.copy()
    future_transactions["InvoiceDate"] = (
        pd.to_datetime(future_transactions["InvoiceDate"])
        + pd.Timedelta(days=100)
    )

    with pytest.raises(
        ValueError, match="No qualifying purchase transactions"
    ):
        create_rfm_features(future_transactions, cutoff_date=cutoff)

    # 6. Verify that returns and excluded transactions do not affect RFM.
    extra_rows = pd.DataFrame(
        {
            "Customer ID": ["C001", "C001", "C001"],
            "Invoice": ["I005", "I006", "I007"],
            "InvoiceDate": [
                "2026-01-06",
                "2026-01-07",
                "2026-01-08",
            ],
            "Transaction Type": ["Return", "Purchase", "Purchase"],
            "Customer ID Missing Flag": [0, 1, 0],
            "Include In Sales Analysis": [True, True, False],
            "Analytical Transaction Value": [-50.0, 400.0, 600.0],
        }
    )

    combined = pd.concat(
        [sample_transactions, extra_rows], ignore_index=True
    )
    filtered_result = create_rfm_features(
        combined, cutoff_date=cutoff
    )
    c001 = filtered_result.loc[
        filtered_result["Customer ID"] == "C001"
    ].iloc[0]

    assert c001["Frequency"] == 2
    assert c001["Total_Monetary_Value"] == 350.0

def test_validate_rfm_features():
    """Test valid RFM data and the principal validation failures."""
    import pandas as pd

    # 1. Valid customer-level RFM data
    valid_df = pd.DataFrame(
        {
            "Customer ID": [101, 102, 103],
            "Recency": [5, 10, 15],
            "Frequency": [2, 4, 5],
            "Total_Monetary_Value": [100.0, 200.0, 250.0],
            "Average_Order_Spend": [50.0, 50.0, 50.0],
        }
    )

    result = validate_rfm_features(valid_df)

    assert result["passed"] is True
    assert result["required_columns_present"] is True
    assert result["customer_count"] == 3
    assert result["has_customers"] is True
    assert result["missing_customer_ids"] == 0
    assert result["duplicate_customer_ids"] == 0
    assert result["unique_customer_ids"] == 3
    assert result["customer_ids_valid"] is True
    assert result["recency_valid"] is True
    assert result["frequency_valid"] is True
    assert result["total_monetary_value_valid"] is True
    assert result["average_order_spend_valid"] is True
    assert result["monetary_consistency_valid"] is True

    # 2. Missing required column returns early
    missing_column_df = valid_df.drop(columns=["Frequency"])
    result = validate_rfm_features(missing_column_df)

    assert result["passed"] is False
    assert result["required_columns_present"] is False
    assert "Frequency" in result["missing_columns"]
    assert "customer_count" not in result

    # 3. Duplicate and missing customer IDs
    invalid_ids_df = valid_df.copy()
    invalid_ids_df.loc[1, "Customer ID"] = 101
    invalid_ids_df.loc[2, "Customer ID"] = None

    result = validate_rfm_features(invalid_ids_df)

    assert result["missing_customer_ids"] == 1
    assert result["duplicate_customer_ids"] == 1
    assert result["customer_ids_valid"] is False
    assert result["passed"] is False

    # 4. Invalid recency, frequency and monetary values
    invalid_values_df = valid_df.copy()
    invalid_values_df.loc[0, "Recency"] = -1
    invalid_values_df.loc[1, "Frequency"] = 0
    invalid_values_df.loc[2, "Total_Monetary_Value"] = 0
    invalid_values_df.loc[0, "Average_Order_Spend"] = 0

    result = validate_rfm_features(invalid_values_df)

    assert result["negative_recency"] == 1
    assert result["recency_valid"] is False
    assert result["non_positive_frequency"] == 1
    assert result["frequency_valid"] is False
    assert result["non_positive_total_monetary_value"] == 1
    assert result["total_monetary_value_valid"] is False
    assert result["non_positive_average_order_spend"] == 1
    assert result["average_order_spend_valid"] is False
    assert result["passed"] is False

    # 5. Inconsistent average order spend
    inconsistent_df = valid_df.copy()
    inconsistent_df.loc[0, "Average_Order_Spend"] = 40.0

    result = validate_rfm_features(inconsistent_df)

    assert result["monetary_consistency_valid"] is False
    assert result["maximum_monetary_difference"] == 10.0
    assert result["passed"] is False

    # 6. Empty dataset
    empty_df = valid_df.iloc[0:0].copy()
    result = validate_rfm_features(empty_df)

    assert result["customer_count"] == 0
    assert result["has_customers"] is False
    assert result["passed"] is False

    # 7. Missing numeric values
    missing_values_df = valid_df.copy()
    missing_values_df.loc[0, "Recency"] = float("nan")
    missing_values_df.loc[1, "Frequency"] = float("nan")
    missing_values_df.loc[2, "Total_Monetary_Value"] = float("nan")
    missing_values_df.loc[0, "Average_Order_Spend"] = float("nan")

    result = validate_rfm_features(missing_values_df)

    assert result["missing_recency"] == 1
    assert result["missing_frequency"] == 1
    assert result["missing_total_monetary_value"] == 1
    assert result["missing_average_order_spend"] == 1
    assert result["passed"] is False

def test_analyze_rfm_basis():
    """
    Test analyze_rfm_basis() for transaction filtering,
    customer-level RFM aggregation, recency, monetary values,
    and summary statistics.
    """
    import pandas as pd
    import pytest

    from src.features.features import analyze_rfm_basis

    # ---------------------------------------------------------
    # 1. Prepare sample transaction data
    # ---------------------------------------------------------
    cutoff_date = pd.Timestamp("2025-01-31 23:59:59")

    df = pd.DataFrame(
        {
            "Invoice": [
                "INV001",
                "INV002",
                "INV003",
                "INV004",
                "INV005",
                "INV006",
                "INV007",
                "INV008",
                "INV009",
            ],
            "Customer ID": [
                1001,
                1001,
                1002,
                1002,
                1003,
                1003,
                1004,
                1005,
                1001,
            ],
            "InvoiceDate": [
                "2025-01-01",
                "2025-01-20",
                "2025-01-10",
                "2025-01-25",
                "2025-01-15",
                "2025-02-05",
                "2025-01-12",
                "2025-01-18",
                "2025-01-22",
            ],
            "Transaction Type": [
                "Purchase",
                "Purchase",
                "Purchase",
                "Purchase",
                "Purchase",
                "Purchase",
                "Return",
                "Purchase",
                "Purchase",
            ],
            "Customer ID Missing Flag": [
                0, 0, 0, 0, 0, 0, 0, 1, 0
            ],
            "Include In Sales Analysis": [
                True, True, True, True, True, True, True, True, False
            ],
            "Analytical Transaction Value": [
                100.00,
                200.00,
                150.00,
                250.00,
                300.00,
                500.00,
                100.00,
                400.00,
                999.00,
            ],
        }
    )

    original_df = df.copy(deep=True)

    # ---------------------------------------------------------
    # 2. Execute RFM analysis
    # ---------------------------------------------------------
    result = analyze_rfm_basis(
        df=df,
        cutoff_date=cutoff_date,
    )

    # ---------------------------------------------------------
    # 3. Validate returned dictionary
    # ---------------------------------------------------------
    assert isinstance(result, dict)

    assert {
        "summary",
        "customer_analysis",
        "rfm_transactions",
    }.issubset(result.keys())

    summary = result["summary"]
    customer_analysis = result["customer_analysis"]
    rfm_transactions = result["rfm_transactions"]

    assert isinstance(summary, dict)
    assert isinstance(customer_analysis, pd.DataFrame)
    assert isinstance(rfm_transactions, pd.DataFrame)

    # ---------------------------------------------------------
    # 4. Validate qualifying transaction filtering
    # ---------------------------------------------------------
    # Expected qualifying transactions:
    # INV001, INV002, INV003, INV004, INV005
    assert len(rfm_transactions) == 5

    assert set(rfm_transactions["Invoice"]) == {
        "INV001",
        "INV002",
        "INV003",
        "INV004",
        "INV005",
    }

    assert (rfm_transactions["InvoiceDate"] <= cutoff_date).all()
    assert (rfm_transactions["Transaction Type"] == "Purchase").all()
    assert (rfm_transactions["Customer ID Missing Flag"] == 0).all()
    assert rfm_transactions["Include In Sales Analysis"].all()

    # ---------------------------------------------------------
    # 5. Validate customer-level aggregation
    # ---------------------------------------------------------
    assert len(customer_analysis) == 3

    assert set(customer_analysis["Customer ID"]) == {
        1001,
        1002,
        1003,
    }

    customer_1001 = customer_analysis.loc[
        customer_analysis["Customer ID"] == 1001
    ].iloc[0]

    assert customer_1001["Frequency"] == 2
    assert customer_1001["Total_Monetary_Value"] == pytest.approx(300.00)
    assert customer_1001["Average_Order_Spend"] == pytest.approx(150.00)

    customer_1002 = customer_analysis.loc[
        customer_analysis["Customer ID"] == 1002
    ].iloc[0]

    assert customer_1002["Frequency"] == 2
    assert customer_1002["Total_Monetary_Value"] == pytest.approx(400.00)
    assert customer_1002["Average_Order_Spend"] == pytest.approx(200.00)

    customer_1003 = customer_analysis.loc[
        customer_analysis["Customer ID"] == 1003
    ].iloc[0]

    assert customer_1003["Frequency"] == 1
    assert customer_1003["Total_Monetary_Value"] == pytest.approx(300.00)
    assert customer_1003["Average_Order_Spend"] == pytest.approx(300.00)

    # ---------------------------------------------------------
    # 6. Validate recency calculation
    # ---------------------------------------------------------
    expected_recency_1001 = (
        cutoff_date - pd.Timestamp("2025-01-20")
    ).total_seconds() / 86400

    expected_recency_1002 = (
        cutoff_date - pd.Timestamp("2025-01-25")
    ).total_seconds() / 86400

    expected_recency_1003 = (
        cutoff_date - pd.Timestamp("2025-01-15")
    ).total_seconds() / 86400

    assert customer_1001["Recency"] == pytest.approx(
        round(expected_recency_1001, 2)
    )

    assert customer_1002["Recency"] == pytest.approx(
        round(expected_recency_1002, 2)
    )

    assert customer_1003["Recency"] == pytest.approx(
        round(expected_recency_1003, 2)
    )

    # ---------------------------------------------------------
    # 7. Validate summary statistics
    # ---------------------------------------------------------
    assert summary["cutoff_date"] == cutoff_date
    assert summary["historical_transaction_count"] == 5
    assert summary["customer_count"] == 3
    assert summary["distinct_invoice_count"] == 5

    assert summary["total_monetary_value"] == pytest.approx(1000.00)
    assert summary["average_order_spend"] == pytest.approx(200.00)

    assert summary["minimum_recency"] == pytest.approx(
        customer_analysis["Recency"].min()
    )

    assert summary["maximum_recency"] == pytest.approx(
        customer_analysis["Recency"].max()
    )

    assert summary["median_recency"] == pytest.approx(
        customer_analysis["Recency"].median()
    )

    assert summary["minimum_frequency"] == 1
    assert summary["maximum_frequency"] == 2

    # ---------------------------------------------------------
    # 8. Verify original DataFrame remains unchanged
    # ---------------------------------------------------------
    pd.testing.assert_frame_equal(df, original_df)


#========================================================================
#-------------- 2.testing of create_clv_feature function -----------------
#========================================================================
from src.features.features import create_clv_features
from src.features.features import validate_clv_features
@pytest.fixture
def sample_rfm_features():
    """Provide customer-level RFM features for CLV testing."""
    return pd.DataFrame(
        {
            "Customer ID": ["C001", "C002"],
            "Frequency": [2, 1],
            "Total_Monetary_Value": [350.0, 300.0],
        }
    )

def test_create_clv_features(sample_transactions, sample_rfm_features):
    """Test CLV calculations, filtering, date handling, and errors."""
    cutoff = pd.Timestamp("2026-01-10")

    original_transactions = sample_transactions.copy(deep=True)
    original_rfm = sample_rfm_features.copy(deep=True)

    # 1. Test successful CLV feature creation.
    result = create_clv_features(
        sample_transactions,
        sample_rfm_features,
        cutoff_date=cutoff,
    )

    assert len(result) == 2
    assert result["Customer ID"].is_unique
    assert set(result["Customer ID"]) == {"C001", "C002"}

    c001 = result.loc[result["Customer ID"] == "C001"].iloc[0]
    assert c001["Historical_CLV"] == 350.0
    assert c001["Customer_Lifetime_Days"] == 4.0
    assert c001["Historical_Average_Order_Value"] == 175.0
    assert c001["Historical_Invoice_Count"] == 2
    assert c001["First_Purchase_Date"] == pd.Timestamp("2026-01-01")
    assert c001["Last_Purchase_Date"] == pd.Timestamp("2026-01-05")

    c002 = result.loc[result["Customer ID"] == "C002"].iloc[0]
    assert c002["Historical_CLV"] == 300.0
    assert c002["Customer_Lifetime_Days"] == 0.0
    assert c002["Historical_Average_Order_Value"] == 300.0
    assert c002["Historical_Invoice_Count"] == 1

    # 2. Verify future transactions do not affect historical customer data.
    assert c002["Last_Purchase_Date"] == pd.Timestamp("2026-01-03")

    # 3. Verify inputs remain unchanged.
    pd.testing.assert_frame_equal(sample_transactions, original_transactions)
    pd.testing.assert_frame_equal(sample_rfm_features, original_rfm)

    # 4. Verify missing RFM columns raise ValueError.
    incomplete_rfm = sample_rfm_features.drop(
        columns=["Frequency"]
    )

    with pytest.raises(ValueError, match="Missing required RFM columns"):
        create_clv_features(
            sample_transactions,
            incomplete_rfm,
            cutoff_date=cutoff,
        )

    # 5. Verify missing transaction columns raise ValueError.
    incomplete_transactions = sample_transactions.drop(
        columns=["Invoice"]
    )

    with pytest.raises(ValueError, match="Missing required transaction columns"):
        create_clv_features(
            incomplete_transactions,
            sample_rfm_features,
            cutoff_date=cutoff,
        )

    # 6. Verify a cutoff after the maximum transaction date is rejected.
    with pytest.raises(ValueError, match="Cutoff date cannot be later"):
        create_clv_features(
            sample_transactions,
            sample_rfm_features,
            cutoff_date=pd.Timestamp("2027-01-01"),
        )

    # 7. Verify no qualifying historical transactions raise ValueError.
    future_transactions = sample_transactions.copy()
    future_transactions["InvoiceDate"] = (
        pd.to_datetime(future_transactions["InvoiceDate"])
        + pd.Timedelta(days=100)
    )

    with pytest.raises(
        ValueError, match="No qualifying historical transactions"
    ):
        create_clv_features(
            future_transactions,
            sample_rfm_features,
            cutoff_date=cutoff,
        )

    6

def test_validate_clv_features():
    """Test CLV validation rules and failure scenarios."""
    import pandas as pd
    import warnings

    # 1. Valid customer-level CLV data
    valid_df = pd.DataFrame(
        {
            "Customer ID": [101, 102, 103],
            "Historical_CLV": [100.0, 200.0, 300.0],
            "Customer_Lifetime_Days": [10.0, 20.0, 30.0],
            "Historical_Average_Order_Value": [50.0, 50.0, 60.0],
            "First_Purchase_Date": pd.to_datetime(
                ["2025-01-01", "2025-01-01", "2025-01-01"]
            ),
            "Last_Purchase_Date": pd.to_datetime(
                ["2025-01-11", "2025-01-21", "2025-01-31"]
            ),
            "Historical_Invoice_Count": [2, 4, 5],
        }
    )
    with warnings.catch_warnings():
         warnings.filterwarnings(
                 "ignore",
                   message="Could not infer format.*",
                category=UserWarning,
               )

         result = validate_clv_features(valid_df)

    assert result["passed"] is True
    assert result["required_columns_present"] is True
    assert result["customer_count"] == 3
    assert result["has_customers"] is True
    assert result["missing_customer_ids"] == 0
    assert result["duplicate_customer_ids"] == 0
    assert result["unique_customer_ids"] == 3
    assert result["customer_ids_valid"] is True
    assert result["historical_clv_valid"] is True
    assert result["customer_lifetime_valid"] is True
    assert result["purchase_dates_valid"] is True
    assert result["historical_invoice_count_valid"] is True
    assert result["average_order_value_valid"] is True
    assert result["average_order_value_consistency_valid"] is True
    assert result["customer_lifetime_consistency_valid"] is True

    # 2. Missing required column
    missing_column_df = valid_df.drop(
        columns=["Historical_Invoice_Count"]
    )
    result = validate_clv_features(missing_column_df)

    assert result["passed"] is False
    assert result["required_columns_present"] is False
    assert "Historical_Invoice_Count" in result["missing_columns"]
    assert "customer_count" not in result

    # 3. Missing and duplicate customer IDs
    invalid_ids_df = valid_df.copy()
    invalid_ids_df.loc[1, "Customer ID"] = 101
    invalid_ids_df.loc[2, "Customer ID"] = None

    result = validate_clv_features(invalid_ids_df)

    assert result["missing_customer_ids"] == 1
    assert result["duplicate_customer_ids"] == 1
    assert result["customer_ids_valid"] is False
    assert result["passed"] is False

    # 4. Invalid CLV, lifetime, invoice count, and average value
    invalid_values_df = valid_df.copy()
    invalid_values_df.loc[0, "Historical_CLV"] = 0
    invalid_values_df.loc[1, "Customer_Lifetime_Days"] = -1
    invalid_values_df.loc[2, "Historical_Invoice_Count"] = 0
    invalid_values_df.loc[0, "Historical_Average_Order_Value"] = 0

    result = validate_clv_features(invalid_values_df)

    assert result["non_positive_historical_clv"] == 1
    assert result["historical_clv_valid"] is False
    assert result["negative_customer_lifetime"] == 1
    assert result["customer_lifetime_valid"] is False
    assert result["non_positive_historical_invoice_count"] == 1
    assert result["historical_invoice_count_valid"] is False
    assert result["non_positive_average_order_value"] == 1
    assert result["average_order_value_valid"] is False
    assert result["passed"] is False

    # 5. Missing numeric values
    missing_values_df = valid_df.copy()
    missing_values_df.loc[0, "Historical_CLV"] = float("nan")
    missing_values_df.loc[1, "Customer_Lifetime_Days"] = float("nan")
    missing_values_df.loc[2, "Historical_Invoice_Count"] = float("nan")
    missing_values_df.loc[0, "Historical_Average_Order_Value"] = float("nan")

    result = validate_clv_features(missing_values_df)

    assert result["missing_historical_clv"] == 1
    assert result["missing_customer_lifetime"] == 1
    assert result["missing_historical_invoice_count"] == 1
    assert result["missing_average_order_value"] == 1
    assert result["passed"] is False

    # 6. Missing purchase dates
    missing_dates_df = valid_df.copy()
    missing_dates_df.loc[0, "First_Purchase_Date"] = pd.NaT
    missing_dates_df.loc[1, "Last_Purchase_Date"] = pd.NaT

    result = validate_clv_features(missing_dates_df)

    assert result["missing_first_purchase_date"] == 1
    assert result["missing_last_purchase_date"] == 1
    assert result["purchase_dates_valid"] is False
    assert result["passed"] is False

    # 7. Last purchase date precedes first purchase date
    invalid_dates_df = valid_df.copy()
    invalid_dates_df.loc[0, "Last_Purchase_Date"] = pd.Timestamp(
        "2024-12-31"
    )

    result = validate_clv_features(invalid_dates_df)

    assert result["invalid_purchase_date_order"] == 1
    assert result["purchase_dates_valid"] is False
    assert result["passed"] is False

    # 8. Inconsistent historical average order value
    inconsistent_aov_df = valid_df.copy()
    inconsistent_aov_df.loc[0, "Historical_Average_Order_Value"] = 40.0

    result = validate_clv_features(inconsistent_aov_df)

    assert result["average_order_value_consistency_valid"] is False
    assert result["maximum_average_order_difference"] == 10.0
    assert result["passed"] is False

    # 9. Inconsistent customer lifetime
    inconsistent_lifetime_df = valid_df.copy()
    inconsistent_lifetime_df.loc[0, "Customer_Lifetime_Days"] = 5.0

    result = validate_clv_features(inconsistent_lifetime_df)

    assert result["customer_lifetime_consistency_valid"] is False
    assert result["maximum_lifetime_difference"] == 5.0
    assert result["passed"] is False

    # 10. Empty dataset
    empty_df = valid_df.iloc[0:0].copy()
    result = validate_clv_features(empty_df)

    assert result["customer_count"] == 0
    assert result["has_customers"] is False
    assert result["passed"] is False

def test_analyze_clv_basis():
    """
    Test analyze_clv_basis() for historical CLV calculation,
    cutoff handling, customer lifetime, RFM consistency,
    transaction filtering, and input validation.
    """
    import pandas as pd
    import pytest
    import warnings

    from src.features.features import analyze_clv_basis

    # ---------------------------------------------------------
    # Helper: Run function while suppressing only date warning
    # ---------------------------------------------------------
    def run_analysis(**kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Could not infer format.*",
                category=UserWarning,
            )
            return analyze_clv_basis(**kwargs)

    # ---------------------------------------------------------
    # 1. Prepare transaction-level sample data
    # ---------------------------------------------------------
    cutoff_date = pd.Timestamp("2025-01-31 23:59:59")

    df_cleaned = pd.DataFrame({
        "Invoice": [
            "INV001", "INV002", "INV003",
            "INV004", "INV005", "INV006",
            "INV007", "INV008", "INV009",
        ],
        "Customer ID": [
            1001, 1001, 1002, 1002, 1003,
            1001, 1004, 1005, 1001,
        ],
        "InvoiceDate": [
            "2025-01-01", "2025-01-20", "2025-01-10",
            "2025-01-25", "2025-01-15", "2025-05-01",
            "2025-01-12", "2025-01-18", "2025-01-22",
        ],
        "Transaction Type": [
            "Purchase", "Purchase", "Purchase",
            "Purchase", "Purchase", "Purchase",
            "Return", "Purchase", "Purchase",
        ],
        "Customer ID Missing Flag": [
            0, 0, 0, 0, 0, 0, 0, 1, 0,
        ],
        "Include In Sales Analysis": [
            True, True, True, True, True,
            True, True, True, False,
        ],
        "Analytical Transaction Value": [
            100.00, 200.00, 150.00,
            250.00, 300.00, 500.00,
            100.00, 400.00, 999.00,
        ],
    })

    # ---------------------------------------------------------
    # 2. Prepare corresponding customer-level RFM data
    # ---------------------------------------------------------
    rfm_features = pd.DataFrame({
        "Customer ID": [1001, 1002, 1003],
        "Recency": [12.0, 6.0, 16.0],
        "Frequency": [2, 2, 1],
        "Total_Monetary_Value": [300.00, 400.00, 300.00],
        "Average_Order_Spend": [150.00, 200.00, 300.00],
    })

    original_transactions = df_cleaned.copy(deep=True)
    original_rfm = rfm_features.copy(deep=True)

    # ---------------------------------------------------------
    # 3. Test analysis with user-supplied cutoff date
    # ---------------------------------------------------------
    result = run_analysis(
        df_cleaned=df_cleaned,
        rfm_features=rfm_features,
        cutoff_date=cutoff_date,
    )

    assert isinstance(result, dict)

    assert {
        "summary",
        "distribution",
        "customer_analysis",
        "clv_transactions",
        "validation",
    }.issubset(result.keys())

    summary = result["summary"]
    distribution = result["distribution"]
    customer_analysis = result["customer_analysis"]
    clv_transactions = result["clv_transactions"]
    validation = result["validation"]

    assert isinstance(summary, dict)
    assert isinstance(distribution, pd.DataFrame)
    assert isinstance(customer_analysis, pd.DataFrame)
    assert isinstance(clv_transactions, pd.DataFrame)
    assert isinstance(validation, dict)

    # ---------------------------------------------------------
    # 4. Validate cutoff information
    # ---------------------------------------------------------
    assert summary["maximum_invoice_date"] == pd.Timestamp("2025-05-01")
    assert summary["cutoff_date"] == cutoff_date
    assert summary["cutoff_source"] == "User supplied"
    assert summary["prediction_window_days"] == 90

    # ---------------------------------------------------------
    # 5. Validate qualifying historical transactions
    # ---------------------------------------------------------
    expected_invoices = {
        "INV001", "INV002", "INV003", "INV004", "INV005",
    }

    assert len(clv_transactions) == 5
    assert set(clv_transactions["Invoice"]) == expected_invoices

    assert (clv_transactions["InvoiceDate"] <= cutoff_date).all()
    assert (clv_transactions["Transaction Type"] == "Purchase").all()
    assert (clv_transactions["Customer ID Missing Flag"] == 0).all()
    assert clv_transactions["Include In Sales Analysis"].all()

    # ---------------------------------------------------------
    # 6. Validate customer-level historical CLV
    # ---------------------------------------------------------
    assert len(customer_analysis) == 3

    assert set(customer_analysis["Customer ID"]) == {
        1001, 1002, 1003,
    }

    customer_1001 = customer_analysis.loc[
        customer_analysis["Customer ID"] == 1001
    ].iloc[0]

    assert customer_1001["Historical_CLV"] == pytest.approx(300.00)
    assert customer_1001["Historical_Average_Order_Value"] == pytest.approx(150.00)
    assert customer_1001["Historical_Invoice_Count"] == 2

    customer_1002 = customer_analysis.loc[
        customer_analysis["Customer ID"] == 1002
    ].iloc[0]

    assert customer_1002["Historical_CLV"] == pytest.approx(400.00)
    assert customer_1002["Historical_Average_Order_Value"] == pytest.approx(200.00)
    assert customer_1002["Historical_Invoice_Count"] == 2

    customer_1003 = customer_analysis.loc[
        customer_analysis["Customer ID"] == 1003
    ].iloc[0]

    assert customer_1003["Historical_CLV"] == pytest.approx(300.00)
    assert customer_1003["Historical_Average_Order_Value"] == pytest.approx(300.00)
    assert customer_1003["Historical_Invoice_Count"] == 1

    # ---------------------------------------------------------
    # 7. Validate first and last purchase dates
    # ---------------------------------------------------------
    assert customer_1001["First_Purchase_Date"] == pd.Timestamp("2025-01-01")
    assert customer_1001["Last_Purchase_Date"] == pd.Timestamp("2025-01-20")

    assert customer_1002["First_Purchase_Date"] == pd.Timestamp("2025-01-10")
    assert customer_1002["Last_Purchase_Date"] == pd.Timestamp("2025-01-25")

    assert customer_1003["First_Purchase_Date"] == pd.Timestamp("2025-01-15")
    assert customer_1003["Last_Purchase_Date"] == pd.Timestamp("2025-01-15")

    # ---------------------------------------------------------
    # 8. Validate customer lifetime duration
    # ---------------------------------------------------------
    assert customer_1001["Customer_Lifetime_Days"] == pytest.approx(19.0)
    assert customer_1002["Customer_Lifetime_Days"] == pytest.approx(15.0)
    assert customer_1003["Customer_Lifetime_Days"] == pytest.approx(0.0)

    # ---------------------------------------------------------
    # 9. Validate CLV distribution
    # ---------------------------------------------------------
    assert isinstance(distribution, pd.DataFrame)

    assert {
        "Historical_CLV",
        "Customer_Lifetime_Days",
        "Historical_Average_Order_Value",
    }.issubset(distribution.columns)

    assert distribution.loc["count", "Historical_CLV"] == 3
    assert distribution.loc["mean", "Historical_CLV"] == pytest.approx(1000 / 3)

    # ---------------------------------------------------------
    # 10. Validate summary statistics
    # ---------------------------------------------------------
    assert summary["historical_transaction_count"] == 5
    assert summary["customer_count"] == 3
    assert summary["distinct_invoice_count"] == 5

    assert summary["total_historical_revenue"] == pytest.approx(1000.00)

    assert summary["minimum_historical_clv"] == pytest.approx(300.00)
    assert summary["median_historical_clv"] == pytest.approx(300.00)
    assert summary["maximum_historical_clv"] == pytest.approx(400.00)

    assert summary["minimum_customer_lifetime_days"] == pytest.approx(0.0)
    assert summary["median_customer_lifetime_days"] == pytest.approx(15.0)
    assert summary["maximum_customer_lifetime_days"] == pytest.approx(19.0)

    assert summary["zero_lifetime_customers"] == 1
    assert summary["multiple_purchase_date_customers"] == 2

    assert summary["maximum_rfm_clv_difference"] == pytest.approx(0.0)
    assert summary["maximum_frequency_difference"] == 0
    assert summary["transactions_after_cutoff"] == 0

    # ---------------------------------------------------------
    # 11. Validate RFM-to-CLV consistency
    # ---------------------------------------------------------
    assert bool(validation["customer_population_valid"]) is True
    assert bool(validation["rfm_clv_monetary_consistency"]) is True
    assert bool(validation["rfm_clv_frequency_consistency"]) is True
    assert bool(validation["point_in_time_cutoff_valid"]) is True
    assert bool(validation["historical_clv_values_valid"]) is True
    assert bool(validation["passed"]) is True

    # ---------------------------------------------------------
    # 12. Test automatic cutoff calculation
    # ---------------------------------------------------------
    automatic_result = run_analysis(
        df_cleaned=df_cleaned,
        rfm_features=rfm_features,
        cutoff_date=None,
    )

    assert automatic_result["summary"]["cutoff_source"] == (
        "Automatically calculated"
    )

    assert automatic_result["summary"]["cutoff_date"] == (
        pd.Timestamp("2025-05-01") - pd.Timedelta(days=90)
    )

    assert automatic_result["summary"]["cutoff_date"] == pd.Timestamp(
        "2025-01-31 00:00:00"
    )

    assert len(automatic_result["clv_transactions"]) == 5

    # ---------------------------------------------------------
    # 13. Verify input DataFrames remain unchanged
    # ---------------------------------------------------------
    pd.testing.assert_frame_equal(df_cleaned, original_transactions)
    pd.testing.assert_frame_equal(rfm_features, original_rfm)

    # ---------------------------------------------------------
    # 14. Test missing required RFM columns
    # ---------------------------------------------------------
    invalid_rfm = rfm_features.drop(columns=["Frequency"])

    with pytest.raises(ValueError, match="Missing required RFM columns"):
        run_analysis(
            df_cleaned=df_cleaned,
            rfm_features=invalid_rfm,
            cutoff_date=cutoff_date,
        )

    # ---------------------------------------------------------
    # 15. Test missing required transaction columns
    # ---------------------------------------------------------
    invalid_transactions = df_cleaned.drop(
        columns=["Analytical Transaction Value"]
    )

    with pytest.raises(
        ValueError,
        match="Missing required transaction columns",
    ):
        run_analysis(
            df_cleaned=invalid_transactions,
            rfm_features=rfm_features,
            cutoff_date=cutoff_date,
        )

    # ---------------------------------------------------------
    # 16. Test invalid InvoiceDate values
    # ---------------------------------------------------------
    invalid_dates = df_cleaned.copy()
    invalid_dates["InvoiceDate"] = invalid_dates["InvoiceDate"].astype(object)
    invalid_dates.loc[0, "InvoiceDate"] = "invalid-date"

    with pytest.raises((ValueError, TypeError)):
        run_analysis(
            df_cleaned=invalid_dates,
            rfm_features=rfm_features,
            cutoff_date=cutoff_date,
        )

    # ---------------------------------------------------------
    # 17. Test dataset with no valid InvoiceDate
    # ---------------------------------------------------------
    no_valid_dates = df_cleaned.copy()
    no_valid_dates["InvoiceDate"] = pd.NaT

    with pytest.raises(ValueError, match="no valid dates"):
        run_analysis(
            df_cleaned=no_valid_dates,
            rfm_features=rfm_features,
        )

    # ---------------------------------------------------------
    # 18. Test cutoff later than maximum InvoiceDate
    # ---------------------------------------------------------
    with pytest.raises(ValueError, match="cannot be later"):
        run_analysis(
            df_cleaned=df_cleaned,
            rfm_features=rfm_features,
            cutoff_date=pd.Timestamp("2025-05-02"),
        )

    # ---------------------------------------------------------
    # 19. Test no qualifying historical transactions
    # ---------------------------------------------------------
    with pytest.raises(ValueError, match="No qualifying historical"):
        run_analysis(
            df_cleaned=df_cleaned,
            rfm_features=rfm_features,
            cutoff_date=pd.Timestamp("2024-12-31"),
        )



#================================================================================================
#------------------ 3. testing of Create_avg_basket_size_features -------------------------------
#================================================================================================

from src.features.features import create_avg_basket_size_features
from src.features.features import validate_avg_basket_size_features

@pytest.fixture
def sample_basket_transactions():
    """Create sample transactions with quantities for basket-size testing."""
    return pd.DataFrame(
        {
            "Customer ID": ["C001", "C001", "C001", "C002", "C002"],
            "Invoice": ["I001", "I001", "I002", "I003", "I004"],
            "InvoiceDate": [
                "2026-01-01",
                "2026-01-01",
                "2026-01-05",
                "2026-01-03",
                "2026-01-20",
            ],
            "Quantity": [2, 3, 5, 4, 10],
            "Transaction Type": ["Purchase"] * 5,
            "Customer ID Missing Flag": [0] * 5,
            "Include In Sales Analysis": [True] * 5,
        }
    )

def test_create_avg_basket_size_features(sample_basket_transactions):
    """Test basket-size calculations, filtering, and error handling."""
    cutoff = pd.Timestamp("2026-01-10")
    original = sample_basket_transactions.copy(deep=True)

    # 1. Test average basket-size calculations.
    result = create_avg_basket_size_features(
        sample_basket_transactions,
        cutoff_date=cutoff,
    )

    assert set(result["Customer ID"]) == {"C001", "C002"}
    assert result["Customer ID"].is_unique

    c001 = result.loc[result["Customer ID"] == "C001"].iloc[0]
    # C001: (2 + 3) items in I001 and 5 items in I002; 10 / 2 orders.
    assert c001["Average_Basket_Size"] == 5.0

    c002 = result.loc[result["Customer ID"] == "C002"].iloc[0]
    # C002: 4 items in I003; the January 20 order is excluded.
    assert c002["Average_Basket_Size"] == 4.0

    # 2. Verify that the input DataFrame remains unchanged.
    pd.testing.assert_frame_equal(sample_basket_transactions, original)

    # 3. Verify missing required columns raise ValueError.
    incomplete = sample_basket_transactions.drop(columns=["Quantity"])

    with pytest.raises(ValueError, match="Missing required columns"):
        create_avg_basket_size_features(incomplete, cutoff_date=cutoff)

    # 4. Verify a future cutoff date is rejected.
    with pytest.raises(ValueError, match="cutoff_date cannot be later"):
        create_avg_basket_size_features(
            sample_basket_transactions,
            cutoff_date=pd.Timestamp("2027-01-01"),
        )

    # 5. Verify no qualifying purchases raise ValueError.
    future_transactions = sample_basket_transactions.copy()
    future_transactions["InvoiceDate"] = (
        pd.to_datetime(future_transactions["InvoiceDate"])
        + pd.Timedelta(days=100)
    )

    with pytest.raises(ValueError, match="No qualifying purchase transactions"):
        create_avg_basket_size_features(
            future_transactions,
            cutoff_date=cutoff,
        )

    # 6. Verify returns and excluded transactions do not affect the result.
    extra_rows = pd.DataFrame(
        {
            "Customer ID": ["C001", "C001", "C001"],
            "Invoice": ["I005", "I006", "I007"],
            "InvoiceDate": [
                "2026-01-06",
                "2026-01-07",
                "2026-01-08",
            ],
            "Quantity": [100, 200, 300],
            "Transaction Type": ["Return", "Purchase", "Purchase"],
            "Customer ID Missing Flag": [0, 1, 0],
            "Include In Sales Analysis": [True, True, False],
        }
    )

    combined = pd.concat(
        [sample_basket_transactions, extra_rows],
        ignore_index=True,
    )

    filtered_result = create_avg_basket_size_features(
        combined,
        cutoff_date=cutoff,
    )

    filtered_c001 = filtered_result.loc[
        filtered_result["Customer ID"] == "C001"
    ].iloc[0]

    assert filtered_c001["Average_Basket_Size"] == 5.0

def test_validate_avg_basket_size_features():
    """Test Average Basket Size validation rules and failure scenarios."""
    import pandas as pd

    # 1. Valid customer-level data
    valid_df = pd.DataFrame(
        {
            "Customer ID": [101, 102, 103],
            "Average_Basket_Size": [3.5, 5.0, 2.25],
        }
    )

    result = validate_avg_basket_size_features(valid_df)

    assert result["passed"] is True
    assert result["required_columns_present"] is True
    assert result["missing_columns"] == []
    assert result["customer_count"] == 3
    assert result["has_customers"] is True
    assert result["missing_customer_ids"] == 0
    assert result["duplicate_customer_ids"] == 0
    assert result["unique_customer_ids"] == 3
    assert result["customer_ids_valid"] is True
    assert result["missing_average_basket_size"] == 0
    assert result["non_positive_average_basket_size"] == 0
    assert result["average_basket_size_valid"] is True

    # 2. Missing required column
    missing_column_df = valid_df.drop(
        columns=["Average_Basket_Size"]
    )
    result = validate_avg_basket_size_features(missing_column_df)

    assert result["passed"] is False
    assert result["required_columns_present"] is False
    assert "Average_Basket_Size" in result["missing_columns"]
    assert "customer_count" not in result

    # 3. Missing and duplicate customer IDs
    invalid_ids_df = valid_df.copy()
    invalid_ids_df.loc[1, "Customer ID"] = 101
    invalid_ids_df.loc[2, "Customer ID"] = None

    result = validate_avg_basket_size_features(invalid_ids_df)

    assert result["missing_customer_ids"] == 1
    assert result["duplicate_customer_ids"] == 1
    assert result["unique_customer_ids"] == 1
    assert result["customer_ids_valid"] is False
    assert result["passed"] is False

    # 4. Zero and negative basket sizes
    invalid_sizes_df = valid_df.copy()
    invalid_sizes_df.loc[0, "Average_Basket_Size"] = 0
    invalid_sizes_df.loc[1, "Average_Basket_Size"] = -2.0

    result = validate_avg_basket_size_features(invalid_sizes_df)

    assert result["non_positive_average_basket_size"] == 2
    assert result["average_basket_size_valid"] is False
    assert result["passed"] is False

    # 5. Missing basket-size values
    missing_sizes_df = valid_df.copy()
    missing_sizes_df.loc[0, "Average_Basket_Size"] = float("nan")

    result = validate_avg_basket_size_features(missing_sizes_df)

    assert result["missing_average_basket_size"] == 1
    assert result["average_basket_size_valid"] is False
    assert result["passed"] is False

    # 6. Empty dataset with the required columns
    empty_df = valid_df.iloc[0:0].copy()

    result = validate_avg_basket_size_features(empty_df)

    assert result["customer_count"] == 0
    assert result["has_customers"] is False
    assert result["customer_ids_valid"] is True
    assert result["average_basket_size_valid"] is True
    assert result["passed"] is False

def test_analyze_avg_basket_size_basis(sample_basket_transactions):
    """
    Test analyze_avg_basket_size_basis() for:
    invoice-level basket calculations, customer-level averages,
    cutoff handling, transaction filtering, validation,
    and input error handling.
    """

    import pandas as pd
    import pytest
    import warnings

    from src.features.features import analyze_avg_basket_size_basis

    # ---------------------------------------------------------
    # Helper: Suppress only pandas date-format warning
    # ---------------------------------------------------------
    def run_analysis(**kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Could not infer format.*",
                category=UserWarning,
            )
            return analyze_avg_basket_size_basis(**kwargs)

    # ---------------------------------------------------------
    # 1. Prepare sample transaction data
    # ---------------------------------------------------------
    df_cleaned = sample_basket_transactions.copy(deep=True)

    cutoff_date = pd.Timestamp("2026-01-20")

    original_transactions = df_cleaned.copy(deep=True)

    # ---------------------------------------------------------
    # 2. Test analysis with user-supplied cutoff
    # ---------------------------------------------------------
    result = run_analysis(
        df_cleaned=df_cleaned,
        cutoff_date=cutoff_date,
    )

    assert isinstance(result, dict)

    assert {
        "summary",
        "customer_analysis",
        "invoice_baskets",
        "transactions",
        "validation",
    }.issubset(result.keys())

    summary = result["summary"]
    customer_analysis = result["customer_analysis"]
    invoice_baskets = result["invoice_baskets"]
    transactions = result["transactions"]
    validation = result["validation"]

    assert isinstance(summary, dict)
    assert isinstance(customer_analysis, pd.DataFrame)
    assert isinstance(invoice_baskets, pd.DataFrame)
    assert isinstance(transactions, pd.DataFrame)
    assert isinstance(validation, dict)

    # ---------------------------------------------------------
    # 3. Validate cutoff information
    # ---------------------------------------------------------
    assert summary["maximum_invoice_date"] == pd.Timestamp(
        "2026-01-20"
    )

    assert summary["cutoff_date"] == cutoff_date
    assert summary["cutoff_source"] == "User supplied"
    assert summary["prediction_window_days"] == 90

    # ---------------------------------------------------------
    # 4. Validate qualifying transactions
    # ---------------------------------------------------------
    assert len(transactions) == 5

    assert (transactions["Transaction Type"] == "Purchase").all()
    assert (transactions["Customer ID Missing Flag"] == 0).all()
    assert transactions["Include In Sales Analysis"].all()
    assert (transactions["InvoiceDate"] <= cutoff_date).all()

    # ---------------------------------------------------------
    # 5. Validate invoice-level basket calculations
    # ---------------------------------------------------------
    assert len(invoice_baskets) == 4

    expected_baskets = {
        ("C001", "I001"): 5,
        ("C001", "I002"): 5,
        ("C002", "I003"): 4,
        ("C002", "I004"): 10,
    }

    actual_baskets = {
        (row["Customer ID"], row["Invoice"]): row["Basket_Size"]
        for _, row in invoice_baskets.iterrows()
    }

    assert actual_baskets == expected_baskets

    # Validate invoice dates
    assert (
        invoice_baskets.loc[
            invoice_baskets["Invoice"] == "I001",
            "Invoice_Date",
        ].iloc[0]
        == pd.Timestamp("2026-01-01")
    )

    # ---------------------------------------------------------
    # 6. Validate customer-level basket analysis
    # ---------------------------------------------------------
    assert len(customer_analysis) == 2

    assert set(customer_analysis["Customer ID"]) == {
        "C001",
        "C002",
    }

    customer_1 = customer_analysis.loc[
        customer_analysis["Customer ID"] == "C001"
    ].iloc[0]

    assert customer_1["Total_Qualifying_Quantity"] == 10
    assert customer_1["Number_of_Orders"] == 2
    assert customer_1["Minimum_Basket_Size"] == 5
    assert customer_1["Median_Basket_Size"] == 5
    assert customer_1["Maximum_Basket_Size"] == 5
    assert customer_1["Average_Basket_Size"] == pytest.approx(5.0)

    customer_2 = customer_analysis.loc[
        customer_analysis["Customer ID"] == "C002"
    ].iloc[0]

    assert customer_2["Total_Qualifying_Quantity"] == 14
    assert customer_2["Number_of_Orders"] == 2
    assert customer_2["Minimum_Basket_Size"] == 4
    assert customer_2["Median_Basket_Size"] == pytest.approx(7.0)
    assert customer_2["Maximum_Basket_Size"] == 10
    assert customer_2["Average_Basket_Size"] == pytest.approx(7.0)

    # ---------------------------------------------------------
    # 7. Validate overall summary statistics
    # ---------------------------------------------------------
    assert summary["historical_transaction_count"] == 5
    assert summary["customer_count"] == 2
    assert summary["distinct_invoice_count"] == 4

    assert summary["total_qualifying_quantity"] == pytest.approx(24.0)

    assert summary["minimum_average_basket_size"] == pytest.approx(5.0)
    assert summary["median_average_basket_size"] == pytest.approx(6.0)
    assert summary["maximum_average_basket_size"] == pytest.approx(7.0)

    assert summary["minimum_orders_per_customer"] == 2
    assert summary["maximum_orders_per_customer"] == 2

    # ---------------------------------------------------------
    # 8. Validate analysis conditions
    # ---------------------------------------------------------
    assert bool(validation["customer_count_valid"]) is True
    assert bool(validation["no_duplicate_customer_ids"]) is True
    assert bool(validation["basket_sizes_positive"]) is True
    assert bool(validation["average_basket_sizes_positive"]) is True
    assert bool(validation["orders_positive"]) is True
    assert bool(validation["cutoff_valid"]) is True
    assert bool(validation["passed"]) is True

    # ---------------------------------------------------------
    # 9. Test automatic cutoff calculation
    # ---------------------------------------------------------
    automatic_transactions = df_cleaned.copy(deep=True)

    # Add one historical transaction before the automatic cutoff
    automatic_transactions = pd.concat(
        [
            automatic_transactions,
            pd.DataFrame({
                "Customer ID": ["C003"],
                "Invoice": ["I005"],
                "InvoiceDate": ["2025-10-01"],
                "Quantity": [6],
                "Transaction Type": ["Purchase"],
                "Customer ID Missing Flag": [0],
                "Include In Sales Analysis": [True],
            }),
        ],
        ignore_index=True,
    )

    automatic_result = run_analysis(
        df_cleaned=automatic_transactions,
        cutoff_date=None,
    )

    automatic_cutoff = (
        pd.Timestamp("2026-01-20") - pd.Timedelta(days=90)
    )

    assert automatic_result["summary"]["cutoff_source"] == (
        "Automatically calculated"
    )

    assert automatic_result["summary"]["cutoff_date"] == automatic_cutoff

    assert len(automatic_result["transactions"]) == 1

    assert set(
        automatic_result["transactions"]["Invoice"]
    ) == {"I005"}

    assert automatic_result["summary"]["customer_count"] == 1
    assert automatic_result["summary"]["total_qualifying_quantity"] == 6.0

    # ---------------------------------------------------------
    # 10. Verify input DataFrame remains unchanged
    # ---------------------------------------------------------
    pd.testing.assert_frame_equal(
        df_cleaned,
        original_transactions,
    )

    # ---------------------------------------------------------
    # 11. Test missing required columns
    # ---------------------------------------------------------
    invalid_transactions = df_cleaned.drop(
        columns=["Quantity"]
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        run_analysis(
            df_cleaned=invalid_transactions,
            cutoff_date=cutoff_date,
        )

    # ---------------------------------------------------------
    # 12. Test invalid InvoiceDate values
    # ---------------------------------------------------------
    invalid_dates = df_cleaned.copy()

    invalid_dates["InvoiceDate"] = (
        invalid_dates["InvoiceDate"].astype(object)
    )

    invalid_dates.loc[0, "InvoiceDate"] = "invalid-date"

    with pytest.raises((ValueError, TypeError)):
        run_analysis(
            df_cleaned=invalid_dates,
            cutoff_date=cutoff_date,
        )

    # ---------------------------------------------------------
    # 13. Test dataset with no valid InvoiceDate
    # ---------------------------------------------------------
    no_valid_dates = df_cleaned.copy()
    no_valid_dates["InvoiceDate"] = pd.NaT

    with pytest.raises(
        ValueError,
        match="no valid dates",
    ):
        run_analysis(
            df_cleaned=no_valid_dates,
        )

    # ---------------------------------------------------------
    # 14. Test cutoff later than maximum InvoiceDate
    # ---------------------------------------------------------
    with pytest.raises(
        ValueError,
        match="cannot be later",
    ):
        run_analysis(
            df_cleaned=df_cleaned,
            cutoff_date=pd.Timestamp("2026-01-21"),
        )

    # ---------------------------------------------------------
    # 15. Test no qualifying purchase transactions
    # ---------------------------------------------------------
    with pytest.raises(
        ValueError,
        match="No qualifying purchase transactions",
    ):
        run_analysis(
            df_cleaned=df_cleaned,
            cutoff_date=pd.Timestamp("2025-12-31"),
        )

#================================================================================================
#----------- 4. testing of test_create_purchase_frequency_trend_features ------------------------
#================================================================================================

from src.features.features import create_purchase_frequency_trend_features
from src.features.features import validate_purchase_frequency_trend_features

@pytest.fixture
def sample_frequency_transactions():
    """Create transactions spanning multiple months for trend testing."""
    return pd.DataFrame(
        {
            "Customer ID": [
                "C001", "C001", "C001", "C001",
                "C002", "C002",
            ],
            "Invoice": [
                "I001", "I002", "I003", "I004",
                "I005", "I006",
            ],
            "InvoiceDate": [
                "2026-01-05",
                "2026-02-05",
                "2026-02-15",
                "2026-04-05",
                "2026-03-10",
                "2026-04-10",
            ],
            "Transaction Type": ["Purchase"] * 6,
            "Customer ID Missing Flag": [0] * 6,
            "Include In Sales Analysis": [True] * 6,
        }
    )

def test_create_purchase_frequency_trend_features(
    sample_frequency_transactions,
):
    """Test frequency trends, filtering, cutoffs, and errors."""
    cutoff = pd.Timestamp("2026-04-10")
    original = sample_frequency_transactions.copy(deep=True)

    # 1. Test successful output and customer uniqueness.
    result = create_purchase_frequency_trend_features(
        sample_frequency_transactions,
        cutoff_date=cutoff,
    )

    assert set(result["Customer ID"]) == {"C001", "C002"}
    assert result["Customer ID"].is_unique
    assert "Purchase_Frequency_Trend" in result.columns

    # C001 monthly frequencies: Jan=1, Feb=2, Mar=0, Apr=1.
    # Its slope across the four monthly observations is 0.0.
    c001 = result.loc[result["Customer ID"] == "C001"].iloc[0]
    assert c001["Purchase_Frequency_Trend"] == -0.2

    # C002 has purchases in two months: Mar=1, Apr=1.
    # The trend is stable, so the slope is 0.0.
    c002 = result.loc[result["Customer ID"] == "C002"].iloc[0]
    assert c002["Purchase_Frequency_Trend"] == 0.0

    # 2. Verify input data is not modified.
    pd.testing.assert_frame_equal(
        sample_frequency_transactions,
        original,
    )

    # 3. Verify missing required columns raise ValueError.
    incomplete = sample_frequency_transactions.drop(
        columns=["Invoice"]
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        create_purchase_frequency_trend_features(
            incomplete,
            cutoff_date=cutoff,
        )

    # 4. Verify cutoff dates after the maximum date are rejected.
    with pytest.raises(ValueError, match="cutoff_date cannot be later"):
        create_purchase_frequency_trend_features(
            sample_frequency_transactions,
            cutoff_date=pd.Timestamp("2027-01-01"),
        )

    # 5. Verify no qualifying transactions raise ValueError.
    future_transactions = sample_frequency_transactions.copy()
    future_transactions["InvoiceDate"] = (
        pd.to_datetime(future_transactions["InvoiceDate"])
        + pd.Timedelta(days=100)
    )

    with pytest.raises(
        ValueError,
        match="No qualifying purchase transactions",
    ):
        create_purchase_frequency_trend_features(
            future_transactions,
            cutoff_date=cutoff,
        )

def test_validate_purchase_frequency_trend_features():
    """Test purchase-frequency trend validation rules."""
    import numpy as np
    import pandas as pd

    # 1. Valid customer-level data
    valid_df = pd.DataFrame(
        {
            "Customer ID": [101, 102, 103],
            "Purchase_Frequency_Trend": [-0.5, 0.0, 1.25],
        }
    )

    result = validate_purchase_frequency_trend_features(valid_df)

    assert result["passed"] is True
    assert result["required_columns_present"] is True
    assert result["missing_columns"] == []
    assert result["customer_count"] == 3
    assert result["has_customers"] is True
    assert result["missing_customer_ids"] == 0
    assert result["duplicate_customer_ids"] == 0
    assert result["unique_customer_ids"] == 3
    assert result["customer_ids_valid"] is True
    assert result["missing_trend_values"] == 0
    assert result["infinite_trend_values"] == 0
    assert result["trend_values_valid"] is True

    # 2. Missing required column
    missing_column_df = valid_df.drop(
        columns=["Purchase_Frequency_Trend"]
    )
    result = validate_purchase_frequency_trend_features(
        missing_column_df
    )

    assert result["passed"] is False
    assert result["required_columns_present"] is False
    assert "Purchase_Frequency_Trend" in result["missing_columns"]
    assert "customer_count" not in result

    # 3. Missing and duplicate customer IDs
    invalid_ids_df = valid_df.copy()
    invalid_ids_df.loc[1, "Customer ID"] = 101
    invalid_ids_df.loc[2, "Customer ID"] = None

    result = validate_purchase_frequency_trend_features(
        invalid_ids_df
    )

    assert result["missing_customer_ids"] == 1
    assert result["duplicate_customer_ids"] == 1
    assert result["unique_customer_ids"] == 1
    assert result["customer_ids_valid"] is False
    assert result["passed"] is False

    # 4. Missing trend values
    missing_trend_df = valid_df.copy()
    missing_trend_df.loc[0, "Purchase_Frequency_Trend"] = np.nan

    result = validate_purchase_frequency_trend_features(
        missing_trend_df
    )

    assert result["missing_trend_values"] == 1
    assert result["trend_values_valid"] is False
    assert result["passed"] is False

    # 5. Infinite trend values
    infinite_trend_df = valid_df.copy()
    infinite_trend_df.loc[0, "Purchase_Frequency_Trend"] = np.inf
    infinite_trend_df.loc[1, "Purchase_Frequency_Trend"] = -np.inf

    result = validate_purchase_frequency_trend_features(
        infinite_trend_df
    )

    assert result["infinite_trend_values"] == 2
    assert result["trend_values_valid"] is False
    assert result["passed"] is False

    # 6. Empty dataset
    empty_df = valid_df.iloc[0:0].copy()

    result = validate_purchase_frequency_trend_features(empty_df)

    assert result["customer_count"] == 0
    assert result["has_customers"] is False
    assert result["customer_ids_valid"] is True
    assert result["trend_values_valid"] is True
    assert result["passed"] is False

def test_analyze_purchase_frequency_trend_basis(sample_frequency_transactions):
    """
    Test analyze_purchase_frequency_trend_basis() for monthly
    purchase frequency, zero-purchase months, trend slopes,
    cutoff handling, validation, and input errors.
    """

    import pandas as pd
    import pytest
    import warnings

    from src.features.features import analyze_purchase_frequency_trend_basis

    # ---------------------------------------------------------
    # Helper: Suppress only pandas date-format warning
    # ---------------------------------------------------------
    def run_analysis(**kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Could not infer format.*",
                category=UserWarning,
            )
            return analyze_purchase_frequency_trend_basis(**kwargs)

    # ---------------------------------------------------------
    # 1. Prepare sample transaction data using fixture
    # ---------------------------------------------------------
    df_cleaned = sample_frequency_transactions.copy(deep=True)

    cutoff_date = pd.Timestamp("2026-04-10")

    original_transactions = df_cleaned.copy(deep=True)

    # ---------------------------------------------------------
    # 2. Test analysis with user-supplied cutoff
    # ---------------------------------------------------------
    result = run_analysis(
        df_cleaned=df_cleaned,
        cutoff_date=cutoff_date,
    )

    assert isinstance(result, dict)

    assert {
        "summary",
        "monthly_frequency",
        "customer_analysis",
        "transactions",
        "validation",
    }.issubset(result.keys())

    summary = result["summary"]
    monthly_frequency = result["monthly_frequency"]
    customer_analysis = result["customer_analysis"]
    transactions = result["transactions"]
    validation = result["validation"]

    assert isinstance(summary, dict)
    assert isinstance(monthly_frequency, pd.DataFrame)
    assert isinstance(customer_analysis, pd.DataFrame)
    assert isinstance(transactions, pd.DataFrame)
    assert isinstance(validation, dict)

    # ---------------------------------------------------------
    # 3. Validate cutoff information
    # ---------------------------------------------------------
    assert summary["maximum_invoice_date"] == pd.Timestamp("2026-04-10")
    assert summary["cutoff_date"] == cutoff_date
    assert summary["cutoff_source"] == "User supplied"
    assert summary["prediction_window_days"] == 90

    # ---------------------------------------------------------
    # 4. Validate qualifying transactions
    # ---------------------------------------------------------
    assert len(transactions) == 6

    assert (transactions["Transaction Type"] == "Purchase").all()
    assert (transactions["Customer ID Missing Flag"] == 0).all()
    assert transactions["Include In Sales Analysis"].all()
    assert (transactions["InvoiceDate"] <= cutoff_date).all()

    # ---------------------------------------------------------
    # 5. Validate monthly purchase frequency
    # ---------------------------------------------------------
    assert len(monthly_frequency) == 6

    assert set(monthly_frequency["Customer ID"]) == {"C001", "C002"}

    expected_monthly_frequency = {
        ("C001", "2026-01"): 1,
        ("C001", "2026-02"): 2,
        ("C001", "2026-03"): 0,
        ("C001", "2026-04"): 1,
        ("C002", "2026-03"): 1,
        ("C002", "2026-04"): 1,
    }

    actual_monthly_frequency = {
        (row["Customer ID"], str(row["Purchase_Month"])):
            row["Monthly_Purchase_Frequency"]
        for _, row in monthly_frequency.iterrows()
    }

    assert actual_monthly_frequency == expected_monthly_frequency

    # Verify zero-purchase month is included
    c001_march = monthly_frequency.loc[
        (monthly_frequency["Customer ID"] == "C001")
        & (monthly_frequency["Purchase_Month"] == pd.Period("2026-03", freq="M"))
    ]

    assert len(c001_march) == 1
    assert c001_march.iloc[0]["Monthly_Purchase_Frequency"] == 0

    # ---------------------------------------------------------
    # 6. Validate ordered month index
    # ---------------------------------------------------------
    assert monthly_frequency["Month_Index"].notna().all()

    assert monthly_frequency.loc[
        monthly_frequency["Customer ID"] == "C001",
        "Month_Index",
    ].is_monotonic_increasing

    # ---------------------------------------------------------
    # 7. Validate customer-level purchase frequency trends
    # ---------------------------------------------------------
    assert len(customer_analysis) == 2

    assert set(customer_analysis["Customer ID"]) == {"C001", "C002"}

    customer_1 = customer_analysis.loc[
        customer_analysis["Customer ID"] == "C001"
    ].iloc[0]

    assert customer_1["Purchase_Frequency_Trend"] == pytest.approx(-0.2)
    assert customer_1["Observed_Purchase_Months"] == 3
    assert customer_1["Analysis_Months"] == 4
    assert customer_1["Total_Purchase_Orders"] == 4

    customer_2 = customer_analysis.loc[
        customer_analysis["Customer ID"] == "C002"
    ].iloc[0]

    assert customer_2["Purchase_Frequency_Trend"] == pytest.approx(0.0)
    assert customer_2["Observed_Purchase_Months"] == 2
    assert customer_2["Analysis_Months"] == 2
    assert customer_2["Total_Purchase_Orders"] == 2

    # ---------------------------------------------------------
    # 8. Validate summary statistics
    # ---------------------------------------------------------
    assert summary["historical_transaction_count"] == 6
    assert summary["customer_count"] == 2
    assert summary["observed_calendar_month_count"] == 4

    assert summary["minimum_observed_purchase_months"] == 2
    assert summary["median_observed_purchase_months"] == pytest.approx(2.5)
    assert summary["maximum_observed_purchase_months"] == 3

    assert summary["minimum_analysis_months"] == 2
    assert summary["maximum_analysis_months"] == 4

    assert summary["minimum_frequency_trend"] == pytest.approx(-0.2)
    assert summary["median_frequency_trend"] == pytest.approx(-0.1)
    assert summary["maximum_frequency_trend"] == pytest.approx(0.0)

    assert summary["customers_with_single_purchase_month"] == 0

    # ---------------------------------------------------------
    # 9. Validate analysis conditions
    # ---------------------------------------------------------
    assert bool(validation["customer_count_valid"]) is True
    assert bool(validation["no_duplicate_customer_ids"]) is True
    assert bool(validation["frequency_values_valid"]) is True
    assert bool(validation["trend_values_valid"]) is True
    assert bool(validation["analysis_months_valid"]) is True
    assert bool(validation["cutoff_valid"]) is True
    assert bool(validation["passed"]) is True

    # ---------------------------------------------------------
    # 10. Test automatic cutoff calculation
    # ---------------------------------------------------------
    automatic_result = run_analysis(
        df_cleaned=df_cleaned,
        cutoff_date=None,
    )

    automatic_cutoff = (
        pd.Timestamp("2026-04-10") - pd.Timedelta(days=90)
    )

    assert automatic_result["summary"]["cutoff_source"] == (
        "Automatically calculated"
    )

    assert automatic_result["summary"]["cutoff_date"] == automatic_cutoff

    # Only C001 has a qualifying transaction before this cutoff
    assert len(automatic_result["transactions"]) == 1

    assert set(
        automatic_result["transactions"]["Customer ID"]
    ) == {"C001"}

    assert automatic_result["summary"]["customer_count"] == 1
    assert automatic_result["summary"]["historical_transaction_count"] == 1

    # Only one observed purchase month, therefore slope is zero
    automatic_customer = automatic_result["customer_analysis"].iloc[0]

    assert automatic_customer["Observed_Purchase_Months"] == 1
    assert automatic_customer["Purchase_Frequency_Trend"] == pytest.approx(0.0)

    # ---------------------------------------------------------
    # 11. Verify input DataFrame remains unchanged
    # ---------------------------------------------------------
    pd.testing.assert_frame_equal(
        df_cleaned,
        original_transactions,
    )

    # ---------------------------------------------------------
    # 12. Test missing required columns
    # ---------------------------------------------------------
    invalid_transactions = df_cleaned.drop(
        columns=["Invoice"]
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        run_analysis(
            df_cleaned=invalid_transactions,
            cutoff_date=cutoff_date,
        )

    # ---------------------------------------------------------
    # 13. Test invalid InvoiceDate values
    # ---------------------------------------------------------
    invalid_dates = df_cleaned.copy()

    invalid_dates["InvoiceDate"] = (
        invalid_dates["InvoiceDate"].astype(object)
    )

    invalid_dates.loc[0, "InvoiceDate"] = "invalid-date"

    with pytest.raises((ValueError, TypeError)):
        run_analysis(
            df_cleaned=invalid_dates,
            cutoff_date=cutoff_date,
        )

    # ---------------------------------------------------------
    # 14. Test dataset with no valid InvoiceDate
    # ---------------------------------------------------------
    no_valid_dates = df_cleaned.copy()
    no_valid_dates["InvoiceDate"] = pd.NaT

    with pytest.raises(
        ValueError,
        match="no valid dates",
    ):
        run_analysis(
            df_cleaned=no_valid_dates,
        )

    # ---------------------------------------------------------
    # 15. Test cutoff later than maximum InvoiceDate
    # ---------------------------------------------------------
    with pytest.raises(
        ValueError,
        match="cannot be later",
    ):
        run_analysis(
            df_cleaned=df_cleaned,
            cutoff_date=pd.Timestamp("2026-04-11"),
        )

    # ---------------------------------------------------------
    # 16. Test no qualifying purchase transactions
    # ---------------------------------------------------------
    with pytest.raises(
        ValueError,
        match="No qualifying purchase transactions",
    ):
        run_analysis(
            df_cleaned=df_cleaned,
            cutoff_date=pd.Timestamp("2025-12-31"),
        )


#================================================================================================
#------------------ 5. testing create_interpurchase_variance_features ---------------------------
#================================================================================================

from src.features.features import create_interpurchase_variance_features
from src.features.features import validate_interpurchase_variance_features
@pytest.fixture
def sample_interpurchase_transactions():
    return pd.DataFrame(
        {
            "Customer ID": [
                "C001", "C001", "C001", "C001",
                "C001", "C002", "C003", "C003",
                "C004", "C004", "C004", "C005",
            ],
            "Invoice": [
                "I001", "I001", "I002", "I003",
                "I003", "I004", "I005", "I006",
                "I007", "I008", "I009", "I010",
            ],
            "InvoiceDate": [
                "2026-01-01", "2026-01-01",
                "2026-01-11", "2026-01-21",
                "2026-01-21", "2026-01-05",
                "2026-01-01", "2026-01-08",
                "2026-01-01", "2026-01-11",
                "2026-01-31", "2026-01-15",
            ],
            "Transaction Type": [
                "Purchase", "Purchase", "Purchase",
                "Purchase", "Purchase", "Purchase",
                "Purchase", "Purchase", "Purchase",
                "Purchase", "Purchase", "Purchase",
            ],
            "Customer ID Missing Flag": [0] * 12,
            "Include In Sales Analysis": [True] * 12,
        }
    )

def test_create_interpurchase_variance_features(
    sample_interpurchase_transactions,
):
    df = sample_interpurchase_transactions
    original_df = df.copy(deep=True)

    result = create_interpurchase_variance_features(
        df,
        cutoff_date=pd.Timestamp("2026-01-31"),
    )

    # 1. Output structure and customer population
    assert list(result.columns) == [
        "Customer ID",
        "Interpurchase_Variance",
    ]
    assert result["Customer ID"].is_unique
    assert result["Customer ID"].notna().all()
    assert set(result["Customer ID"]) == {
        "C001", "C002", "C003", "C004", "C005"
    }

    # 2. C001 has purchase intervals of 10 and 10 days.
    # Population variance of [10, 10] is 0.
    variances = result.set_index("Customer ID")[
        "Interpurchase_Variance"
    ]
    assert variances.loc["C001"] == pytest.approx(0.0)

    # 3. C002 has only one purchase order: variance is 0.
    assert variances.loc["C002"] == pytest.approx(0.0)

    # 4. C003 has one interval of 7 days: variance is 0.
    assert variances.loc["C003"] == pytest.approx(0.0)

    # 5. C004 has intervals of 10 and 20 days.
    # Population variance of [10, 20] is 25.
    assert variances.loc["C004"] == pytest.approx(25.0)

    # 6. C005 has one purchase order: variance is 0.
    assert variances.loc["C005"] == pytest.approx(0.0)

    # 7. All variances must be finite, nonnegative, and nonmissing.
    assert result["Interpurchase_Variance"].notna().all()
    assert np.isfinite(result["Interpurchase_Variance"]).all()
    assert result["Interpurchase_Variance"].ge(0).all()

    # 8. Multiple lines in one invoice must count as one event.
    assert len(result) == 5

    # 9. Input dataframe must remain unchanged.
    pd.testing.assert_frame_equal(df, original_df)

    # 10. Missing required column must raise ValueError.
    invalid_df = df.drop(columns=["Invoice"])

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        create_interpurchase_variance_features(
            invalid_df,
            cutoff_date=pd.Timestamp("2026-01-31"),
        )

    # 11. A cutoff after the maximum invoice date must be rejected.
    with pytest.raises(
        ValueError,
        match="cutoff_date cannot be later",
    ):
        create_interpurchase_variance_features(
            df,
            cutoff_date=pd.Timestamp("2027-01-01"),
        )
    import warnings

    # 12. Invalid dates must raise an error.
    invalid_dates = df.copy()
    invalid_dates.loc[0, "InvoiceDate"] = "not-a-date"

    with warnings.catch_warnings():
       warnings.filterwarnings(
        "ignore",
        message="Could not infer format.*",
        category=UserWarning,)

    # 13. No qualifying purchases before the cutoff must raise ValueError.
    with pytest.raises(
        ValueError,
        match="No qualifying purchase transactions",
    ):
        create_interpurchase_variance_features(
            df,
            cutoff_date=pd.Timestamp("2025-01-01"),
        )

    # 14. Return transactions must not affect purchase intervals.
    with_returns = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "Customer ID": ["C001"],
                    "Invoice": ["R001"],
                    "InvoiceDate": ["2026-01-15"],
                    "Transaction Type": ["Return"],
                    "Customer ID Missing Flag": [0],
                    "Include In Sales Analysis": [True],
                }
            ),
        ],
        ignore_index=True,
    )

    result_with_return = create_interpurchase_variance_features(
        with_returns,
        cutoff_date=pd.Timestamp("2026-01-31"),
    )

    return_variances = result_with_return.set_index("Customer ID")[
        "Interpurchase_Variance"
    ]
    assert return_variances.loc["C001"] == pytest.approx(0.0)
    assert return_variances.loc["C004"] == pytest.approx(25.0)

    # 15. Excluded and missing-customer transactions must not affect results.
    excluded_rows = pd.DataFrame(
        {
            "Customer ID": ["C001", "C001"],
            "Invoice": ["I_EXCLUDED", "I_MISSING"],
            "InvoiceDate": ["2026-01-15", "2026-01-16"],
            "Transaction Type": ["Purchase", "Purchase"],
            "Customer ID Missing Flag": [0, 1],
            "Include In Sales Analysis": [False, True],
        }
    )

    with_excluded = pd.concat(
        [df, excluded_rows],
        ignore_index=True,
    )

    result_with_excluded = create_interpurchase_variance_features(
        with_excluded,
        cutoff_date=pd.Timestamp("2026-01-31"),
    )

    excluded_variances = result_with_excluded.set_index("Customer ID")[
        "Interpurchase_Variance"
    ]
    assert excluded_variances.loc["C001"] == pytest.approx(0.0)
    assert excluded_variances.loc["C004"] == pytest.approx(25.0)

def test_validate_interpurchase_variance_features():
    """Test interpurchase variance validation rules and failure scenarios."""
    import numpy as np
    import pandas as pd

    # 1. Valid data: variance can be zero but not negative
    valid_df = pd.DataFrame(
        {
            "Customer ID": [101, 102, 103],
            "Interpurchase_Variance": [0.0, 12.5, 100.0],
        }
    )

    result = validate_interpurchase_variance_features(valid_df)

    assert result["passed"] is True
    assert result["required_columns_present"] is True
    assert result["missing_columns"] == []
    assert result["customer_count"] == 3
    assert result["has_customers"] is True
    assert result["missing_customer_ids"] == 0
    assert result["duplicate_customer_ids"] == 0
    assert result["unique_customer_ids"] == 3
    assert result["customer_ids_valid"] is True
    assert result["missing_variance_values"] == 0
    assert result["infinite_variance_values"] == 0
    assert result["negative_variance_values"] == 0
    assert result["variance_values_valid"] is True

    # 2. Missing required column
    missing_column_df = valid_df.drop(
        columns=["Interpurchase_Variance"]
    )
    result = validate_interpurchase_variance_features(
        missing_column_df
    )

    assert result["passed"] is False
    assert result["required_columns_present"] is False
    assert "Interpurchase_Variance" in result["missing_columns"]
    assert "customer_count" not in result

    # 3. Missing and duplicate customer IDs
    invalid_ids_df = valid_df.copy()
    invalid_ids_df.loc[1, "Customer ID"] = 101
    invalid_ids_df.loc[2, "Customer ID"] = None

    result = validate_interpurchase_variance_features(invalid_ids_df)

    assert result["missing_customer_ids"] == 1
    assert result["duplicate_customer_ids"] == 1
    assert result["unique_customer_ids"] == 1
    assert result["customer_ids_valid"] is False
    assert result["passed"] is False

    # 4. Missing variance values
    missing_variance_df = valid_df.copy()
    missing_variance_df.loc[0, "Interpurchase_Variance"] = np.nan

    result = validate_interpurchase_variance_features(
        missing_variance_df
    )

    assert result["missing_variance_values"] == 1
    assert result["variance_values_valid"] is False
    assert result["passed"] is False

    # 5. Infinite variance values
    infinite_variance_df = valid_df.copy()
    infinite_variance_df.loc[0, "Interpurchase_Variance"] = np.inf
    infinite_variance_df.loc[1, "Interpurchase_Variance"] = -np.inf

    result = validate_interpurchase_variance_features(
        infinite_variance_df
    )

    assert result["infinite_variance_values"] == 2
    assert result["variance_values_valid"] is False
    assert result["passed"] is False

    # 6. Negative variance values
    negative_variance_df = valid_df.copy()
    negative_variance_df.loc[0, "Interpurchase_Variance"] = -1.0

    result = validate_interpurchase_variance_features(
        negative_variance_df
    )

    assert result["negative_variance_values"] == 1
    assert result["variance_values_valid"] is False
    assert result["passed"] is False

    # 7. Empty dataset
    empty_df = valid_df.iloc[0:0].copy()

    result = validate_interpurchase_variance_features(empty_df)

    assert result["customer_count"] == 0
    assert result["has_customers"] is False
    assert result["customer_ids_valid"] is True
    assert result["variance_values_valid"] is True
    assert result["passed"] is False

def test_analyze_interpurchase_variance_basis(sample_interpurchase_transactions):
    """
    Test analyze_interpurchase_variance_basis() for invoice-level
    purchase events, interpurchase intervals, population variance,
    cutoff handling, validation, and input errors.
    """

    import pandas as pd
    import pytest
    import warnings

    from src.features.features import analyze_interpurchase_variance_basis

    # ---------------------------------------------------------
    # Helper: Suppress only pandas date-format warning
    # ---------------------------------------------------------
    def run_analysis(**kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Could not infer format.*",
                category=UserWarning,
            )
            return analyze_interpurchase_variance_basis(**kwargs)

    # ---------------------------------------------------------
    # 1. Prepare sample transactions using existing fixture
    # ---------------------------------------------------------
    df_cleaned = sample_interpurchase_transactions.copy(deep=True)

    cutoff_date = pd.Timestamp("2026-01-31")

    original_transactions = df_cleaned.copy(deep=True)

    # ---------------------------------------------------------
    # 2. Test analysis with user-supplied cutoff
    # ---------------------------------------------------------
    result = run_analysis(
        df_cleaned=df_cleaned,
        cutoff_date=cutoff_date,
    )

    assert isinstance(result, dict)

    assert {
        "summary",
        "invoice_purchases",
        "interpurchase_intervals",
        "customer_analysis",
        "validation",
    }.issubset(result.keys())

    summary = result["summary"]
    invoice_purchases = result["invoice_purchases"]
    interpurchase_intervals = result["interpurchase_intervals"]
    customer_analysis = result["customer_analysis"]
    validation = result["validation"]

    assert isinstance(summary, dict)
    assert isinstance(invoice_purchases, pd.DataFrame)
    assert isinstance(interpurchase_intervals, pd.DataFrame)
    assert isinstance(customer_analysis, pd.DataFrame)
    assert isinstance(validation, dict)

    # ---------------------------------------------------------
    # 3. Validate cutoff information
    # ---------------------------------------------------------
    assert summary["maximum_invoice_date"] == pd.Timestamp("2026-01-31")
    assert summary["cutoff_date"] == cutoff_date
    assert summary["cutoff_source"] == "User supplied"
    assert summary["prediction_window_days"] == 90

    # ---------------------------------------------------------
    # 4. Validate qualifying purchase transactions
    # ---------------------------------------------------------
    assert summary["historical_transaction_count"] == 12

    assert (
        (invoice_purchases["Purchase_Date"] <= cutoff_date).all()
    )

    assert len(invoice_purchases) == 10

    # Multiple lines belonging to the same invoice
    # must count as one purchase event.
    assert (
        invoice_purchases.groupby(
            ["Customer ID", "Invoice"]
        ).size() == 1
    ).all()

    assert set(invoice_purchases["Customer ID"]) == {
        "C001", "C002", "C003", "C004", "C005",
    }

    # ---------------------------------------------------------
    # 5. Validate invoice-level purchase events
    # ---------------------------------------------------------
    expected_invoice_dates = {
        ("C001", "I001"): pd.Timestamp("2026-01-01"),
        ("C001", "I002"): pd.Timestamp("2026-01-11"),
        ("C001", "I003"): pd.Timestamp("2026-01-21"),
        ("C002", "I004"): pd.Timestamp("2026-01-05"),
        ("C003", "I005"): pd.Timestamp("2026-01-01"),
        ("C003", "I006"): pd.Timestamp("2026-01-08"),
        ("C004", "I007"): pd.Timestamp("2026-01-01"),
        ("C004", "I008"): pd.Timestamp("2026-01-11"),
        ("C004", "I009"): pd.Timestamp("2026-01-31"),
        ("C005", "I010"): pd.Timestamp("2026-01-15"),
    }

    actual_invoice_dates = {
        (row["Customer ID"], row["Invoice"]): row["Purchase_Date"]
        for _, row in invoice_purchases.iterrows()
    }

    assert actual_invoice_dates == expected_invoice_dates

    # ---------------------------------------------------------
    # 6. Validate interpurchase intervals
    # ---------------------------------------------------------
    assert len(interpurchase_intervals) == 5

    expected_intervals = {
        ("C001", "I002"): 10.0,
        ("C001", "I003"): 10.0,
        ("C003", "I006"): 7.0,
        ("C004", "I008"): 10.0,
        ("C004", "I009"): 20.0,
    }

    actual_intervals = {
        (row["Customer ID"], row["Invoice"]):
            row["Interpurchase_Days"]
        for _, row in interpurchase_intervals.iterrows()
    }

    assert actual_intervals == expected_intervals

    # Verify previous purchase dates
    c001_i002 = invoice_purchases.loc[
        (invoice_purchases["Customer ID"] == "C001")
        & (invoice_purchases["Invoice"] == "I002")
    ].iloc[0]

    assert c001_i002["Previous_Purchase_Date"] == pd.Timestamp(
        "2026-01-01"
    )

    assert c001_i002["Interpurchase_Days"] == pytest.approx(10.0)

    # First purchase of every customer should have no previous date.
    first_purchases = invoice_purchases.groupby(
        "Customer ID"
    ).head(1)

    assert first_purchases["Previous_Purchase_Date"].isna().all()

    # ---------------------------------------------------------
    # 7. Validate customer-level interval statistics
    # ---------------------------------------------------------
    assert len(customer_analysis) == 5

    assert set(customer_analysis["Customer ID"]) == {
        "C001", "C002", "C003", "C004", "C005",
    }

    customer_1 = customer_analysis.loc[
        customer_analysis["Customer ID"] == "C001"
    ].iloc[0]

    assert customer_1["Number_of_Orders"] == 3
    assert customer_1["Number_of_Intervals"] == 2
    assert customer_1["Mean_Interpurchase_Days"] == pytest.approx(10.0)
    assert customer_1["Median_Interpurchase_Days"] == pytest.approx(10.0)
    assert customer_1["Minimum_Interpurchase_Days"] == pytest.approx(10.0)
    assert customer_1["Maximum_Interpurchase_Days"] == pytest.approx(10.0)
    assert customer_1["Interpurchase_Variance"] == pytest.approx(0.0)

    customer_2 = customer_analysis.loc[
        customer_analysis["Customer ID"] == "C002"
    ].iloc[0]

    assert customer_2["Number_of_Orders"] == 1
    assert customer_2["Number_of_Intervals"] == 0
    assert customer_2["Interpurchase_Variance"] == pytest.approx(0.0)
    assert pd.isna(customer_2["Mean_Interpurchase_Days"])

    customer_3 = customer_analysis.loc[
        customer_analysis["Customer ID"] == "C003"
    ].iloc[0]

    assert customer_3["Number_of_Orders"] == 2
    assert customer_3["Number_of_Intervals"] == 1
    assert customer_3["Mean_Interpurchase_Days"] == pytest.approx(7.0)
    assert customer_3["Interpurchase_Variance"] == pytest.approx(0.0)

    customer_4 = customer_analysis.loc[
        customer_analysis["Customer ID"] == "C004"
    ].iloc[0]

    assert customer_4["Number_of_Orders"] == 3
    assert customer_4["Number_of_Intervals"] == 2
    assert customer_4["Mean_Interpurchase_Days"] == pytest.approx(15.0)
    assert customer_4["Median_Interpurchase_Days"] == pytest.approx(15.0)
    assert customer_4["Minimum_Interpurchase_Days"] == pytest.approx(10.0)
    assert customer_4["Maximum_Interpurchase_Days"] == pytest.approx(20.0)

    # Population variance of [10, 20] = 25
    assert customer_4["Interpurchase_Variance"] == pytest.approx(25.0)

    customer_5 = customer_analysis.loc[
        customer_analysis["Customer ID"] == "C005"
    ].iloc[0]

    assert customer_5["Number_of_Orders"] == 1
    assert customer_5["Number_of_Intervals"] == 0
    assert customer_5["Interpurchase_Variance"] == pytest.approx(0.0)

    # ---------------------------------------------------------
    # 8. Validate first and last purchase dates
    # ---------------------------------------------------------
    assert customer_1["First_Purchase_Date"] == pd.Timestamp("2026-01-01")
    assert customer_1["Last_Purchase_Date"] == pd.Timestamp("2026-01-21")

    assert customer_4["First_Purchase_Date"] == pd.Timestamp("2026-01-01")
    assert customer_4["Last_Purchase_Date"] == pd.Timestamp("2026-01-31")

    # ---------------------------------------------------------
    # 9. Validate summary statistics
    # ---------------------------------------------------------
    assert summary["customer_count"] == 5
    assert summary["distinct_invoice_count"] == 10
    assert summary["total_interpurchase_intervals"] == 5

    assert summary["minimum_orders_per_customer"] == 1
    assert summary["median_orders_per_customer"] == pytest.approx(2.0)
    assert summary["maximum_orders_per_customer"] == 3

    assert summary["customers_with_no_interpurchase_interval"] == 2
    assert summary["customers_with_one_interpurchase_interval"] == 1
    assert summary["customers_with_variance"] == 2

    assert summary["minimum_interpurchase_days"] == pytest.approx(7.0)
    assert summary["median_interpurchase_days"] == pytest.approx(10.0)
    assert summary["maximum_interpurchase_days"] == pytest.approx(20.0)

    assert summary["minimum_interpurchase_variance"] == pytest.approx(0.0)
    assert summary["median_interpurchase_variance"] == pytest.approx(0.0)
    assert summary["maximum_interpurchase_variance"] == pytest.approx(25.0)

    # ---------------------------------------------------------
    # 10. Validate analysis conditions
    # ---------------------------------------------------------
    assert bool(validation["customer_count_valid"]) is True
    assert bool(validation["no_duplicate_customer_ids"]) is True
    assert bool(validation["order_counts_valid"]) is True
    assert bool(validation["interpurchase_intervals_valid"]) is True
    assert bool(validation["variance_values_non_negative"]) is True
    assert bool(validation["cutoff_valid"]) is True
    assert bool(validation["passed"]) is True

    # ---------------------------------------------------------
    # 11. Test automatic cutoff calculation
    # ---------------------------------------------------------
    automatic_transactions = df_cleaned.copy(deep=True)

    # Add one historical purchase before the automatic cutoff.
    automatic_transactions = pd.concat(
        [
            automatic_transactions,
            pd.DataFrame({
                "Customer ID": ["C006"],
                "Invoice": ["I011"],
                "InvoiceDate": ["2025-10-01"],
                "Transaction Type": ["Purchase"],
                "Customer ID Missing Flag": [0],
                "Include In Sales Analysis": [True],
            }),
        ],
        ignore_index=True,
    )

    automatic_result = run_analysis(
        df_cleaned=automatic_transactions,
        cutoff_date=None,
    )

    automatic_cutoff = (
        pd.Timestamp("2026-01-31") - pd.Timedelta(days=90)
    )

    assert automatic_result["summary"]["cutoff_source"] == (
        "Automatically calculated"
    )

    assert automatic_result["summary"]["cutoff_date"] == automatic_cutoff

    assert len(automatic_result["invoice_purchases"]) == 1
    assert len(automatic_result["interpurchase_intervals"]) == 0

    assert automatic_result["summary"]["customer_count"] == 1
    assert automatic_result["summary"]["customers_with_no_interpurchase_interval"] == 1

    automatic_customer = automatic_result["customer_analysis"].iloc[0]

    assert automatic_customer["Number_of_Orders"] == 1
    assert automatic_customer["Number_of_Intervals"] == 0
    assert automatic_customer["Interpurchase_Variance"] == pytest.approx(0.0)

    # ---------------------------------------------------------
    # 12. Verify input DataFrame remains unchanged
    # ---------------------------------------------------------
    pd.testing.assert_frame_equal(
        df_cleaned,
        original_transactions,
    )

    # ---------------------------------------------------------
    # 13. Test missing required columns
    # ---------------------------------------------------------
    invalid_transactions = df_cleaned.drop(
        columns=["Invoice"]
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        run_analysis(
            df_cleaned=invalid_transactions,
            cutoff_date=cutoff_date,
        )

    # ---------------------------------------------------------
    # 14. Test invalid InvoiceDate values
    # ---------------------------------------------------------
    invalid_dates = df_cleaned.copy()

    invalid_dates["InvoiceDate"] = (
        invalid_dates["InvoiceDate"].astype(object)
    )

    invalid_dates.loc[0, "InvoiceDate"] = "invalid-date"

    with pytest.raises((ValueError, TypeError)):
        run_analysis(
            df_cleaned=invalid_dates,
            cutoff_date=cutoff_date,
        )

    # ---------------------------------------------------------
    # 15. Test dataset with no valid InvoiceDate
    # ---------------------------------------------------------
    no_valid_dates = df_cleaned.copy()
    no_valid_dates["InvoiceDate"] = pd.NaT

    with pytest.raises(
        ValueError,
        match="no valid dates",
    ):
        run_analysis(
            df_cleaned=no_valid_dates,
        )

    # ---------------------------------------------------------
    # 16. Test cutoff later than maximum InvoiceDate
    # ---------------------------------------------------------
    with pytest.raises(
        ValueError,
        match="cannot be later",
    ):
        run_analysis(
            df_cleaned=df_cleaned,
            cutoff_date=pd.Timestamp("2026-02-01"),
        )

    # ---------------------------------------------------------
    # 17. Test no qualifying purchase transactions
    # ---------------------------------------------------------
    with pytest.raises(
        ValueError,
        match="No qualifying purchase transactions",
    ):
        run_analysis(
            df_cleaned=df_cleaned,
            cutoff_date=pd.Timestamp("2025-12-31"),
        )


#================================================================================================
#------------------ 6.create_seasonal_purchase_concentration_features ---------------------------
#================================================================================================
from src.features.features import create_seasonal_purchase_concentration_features
from src.features.features import validate_seasonal_purchase_concentration_features

@pytest.fixture
def sample_seasonal_transactions():
    """Create transactions across calendar months for HHI testing."""
    return pd.DataFrame(
        {
            "Customer ID": [
                "C001", "C001", "C001", "C001", "C002", "C002"
            ],
            "Invoice": [
                "I001", "I001", "I002", "I003", "I004", "I005"
            ],
            "InvoiceDate": [
                "2026-01-05",
                "2026-01-05",
                "2026-02-10",
                "2026-02-20",
                "2026-03-10",
                "2026-04-10",
            ],
            "Transaction Type": ["Purchase"] * 6,
            "Customer ID Missing Flag": [0] * 6,
            "Include In Sales Analysis": [True] * 6,
        }
    )

def test_create_seasonal_purchase_concentration_features(
    sample_seasonal_transactions,
):
    """Test HHI calculation, filtering, cutoff, and error handling."""
    cutoff = pd.Timestamp("2026-04-10")
    original = sample_seasonal_transactions.copy(deep=True)

    # 1. Test output structure and customer uniqueness.
    result = create_seasonal_purchase_concentration_features(
        sample_seasonal_transactions,
        cutoff_date=cutoff,
    )

    assert set(result["Customer ID"]) == {"C001", "C002"}
    assert result["Customer ID"].is_unique
    assert "Seasonal_Purchase_Concentration" in result.columns

    # 2. C001 has 3 distinct orders: January=1, February=2.
    # HHI = (1/3)^2 + (2/3)^2 = 5/9.
    c001 = result.loc[result["Customer ID"] == "C001"].iloc[0]
    assert c001["Seasonal_Purchase_Concentration"] == 0.555556

    # C002 has one order in March and one in April.
    # HHI = (1/2)^2 + (1/2)^2 = 0.5.
    c002 = result.loc[result["Customer ID"] == "C002"].iloc[0]
    assert c002["Seasonal_Purchase_Concentration"] == 0.5

    # 3. Verify the input DataFrame is unchanged.
    pd.testing.assert_frame_equal(
        sample_seasonal_transactions,
        original,
    )

    # 4. Verify missing required columns raise ValueError.
    incomplete = sample_seasonal_transactions.drop(
        columns=["Invoice"]
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        create_seasonal_purchase_concentration_features(
            incomplete,
            cutoff_date=cutoff,
        )

    # 5. Verify a cutoff after the maximum date is rejected.
    with pytest.raises(ValueError, match="cutoff_date cannot be later"):
        create_seasonal_purchase_concentration_features(
            sample_seasonal_transactions,
            cutoff_date=pd.Timestamp("2027-01-01"),
        )

    # 6. Verify no qualifying transactions raise ValueError.
    future_transactions = sample_seasonal_transactions.copy()
    future_transactions["InvoiceDate"] = (
        pd.to_datetime(future_transactions["InvoiceDate"])
        + pd.Timedelta(days=100)
    )

    with pytest.raises(
        ValueError,
        match="No qualifying purchase transactions",
    ):
        create_seasonal_purchase_concentration_features(
            future_transactions,
            cutoff_date=cutoff,
        )

    # 7. Verify returns and excluded transactions do not affect HHI.
    extra_rows = pd.DataFrame(
        {
            "Customer ID": ["C001", "C001", "C001"],
            "Invoice": ["I006", "I007", "I008"],
            "InvoiceDate": [
                "2026-03-01",
                "2026-03-02",
                "2026-03-03",
            ],
            "Transaction Type": ["Return", "Purchase", "Purchase"],
            "Customer ID Missing Flag": [0, 1, 0],
            "Include In Sales Analysis": [True, True, False],
        }
    )

    combined = pd.concat(
        [sample_seasonal_transactions, extra_rows],
        ignore_index=True,
    )

    filtered_result = create_seasonal_purchase_concentration_features(
        combined,
        cutoff_date=cutoff,
    )

    filtered_c001 = filtered_result.loc[
        filtered_result["Customer ID"] == "C001"
    ].iloc[0]

    assert filtered_c001["Seasonal_Purchase_Concentration"] == 0.555556

def test_validate_seasonal_purchase_concentration_features():
    """Test seasonal purchase concentration validation rules."""
    import numpy as np
    import pandas as pd

    # 1. Valid data, including boundary values 0 and 1
    valid_df = pd.DataFrame(
        {
            "Customer ID": [101, 102, 103, 104],
            "Seasonal_Purchase_Concentration": [0.0, 0.25, 0.75, 1.0],
        }
    )

    result = validate_seasonal_purchase_concentration_features(valid_df)

    assert result["passed"] is True
    assert result["required_columns_present"] is True
    assert result["missing_columns"] == []
    assert result["customer_count"] == 4
    assert result["has_customers"] is True
    assert result["missing_customer_ids"] == 0
    assert result["duplicate_customer_ids"] == 0
    assert result["unique_customer_ids"] == 4
    assert result["customer_ids_valid"] is True
    assert result["missing_concentration_values"] == 0
    assert result["infinite_concentration_values"] == 0
    assert result["below_zero_values"] == 0
    assert result["above_one_values"] == 0
    assert result["concentration_values_valid"] is True

    # 2. Missing required column
    missing_column_df = valid_df.drop(
        columns=["Seasonal_Purchase_Concentration"]
    )
    result = validate_seasonal_purchase_concentration_features(
        missing_column_df
    )

    assert result["passed"] is False
    assert result["required_columns_present"] is False
    assert (
        "Seasonal_Purchase_Concentration"
        in result["missing_columns"]
    )
    assert "customer_count" not in result

    # 3. Missing and duplicate customer IDs
    invalid_ids_df = valid_df.copy()
    invalid_ids_df.loc[1, "Customer ID"] = 101
    invalid_ids_df.loc[2, "Customer ID"] = None

    result = validate_seasonal_purchase_concentration_features(
        invalid_ids_df
    )

    assert result["missing_customer_ids"] == 1
    assert result["duplicate_customer_ids"] == 1
    assert result["unique_customer_ids"] == 2
    assert result["customer_ids_valid"] is False
    assert result["passed"] is False

    # 4. Missing concentration values
    missing_values_df = valid_df.copy()
    missing_values_df.loc[0, "Seasonal_Purchase_Concentration"] = np.nan

    result = validate_seasonal_purchase_concentration_features(
        missing_values_df
    )

    assert result["missing_concentration_values"] == 1
    assert result["concentration_values_valid"] is False
    assert result["passed"] is False

    # 5. Infinite concentration values
    infinite_values_df = valid_df.copy()
    infinite_values_df.loc[0, "Seasonal_Purchase_Concentration"] = np.inf
    infinite_values_df.loc[1, "Seasonal_Purchase_Concentration"] = -np.inf

    result = validate_seasonal_purchase_concentration_features(
        infinite_values_df
    )

    assert result["infinite_concentration_values"] == 2
    assert result["concentration_values_valid"] is False
    assert result["passed"] is False

    # 6. Values below zero and above one
    out_of_range_df = valid_df.copy()
    out_of_range_df.loc[0, "Seasonal_Purchase_Concentration"] = -0.1
    out_of_range_df.loc[1, "Seasonal_Purchase_Concentration"] = 1.1

    result = validate_seasonal_purchase_concentration_features(
        out_of_range_df
    )

    assert result["below_zero_values"] == 1
    assert result["above_one_values"] == 1
    assert result["concentration_values_valid"] is False
    assert result["passed"] is False

    # 7. Empty dataset
    empty_df = valid_df.iloc[0:0].copy()

    result = validate_seasonal_purchase_concentration_features(empty_df)

    assert result["customer_count"] == 0
    assert result["has_customers"] is False
    assert result["customer_ids_valid"] is True
    assert result["concentration_values_valid"] is True
    assert result["passed"] is False

def test_analyze_seasonal_purchase_concentration_basis(
    sample_seasonal_transactions,
):
    """
    Test analyze_seasonal_purchase_concentration_basis()
    for invoice-level purchase events, monthly order shares,
    seasonal HHI, peak month identification, cutoff handling,
    validation, and input errors.
    """

    import pandas as pd
    import pytest
    import warnings

    from src.features.features import (
        analyze_seasonal_purchase_concentration_basis,
    )

    # ---------------------------------------------------------
    # Helper: Suppress only pandas date-format warning
    # ---------------------------------------------------------
    def run_analysis(**kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Could not infer format.*",
                category=UserWarning,
            )
            return analyze_seasonal_purchase_concentration_basis(
                **kwargs
            )

    # ---------------------------------------------------------
    # 1. Prepare sample transactions using existing fixture
    # ---------------------------------------------------------
    df_cleaned = sample_seasonal_transactions.copy(deep=True)

    cutoff_date = pd.Timestamp("2026-04-10")

    original_transactions = df_cleaned.copy(deep=True)

    # ---------------------------------------------------------
    # 2. Test analysis with user-supplied cutoff
    # ---------------------------------------------------------
    result = run_analysis(
        df_cleaned=df_cleaned,
        cutoff_date=cutoff_date,
    )

    assert isinstance(result, dict)

    assert {
        "summary",
        "invoice_purchases",
        "monthly_orders",
        "customer_analysis",
        "validation",
    }.issubset(result.keys())

    summary = result["summary"]
    invoice_purchases = result["invoice_purchases"]
    monthly_orders = result["monthly_orders"]
    customer_analysis = result["customer_analysis"]
    validation = result["validation"]

    assert isinstance(summary, dict)
    assert isinstance(invoice_purchases, pd.DataFrame)
    assert isinstance(monthly_orders, pd.DataFrame)
    assert isinstance(customer_analysis, pd.DataFrame)
    assert isinstance(validation, dict)

    # ---------------------------------------------------------
    # 3. Validate cutoff information
    # ---------------------------------------------------------
    assert summary["maximum_invoice_date"] == pd.Timestamp(
        "2026-04-10"
    )

    assert summary["cutoff_date"] == cutoff_date
    assert summary["cutoff_source"] == "User supplied"
    assert summary["prediction_window_days"] == 90

    # ---------------------------------------------------------
    # 4. Validate qualifying purchase transactions
    # ---------------------------------------------------------
    assert summary["historical_transaction_count"] == 6

    assert len(invoice_purchases) == 5

    assert (invoice_purchases["Purchase_Date"] <= cutoff_date).all()

    assert set(invoice_purchases["Customer ID"]) == {
        "C001",
        "C002",
    }

    # Multiple transaction lines belonging to one invoice
    # must count as one purchase event.
    assert not invoice_purchases.duplicated(
        subset=["Customer ID", "Invoice"]
    ).any()

    assert summary["distinct_invoice_count"] == 5

    # ---------------------------------------------------------
    # 5. Validate calendar month extraction
    # ---------------------------------------------------------
    expected_months = {
        ("C001", 1): "January",
        ("C001", 2): "February",
        ("C002", 3): "March",
        ("C002", 4): "April",
    }

    actual_months = {
        (row["Customer ID"], row["Purchase_Month"]):
            row["Purchase_Month_Name"]
        for _, row in invoice_purchases.drop_duplicates(
            subset=["Customer ID", "Purchase_Month"]
        ).iterrows()
    }

    assert actual_months == expected_months

    # ---------------------------------------------------------
    # 6. Validate monthly order counts and purchase shares
    # ---------------------------------------------------------
    assert len(monthly_orders) == 4

    expected_monthly_orders = {
        ("C001", 1): (1, 3, pytest.approx(1 / 3)),
        ("C001", 2): (2, 3, pytest.approx(2 / 3)),
        ("C002", 3): (1, 2, pytest.approx(0.5)),
        ("C002", 4): (1, 2, pytest.approx(0.5)),
    }

    for _, row in monthly_orders.iterrows():
        key = (row["Customer ID"], row["Purchase_Month"])

        expected_count, expected_total, expected_share = (
            expected_monthly_orders[key]
        )

        assert row["Monthly_Order_Count"] == expected_count
        assert row["Total_Orders"] == expected_total
        assert row["Monthly_Purchase_Share"] == expected_share

    # Verify monthly shares sum to one for each customer.
    monthly_share_totals = monthly_orders.groupby(
        "Customer ID"
    )["Monthly_Purchase_Share"].sum()

    assert monthly_share_totals["C001"] == pytest.approx(1.0)
    assert monthly_share_totals["C002"] == pytest.approx(1.0)

    # ---------------------------------------------------------
    # 7. Validate customer-level seasonal concentration
    # ---------------------------------------------------------
    assert len(customer_analysis) == 2

    assert set(customer_analysis["Customer ID"]) == {
        "C001",
        "C002",
    }

    customer_1 = customer_analysis.loc[
        customer_analysis["Customer ID"] == "C001"
    ].iloc[0]

    assert customer_1["Active_Calendar_Months"] == 2
    assert customer_1["Total_Orders"] == 3

    assert customer_1["Peak_Month_Share"] == pytest.approx(
        2 / 3
    )

    assert customer_1["Seasonal_Concentration_HHI"] == pytest.approx(
        5 / 9
    )

    assert customer_1["Peak_Purchase_Month"] == "February"

    customer_2 = customer_analysis.loc[
        customer_analysis["Customer ID"] == "C002"
    ].iloc[0]

    assert customer_2["Active_Calendar_Months"] == 2
    assert customer_2["Total_Orders"] == 2

    assert customer_2["Peak_Month_Share"] == pytest.approx(
        0.5
    )

    assert customer_2["Seasonal_Concentration_HHI"] == pytest.approx(
        0.5
    )

    # March and April have equal shares.
    # Production logic selects the earlier calendar month.
    assert customer_2["Peak_Purchase_Month"] == "March"

    # ---------------------------------------------------------
    # 8. Validate summary statistics
    # ---------------------------------------------------------
    assert summary["customer_count"] == 2
    assert summary["distinct_invoice_count"] == 5

    assert summary["minimum_orders_per_customer"] == 2
    assert summary["median_orders_per_customer"] == pytest.approx(2.5)
    assert summary["maximum_orders_per_customer"] == 3

    assert summary["minimum_active_calendar_months"] == 2
    assert summary["median_active_calendar_months"] == pytest.approx(2.0)
    assert summary["maximum_active_calendar_months"] == 2

    assert summary["customers_with_one_order"] == 0
    assert summary["customers_with_one_active_month"] == 0

    assert summary["minimum_peak_month_share"] == pytest.approx(0.5)
    assert summary["median_peak_month_share"] == pytest.approx(
        7 / 12
    )
    assert summary["maximum_peak_month_share"] == pytest.approx(
        2 / 3
    )

    assert summary["minimum_seasonal_hhi"] == pytest.approx(0.5)
    assert summary["median_seasonal_hhi"] == pytest.approx(
        19 / 36
    )
    assert summary["maximum_seasonal_hhi"] == pytest.approx(
        5 / 9
    )

    # ---------------------------------------------------------
    # 9. Validate analysis conditions
    # ---------------------------------------------------------
    assert bool(validation["customer_count_valid"]) is True
    assert bool(validation["no_duplicate_customer_ids"]) is True
    assert bool(validation["order_counts_valid"]) is True
    assert bool(validation["monthly_shares_valid"]) is True
    assert bool(validation["peak_month_share_valid"]) is True
    assert bool(validation["seasonal_hhi_valid"]) is True
    assert bool(validation["cutoff_valid"]) is True
    assert bool(validation["passed"]) is True

    # ---------------------------------------------------------
    # 10. Test automatic cutoff calculation
    # ---------------------------------------------------------
    automatic_transactions = df_cleaned.copy(deep=True)

    automatic_result = run_analysis(
        df_cleaned=automatic_transactions,
        cutoff_date=None,
    )

    automatic_cutoff = (
        pd.Timestamp("2026-04-10") - pd.Timedelta(days=90)
    )

    assert automatic_result["summary"]["cutoff_source"] == (
        "Automatically calculated"
    )

    assert automatic_result["summary"]["cutoff_date"] == (
        automatic_cutoff
    )

    assert len(automatic_result["invoice_purchases"]) == 1

    assert automatic_result["invoice_purchases"]["Invoice"].tolist() == [
        "I001"
    ]

    assert automatic_result["invoice_purchases"]["Customer ID"].tolist() == [
        "C001"
    ]

    assert automatic_result["summary"]["customer_count"] == 1

    assert automatic_result["summary"]["distinct_invoice_count"] == 1

    assert automatic_result["validation"]["passed"] is True

    # ---------------------------------------------------------
    # 11. Verify input DataFrame remains unchanged
    # ---------------------------------------------------------
    pd.testing.assert_frame_equal(
        df_cleaned,
        original_transactions,
    )

    # ---------------------------------------------------------
    # 12. Test missing required columns
    # ---------------------------------------------------------
    invalid_transactions = df_cleaned.drop(
        columns=["Invoice"]
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        run_analysis(
            df_cleaned=invalid_transactions,
            cutoff_date=cutoff_date,
        )

    # ---------------------------------------------------------
    # 13. Test invalid InvoiceDate values
    # ---------------------------------------------------------
    invalid_dates = df_cleaned.copy()

    invalid_dates["InvoiceDate"] = (
        invalid_dates["InvoiceDate"].astype(object)
    )

    invalid_dates.loc[0, "InvoiceDate"] = "invalid-date"

    with pytest.raises((ValueError, TypeError)):
        run_analysis(
            df_cleaned=invalid_dates,
            cutoff_date=cutoff_date,
        )

    # ---------------------------------------------------------
    # 14. Test dataset with no valid InvoiceDate
    # ---------------------------------------------------------
    no_valid_dates = df_cleaned.copy()
    no_valid_dates["InvoiceDate"] = pd.NaT

    with pytest.raises(
        ValueError,
        match="no valid dates",
    ):
        run_analysis(
            df_cleaned=no_valid_dates,
        )

    # ---------------------------------------------------------
    # 15. Test cutoff later than maximum InvoiceDate
    # ---------------------------------------------------------
    with pytest.raises(
        ValueError,
        match="cannot be later",
    ):
        run_analysis(
            df_cleaned=df_cleaned,
            cutoff_date=pd.Timestamp("2026-04-11"),
        )

    # ---------------------------------------------------------
    # 16. Test no qualifying purchase transactions
    # ---------------------------------------------------------
    with pytest.raises(
        ValueError,
        match="No qualifying purchase transactions",
    ):
        run_analysis(
            df_cleaned=df_cleaned,
            cutoff_date=pd.Timestamp("2025-12-31"),
        )



#================================================================================================================================
#------------- 7.testing create_product_category_affinity_features ----------------------------------------------------
#================================================================================================================================
from src.features.features import create_product_category_affinity_features,analyze_product_taxonomy
from src.features.features import validate_product_category_affinity_features
@pytest.fixture
def sample_category_affinity_transactions():
    return pd.DataFrame(
        {
            "Customer ID": [
                101, 101, 101,
                202, 202,
                0, 303, 404,
            ],
            "StockCode": [
                "85123A",
                "85123A",
                "22423",
                "22423",
                "85123A",
                "22423",
                "85123A",
                "22423",
            ],
            "Description": [
                "White hanging heart decoration",
                "White hanging heart decoration",
                "Rabbit decoration",
                "Rabbit decoration",
                "White hanging heart decoration",
                "Rabbit decoration",
                "White hanging heart decoration",
                "Rabbit decoration",
            ],
            "InvoiceDate": [
                "2026-01-05",
                "2026-01-05",
                "2026-02-10",
                "2026-02-15",
                "2026-03-10",
                "2026-03-12",
                "2026-03-15",
                "2026-03-20",
            ],
            "Transaction Type": [
                "Purchase",
                "Purchase",
                "Purchase",
                "Purchase",
                "Purchase",
                "Purchase",
                "Return",
                "Purchase",
            ],
            "Include In Sales Analysis": [
                True, True, True, True,
                True, True, True, False,
            ],
            "Analytical Transaction Value": [
                100.0, 50.0, 150.0,
                200.0, 100.0, 500.0,
                -50.0, 300.0,
            ],
        }
    )

def test_create_product_category_affinity_features(
    sample_category_affinity_transactions,
):
    df = sample_category_affinity_transactions
    original_df = df.copy(deep=True)
    cutoff = pd.Timestamp("2026-03-20")

    # 1. Generate category-affinity features.
    result = create_product_category_affinity_features(
        df,
        cutoff_date=cutoff,
    )

    # 2. Validate the output structure.
    assert not result.empty
    assert "Customer ID" in result.columns
    assert result["Customer ID"].is_unique
    assert result["Customer ID"].notna().all()
    assert set(result["Customer ID"]) == {101, 202}

    # 3. Identify the generated affinity columns.
    affinity_columns = [
        column
        for column in result.columns
        if column.endswith("_Affinity")
    ]

    assert affinity_columns
    assert len(affinity_columns) == len(set(affinity_columns))

    # 4. Validate the affinity ranges.
    affinity_values = result[affinity_columns]

    assert affinity_values.notna().all().all()
    assert np.isfinite(affinity_values.to_numpy()).all()
    assert affinity_values.ge(0).all().all()
    assert affinity_values.le(1).all().all()

    # 5. Affinities must sum to one for every customer.
    assert np.allclose(
        affinity_values.sum(axis=1).to_numpy(),
        1.0,
        atol=1e-9,
    )

    # 6. Independently calculate expected affinities using
    # the actual taxonomy returned by the production function.
    taxonomy = analyze_product_taxonomy(df)
    product_catalog = taxonomy["product_catalog"].copy()

    product_catalog["StockCode_Key"] = (
        product_catalog["StockCode_Key"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    category_mapping = (
        product_catalog[
            ["StockCode_Key", "Product_Category"]
        ]
        .drop_duplicates(subset=["StockCode_Key"])
    )

    expected_transactions = df.copy()
    expected_transactions["InvoiceDate"] = pd.to_datetime(
        expected_transactions["InvoiceDate"],
        errors="coerce",
    )
    expected_transactions["StockCode_Key"] = (
        expected_transactions["StockCode"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    expected_transactions = expected_transactions[
        (expected_transactions["InvoiceDate"] <= cutoff)
        & expected_transactions["Customer ID"].notna()
        & (expected_transactions["Customer ID"] != 0)
        & (
            expected_transactions["Transaction Type"]
            == "Purchase"
        )
        & (
            expected_transactions["Include In Sales Analysis"]
            == True
        )
    ].copy()

    expected_transactions = expected_transactions.merge(
        category_mapping,
        on="StockCode_Key",
        how="left",
        validate="many_to_one",
    )

    expected_transactions["Product_Category"] = (
        expected_transactions["Product_Category"]
        .fillna("Other / Unclassified")
    )

    category_values = (
        expected_transactions.groupby(
            ["Customer ID", "Product_Category"]
        )["Analytical Transaction Value"]
        .sum()
    )

    customer_totals = (
        expected_transactions.groupby("Customer ID")[
            "Analytical Transaction Value"
        ].sum()
    )

    for customer_id in [101, 202]:
        customer_result = result.loc[
            result["Customer ID"] == customer_id
        ].iloc[0]

        for category_column in affinity_columns:
            category_name = category_column.removesuffix(
                "_Affinity"
            )

            # Reverse the output column-name normalization.
            # If two names normalize to the same output name,
            # this check will flag that ambiguity.
            matching_categories = [
                category
                for category in taxonomy["category_priority"]
                if (
                    category.replace(" & ", "_")
                    .replace(" ", "_")
                    .replace("/", "_")
                    == category_name
                )
            ]

            assert len(matching_categories) == 1

            category = matching_categories[0]
            category_value = category_values.get(
                (customer_id, category),
                0.0,
            )
            total_value = customer_totals.loc[customer_id]
            expected_affinity = category_value / total_value

            assert customer_result[category_column] == pytest.approx(
                expected_affinity,
                abs=1e-9,
            )

    # 7. Validate metadata.
    validation = result.attrs["validation"]

    assert bool(validation["passed"])
    assert bool(validation["one_row_per_customer"])
    assert bool(validation["customer_count"])
    assert bool(validation["no_missing_customer_id"])
    assert bool(validation["no_negative_affinity"])
    assert bool(validation["no_affinity_above_one"])
    assert bool(validation["affinity_sums_to_one"])
    assert bool(validation["expected_category_count"])

    assert result.attrs["cutoff_date"] == cutoff
    assert result.attrs["customer_count"] == 2
    assert result.attrs["qualifying_transaction_count"] == 5
    assert result.attrs["category_count"] == len(affinity_columns)

    # 8. Verify that the original input was not modified.
    pd.testing.assert_frame_equal(df, original_df)

    # 9. Missing required columns must raise ValueError.
    invalid_df = df.drop(columns=["StockCode"])

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        create_product_category_affinity_features(
            invalid_df,
            cutoff_date=cutoff,
        )

    # 10. A cutoff after the maximum transaction date is invalid.
    with pytest.raises(
        ValueError,
        match="cutoff_date cannot be later",
    ):
        create_product_category_affinity_features(
            df,
            cutoff_date=pd.Timestamp("2027-01-01"),
        )

    # 11. Transactions after the cutoff must not qualify.
    future_df = df.copy()
    future_df["InvoiceDate"] = (
        pd.to_datetime(future_df["InvoiceDate"])
        + pd.Timedelta(days=100)
    )

    with pytest.raises(
        ValueError,
        match="No qualifying customer transactions",
    ):
        create_product_category_affinity_features(
            future_df,
            cutoff_date=cutoff,
        )

    # 12. Transactions with invalid customer IDs, returns,
    # or Include In Sales Analysis=False must be excluded.
    filtered_df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "Customer ID": [101, 101, 0],
                    "StockCode": [
                        "22423", "85123A", "22423"
                    ],
                    "Description": [
                        "Rabbit decoration",
                        "White hanging heart decoration",
                        "Rabbit decoration",
                    ],
                    "InvoiceDate": [
                        "2026-03-16",
                        "2026-03-17",
                        "2026-03-18",
                    ],
                    "Transaction Type": [
                        "Return", "Purchase", "Purchase"
                    ],
                    "Include In Sales Analysis": [
                        True, False, True
                    ],
                    "Analytical Transaction Value": [
                        -25.0, 999.0, 888.0
                    ],
                }
            ),
        ],
        ignore_index=True,
    )

    filtered_result = create_product_category_affinity_features(
        filtered_df,
        cutoff_date=cutoff,
    )

    pd.testing.assert_frame_equal(
        result.sort_values("Customer ID").reset_index(drop=True),
        filtered_result.sort_values("Customer ID").reset_index(drop=True),
        check_flags=False,
    )

def test_validate_product_category_affinity_features():
    """Test category-affinity validation and failure scenarios."""
    import numpy as np
    import pandas as pd
    import pytest
    import warnings

    warnings.filterwarnings(
        "ignore",
        message="Could not infer format, so each element will be parsed individually.*",
        category=UserWarning,
    )

    affinity_columns = [
        "Home_Fragrance_Affinity",
        "Lighting_Candles_Affinity",
        "Stationery_Paper_Affinity",
        "Jewelry_Fashion_Accessories_Affinity",
        "Bags_Travel_Accessories_Affinity",
        "Kitchen_Dining_Affinity",
        "Storage_Organization_Affinity",
        "Garden_Outdoor_Affinity",
        "Bath_Personal_Care_Affinity",
        "Toys_Games_Affinity",
        "Party_Celebration_Affinity",
        "Gifts_Seasonal_Affinity",
        "Food_Consumables_Affinity",
        "Home_Decor_Affinity",
        "Other___Unclassified_Affinity",
    ]

    # 1. Valid data: two customers with 15 affinity values summing to 1
    customer_ids = [101, 102]
    affinity_values = np.full((2, 15), 1 / 15)

    affinity_df = pd.DataFrame(
        affinity_values,
        columns=affinity_columns,
    )
    affinity_df.insert(0, "Customer ID", customer_ids)

    transactions_df = pd.DataFrame(
        {
            "Customer ID": [101, 102, 101, 102],
            "InvoiceDate": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-02-01",
                    "2025-02-02",
                ]
            ),
            "Transaction Type": ["Purchase"] * 4,
            "Include In Sales Analysis": [True] * 4,
            "Analytical Transaction Value": [100.0, 150.0, 50.0, 75.0],
        }
    )

    cutoff = pd.Timestamp("2025-01-31")

    result = validate_product_category_affinity_features(
        affinity_df,
        transactions_df,
        cutoff_date=cutoff,
    )

    # Validate successful checks
    assert bool(result["validation"]["passed"]) is True
    assert bool(result["validation"]["customer_id_column_present"]) is True
    assert bool(result["validation"]["no_missing_customer_id"]) is True
    assert bool(result["validation"]["customer_id_is_unique"]) is True
    assert bool(result["validation"]["customer_count_matches"]) is True
    assert bool(result["validation"]["customer_population_matches"]) is True
    assert bool(result["validation"]["affinity_columns_present"]) is True
    assert bool(result["validation"]["affinity_values_are_numeric"]) is True
    assert bool(result["validation"]["no_missing_affinity_values"]) is True
    assert bool(result["validation"]["no_negative_affinity"]) is True
    assert bool(result["validation"]["no_affinity_above_one"]) is True
    assert bool(result["validation"]["affinity_sums_to_one"]) is True
    assert bool(result["validation"]["expected_category_count"]) is True
    assert result["failed_checks"] == []

    # Validate summary
    assert result["summary"]["customer_count"] == 2
    assert result["summary"]["expected_customer_count"] == 2
    assert result["summary"]["category_affinity_count"] == 15
    assert result["summary"]["qualifying_transaction_count"] == 2
    assert result["summary"]["cutoff_date"] == cutoff
    assert result["summary"]["minimum_affinity_sum"] == pytest.approx(1.0)
    assert result["summary"]["maximum_affinity_sum"] == pytest.approx(1.0)
    assert result["summary"]["median_affinity_sum"] == pytest.approx(1.0)

    # 2. Missing required transaction columns raises ValueError
    with pytest.raises(
        ValueError,
        match="Missing required transaction columns",
    ):
        validate_product_category_affinity_features(
            affinity_df,
            transactions_df.drop(columns=["InvoiceDate"]),
            cutoff_date=cutoff,
        )

    # 3. Missing Customer ID in the affinity DataFrame raises ValueError
    with pytest.raises(ValueError, match="Affinity dataframe is missing"):
        validate_product_category_affinity_features(
            affinity_df.drop(columns=["Customer ID"]),
            transactions_df,
            cutoff_date=cutoff,
        )

    # 4. No affinity columns raises ValueError
    with pytest.raises(
        ValueError,
        match="No Product Category Affinity columns",
    ):
        validate_product_category_affinity_features(
            affinity_df[["Customer ID"]].copy(),
            transactions_df,
            cutoff_date=cutoff,
        )

    # 5. Invalid dates throughout transaction data raise ValueError
    invalid_dates_df = transactions_df.copy()
    invalid_dates_df["InvoiceDate"] = "not-a-date"

    with pytest.raises(
        ValueError,
        match="InvoiceDate does not contain a valid date",
    ):
        validate_product_category_affinity_features(
            affinity_df,
            invalid_dates_df,
            cutoff_date=cutoff,
        )

    # 6. A cutoff later than the maximum invoice date raises ValueError
    with pytest.raises(
        ValueError,
        match="cutoff_date cannot be later",
    ):
        validate_product_category_affinity_features(
            affinity_df,
            transactions_df,
            cutoff_date=pd.Timestamp("2025-03-01"),
        )

    # 7. Customer population mismatch is reported
    missing_customer_affinity_df = affinity_df.iloc[[0]].copy()

    result = validate_product_category_affinity_features(
        missing_customer_affinity_df,
        transactions_df,
        cutoff_date=cutoff,
    )

    assert bool(result["validation"]["customer_count_matches"]) is False
    assert bool(result["validation"]["customer_population_matches"]) is False
    assert "customer_population_matches" in result["failed_checks"]
    assert bool(result["validation"]["passed"]) is False

    # 8. Duplicate customer IDs are detected
    duplicate_customer_df = pd.concat(
        [affinity_df, affinity_df.iloc[[0]]],
        ignore_index=True,
    )

    result = validate_product_category_affinity_features(
        duplicate_customer_df,
        transactions_df,
        cutoff_date=cutoff,
    )

    assert bool(result["validation"]["customer_id_is_unique"]) is False
    assert bool(result["validation"]["passed"]) is False
    assert "customer_id_is_unique" in result["failed_checks"]

    # 9. Missing and negative affinity values are detected
    invalid_affinity_df = affinity_df.copy()
    invalid_affinity_df.loc[0, affinity_columns[0]] = np.nan
    invalid_affinity_df.loc[1, affinity_columns[1]] = -0.1

    result = validate_product_category_affinity_features(
        invalid_affinity_df,
        transactions_df,
        cutoff_date=cutoff,
    )

    assert bool(result["validation"]["no_missing_affinity_values"]) is False
    assert bool(result["validation"]["no_negative_affinity"]) is False
    assert bool(result["validation"]["passed"]) is False

    # 10. Affinity values above one are detected
    above_one_df = affinity_df.copy()
    above_one_df.loc[0, affinity_columns[0]] = 1.1

    result = validate_product_category_affinity_features(
        above_one_df,
        transactions_df,
        cutoff_date=cutoff,
    )

    assert bool(result["validation"]["no_affinity_above_one"]) is False
    assert bool(result["validation"]["affinity_sums_to_one"]) is False
    assert bool(result["validation"]["passed"]) is False

    # 11. Automatic cutoff is max invoice date minus 90 days
    automatic_cutoff_result = validate_product_category_affinity_features(
        affinity_df,
        transactions_df,
    )

    expected_cutoff = (
        transactions_df["InvoiceDate"].max() - pd.Timedelta(days=90)
    )

    assert automatic_cutoff_result["summary"]["cutoff_date"] == expected_cutoff


def test_analyze_product_taxonomy(sample_category_affinity_transactions):

    import pandas as pd
    import pytest
    import warnings

    from src.features.features import analyze_product_taxonomy

    # ---------------------------------------------------------
    # Helper: execute function with targeted warning suppression
    # ---------------------------------------------------------
    def run_analysis(data):

        with warnings.catch_warnings():

            warnings.filterwarnings(
                "ignore",
                message="Could not infer format.*",
                category=UserWarning,
            )

            return analyze_product_taxonomy(data)

    # ---------------------------------------------------------
    # 1. Prepare test data
    # ---------------------------------------------------------
    df = sample_category_affinity_transactions.copy(deep=True)

    additional_products = pd.DataFrame(
        {
            "Customer ID": [
                505, 606, 707, 808, 909, 1001, 1101
            ],

            "StockCode": [
                "23444",
                "NEW001",
                "NEW002",
                "NEW003",
                "NEW004",
                "NEW005",
                " NEW006 ",
            ],

            "Description": [
                "Administrative adjustment",
                None,
                "CANDLE GIFT",
                "MUG TEA",
                "UNUSUAL ITEM XYZ",
                "PENCIL",
                "BALL PEN",
            ],

            "InvoiceDate": ["2026-04-01"] * 7,

            "Transaction Type": ["Purchase"] * 7,

            "Include In Sales Analysis": [True] * 7,

            "Analytical Transaction Value": [100.0] * 7,
        }
    )

    df = pd.concat(
        [df, additional_products],
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # 2. Execute production function
    # ---------------------------------------------------------
    result = run_analysis(df)

    # ---------------------------------------------------------
    # 3. Validate returned package structure
    # ---------------------------------------------------------
    expected_keys = {
        "product_catalog",
        "category_summary",
        "assignment_summary",
        "multi_match_products",
        "unclassified_products",
        "validation",
        "category_rules",
        "category_priority",
        "non_product_stock_codes",
        "summary",
    }

    assert set(result.keys()) == expected_keys

    # ---------------------------------------------------------
    # 4. Check product catalogue
    # ---------------------------------------------------------
    catalog = result["product_catalog"]

    assert len(catalog) == 8

    assert catalog["StockCode_Key"].is_unique

    assert bool(catalog["Product_Category"].notna().all()) is True

    assert "23444" not in catalog["StockCode_Key"].tolist()

    assert "NEW001" in catalog["StockCode_Key"].tolist()

    assert "NEW006" in catalog["StockCode_Key"].tolist()

    assert " NEW006 " not in catalog["StockCode_Key"].tolist()

    # ---------------------------------------------------------
    # 5. Check canonical descriptions
    # ---------------------------------------------------------
    rabbit_product = catalog.loc[
        catalog["StockCode_Key"] == "22423"
    ].iloc[0]

    heart_product = catalog.loc[
        catalog["StockCode_Key"] == "85123A"
    ].iloc[0]

    assert rabbit_product["Description"] == "RABBIT DECORATION"

    assert heart_product["Description"] == (
        "WHITE HANGING HEART DECORATION"
    )

    # ---------------------------------------------------------
    # 6. Check missing description handling
    # ---------------------------------------------------------
    missing_description = catalog.loc[
        catalog["StockCode_Key"] == "NEW001"
    ].iloc[0]

    assert missing_description["Description"] == "NO DESCRIPTION"

    assert bool(
        missing_description["Description_Present"]
    ) is False

    assert missing_description["Product_Category"] == (
        "Other / Unclassified"
    )

    assert missing_description["Category_Assignment_Status"] == (
        "Missing_Description"
    )

    # ---------------------------------------------------------
    # 7. Check primary category assignment
    # ---------------------------------------------------------
    def get_category(stock_code):

        return catalog.loc[
            catalog["StockCode_Key"] == stock_code,
            "Product_Category",
        ].iloc[0]

    assert get_category("85123A") == "Home Decor"

    assert get_category("22423") == "Home Decor"

    assert get_category("NEW005") == "Stationery & Paper"

    assert get_category("NEW006") == "Stationery & Paper"

    assert get_category("NEW004") == "Other / Unclassified"

    # ---------------------------------------------------------
    # 8. Check strong keyword priority over weak keywords
    # ---------------------------------------------------------
    candle_product = catalog.loc[
        catalog["StockCode_Key"] == "NEW002"
    ].iloc[0]

    assert candle_product["Product_Category"] == "Lighting & Candles"

    assert bool(
        candle_product["Strong_Lighting_Candles"]
    ) is True

    assert bool(
        candle_product["Weak_Gifts_Seasonal"]
    ) is True

    mug_product = catalog.loc[
        catalog["StockCode_Key"] == "NEW003"
    ].iloc[0]

    assert mug_product["Product_Category"] == "Kitchen & Dining"

    assert bool(
        mug_product["Strong_Kitchen_Dining"]
    ) is True

    assert bool(
        mug_product["Weak_Food_Consumables"]
    ) is True

    # ---------------------------------------------------------
    # 9. Check multiple category matches
    # ---------------------------------------------------------
    multi_match = result["multi_match_products"]

    assert len(multi_match) == 2

    assert set(multi_match["StockCode_Key"]) == {
        "NEW002",
        "NEW003",
    }

    assert bool(
        (multi_match["Matched_Category_Count"] == 2).all()
    ) is True

    assert bool(
        (multi_match["Strong_Category_Match_Count"] == 1).all()
    ) is True

    # ---------------------------------------------------------
    # 10. Check unclassified products
    # ---------------------------------------------------------
    unclassified = result["unclassified_products"]

    assert set(unclassified["StockCode_Key"]) == {
        "NEW001",
        "NEW004",
    }

    assert set(
        unclassified["Category_Assignment_Status"]
    ) == {
        "Missing_Description",
        "No_Rule_Match",
    }

    # ---------------------------------------------------------
    # 11. Check category summary
    # ---------------------------------------------------------
    category_summary = result["category_summary"]

    assert len(category_summary) == len(
        result["category_priority"]
    )

    assert category_summary["Product_Category"].tolist() == (
        result["category_priority"]
    )

    assert (
        category_summary["Product_Count"].sum()
        == len(catalog)
    )

    # Home Decor
    assert category_summary.loc[
        category_summary["Product_Category"] == "Home Decor",
        "Product_Count",
    ].iloc[0] == 2

    # Lighting & Candles
    assert category_summary.loc[
        category_summary["Product_Category"] == "Lighting & Candles",
        "Product_Count",
    ].iloc[0] == 1

    # Kitchen & Dining
    assert category_summary.loc[
        category_summary["Product_Category"] == "Kitchen & Dining",
        "Product_Count",
    ].iloc[0] == 1

    # Stationery & Paper: PENCIL + BALL PEN
    assert category_summary.loc[
        category_summary["Product_Category"] == "Stationery & Paper",
        "Product_Count",
    ].iloc[0] == 2

    # Other / Unclassified
    assert category_summary.loc[
        category_summary["Product_Category"] == "Other / Unclassified",
        "Product_Count",
    ].iloc[0] == 2

    assert category_summary["Product_Percentage"].sum() == pytest.approx(
        100.0,
        abs=0.02,
    )

    # ---------------------------------------------------------
    # 12. Check assignment summary
    # ---------------------------------------------------------
    assignment_summary = result["assignment_summary"]

    assert (
        assignment_summary["Product_Count"].sum()
        == len(catalog)
    )

    assignment_counts = assignment_summary.set_index(
        "Category_Assignment_Status"
    )["Product_Count"]

    assert assignment_counts["Rule_Assigned"] == 6

    assert assignment_counts["Missing_Description"] == 1

    assert assignment_counts["No_Rule_Match"] == 1

    assert assignment_summary["Product_Percentage"].sum() == pytest.approx(
        100.0,
        abs=0.02,
    )

    # ---------------------------------------------------------
    # 13. Check overall summary
    # ---------------------------------------------------------
    summary = result["summary"]

    assert summary["total_genuine_products"] == 8

    assert summary["rule_assigned_products"] == 6

    assert summary["missing_description_products"] == 1

    assert summary["no_rule_match_products"] == 1

    assert summary["multi_match_products"] == 2

    assert summary["category_count"] == 15

    assert bool(summary["validation_passed"]) is False

    # ---------------------------------------------------------
    # 14. Check validation results
    # ---------------------------------------------------------
    validation = result["validation"]

    assert bool(
        validation["product_count_is_5120"]
    ) is False

    assert bool(
        validation["one_row_per_stockcode"]
    ) is True

    assert bool(
        validation["no_missing_product_category"]
    ) is True

    assert bool(
        validation["administrative_codes_excluded"]
    ) is True

    assert bool(
        validation["missing_description_count_valid"]
    ) is True

    assert bool(
        validation["assignment_counts_reconcile"]
    ) is True

    assert bool(
        validation["category_counts_reconcile"]
    ) is True

    assert bool(validation["passed"]) is False

    # ---------------------------------------------------------
    # 15. Check category rules and administrative codes
    # ---------------------------------------------------------
    assert "Home Decor" in result["category_rules"]

    assert "Lighting & Candles" in result["category_rules"]

    assert "Stationery & Paper" in result["category_rules"]

    assert "23444" in result["non_product_stock_codes"]

    assert "BANK CHARGES" in result["non_product_stock_codes"]

    # ---------------------------------------------------------
    # 16. Check missing required columns
    # ---------------------------------------------------------
    invalid_df = df.drop(
        columns=["Description"]
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):

        run_analysis(invalid_df)



#================================================================================================
#-------------------------- 8. testing create_return_rate_features  -----------------------------
#================================================================================================
from src.features.features import create_return_rate_features
from src.features.features import validate_return_rate_features
@pytest.fixture
def sample_return_rate_transactions():
    return pd.DataFrame(
        {
            "Customer ID": [
                101, 101, 101, 101,
                202, 202, 202,
                303, 303,
                0,
                404, 404,
            ],
            "Invoice": [
                "P001", "P002", "R001", "R001",
                "P003", "P004", "R002",
                "P005", "P005",
                "P006",
                "P007", "R003",
            ],
            "InvoiceDate": [
                "2026-01-05", "2026-02-10",
                "2026-02-15", "2026-02-15",
                "2026-01-10", "2026-02-10",
                "2026-02-20",
                "2026-03-01", "2026-03-01",
                "2026-03-05",
                "2026-03-10", "2026-03-15",
            ],
            "Quantity": [
                2, 3, -1, -2,
                1, 1, -1,
                2, 3,
                5,
                1, -1,
            ],
            "Transaction Type": [
                "Purchase", "Purchase", "Return", "Return",
                "Purchase", "Purchase", "Cancellation",
                "Purchase", "Purchase",
                "Purchase",
                "Purchase", "Return",
            ],
            "Include In Sales Analysis": [
                True, True, True, True,
                True, True, True,
                True, True,
                True,
                True, True,
            ],
        }
    )

def test_create_return_rate_features(
    sample_return_rate_transactions,
):
    df = sample_return_rate_transactions
    original_df = df.copy(deep=True)

    result = create_return_rate_features(
        df,
        cutoff_date=pd.Timestamp("2026-03-15"),
    )

    # 1. Output structure and customer population
    assert list(result.columns) == ["Customer ID", "Return_Rate"]
    assert result["Customer ID"].is_unique
    assert result["Customer ID"].notna().all()
    assert set(result["Customer ID"]) == {101, 202, 303, 404}

    # 2. Verify return-rate calculations
    rates = result.set_index("Customer ID")["Return_Rate"]

    # Customer 101: 1 distinct return order / 2 purchase orders
    assert rates.loc[101] == pytest.approx(0.5)

    # Customer 202: 1 cancellation order / 2 purchase orders
    assert rates.loc[202] == pytest.approx(0.5)

    # Customer 303: no return orders / 1 purchase order
    assert rates.loc[303] == pytest.approx(0.0)

    # Customer 404: 1 return order / 1 purchase order
    assert rates.loc[404] == pytest.approx(1.0)

    # 3. Rates must be nonnegative and finite
    assert result["Return_Rate"].notna().all()
    assert np.isfinite(result["Return_Rate"]).all()
    assert result["Return_Rate"].ge(0).all()

    # 4. Check metadata
    assert result.attrs["cutoff_date"] == pd.Timestamp("2026-03-15")
    assert result.attrs["max_invoice_date"] == pd.Timestamp("2026-03-15")
    assert result.attrs["customer_count"] == 4
    assert result.attrs["purchase_order_count"] == 6
    assert result.attrs["return_cancellation_order_count"] == 3
    assert result.attrs["customers_with_returns"] == 3
    assert result.attrs["customers_without_returns"] == 1

    # 5. Customer ID zero must be excluded
    assert 0 not in result["Customer ID"].values

    # 6. Input data must remain unchanged
    pd.testing.assert_frame_equal(df, original_df)

    # 7. Missing required column must raise ValueError
    invalid_df = df.drop(columns=["Quantity"])

    with pytest.raises(ValueError, match="Missing required columns"):
        create_return_rate_features(
            invalid_df,
            cutoff_date=pd.Timestamp("2026-03-15"),
        )

    # 8. Invalid or missing dates must raise ValueError

    invalid_dates = df.copy()
    invalid_dates.loc[0, "InvoiceDate"] = "invalid-date"

    import warnings
    with warnings.catch_warnings():
         warnings.filterwarnings(
          "ignore",
        message="Could not infer format.*",
        category=UserWarning,)
         
         with pytest.raises(ValueError,match="InvoiceDate contains missing or invalid values",):
           create_return_rate_features(invalid_dates)
         

    # 9. A cutoff before all transactions must be rejected
    with pytest.raises(
        ValueError,
        match="No transactions are available",
    ):
        create_return_rate_features(
            df,
            cutoff_date=pd.Timestamp("2025-01-01"),
        )

    # 10. Transactions without valid customer IDs must be rejected
    invalid_customers = df.copy()
    invalid_customers["Customer ID"] = 0

    with pytest.raises(
        ValueError,
        match="No customer-identifiable transactions",
    ):
        create_return_rate_features(
            invalid_customers,
            cutoff_date=pd.Timestamp("2026-03-15"),
        )

    # 11. No qualifying purchase orders must raise ValueError
    no_purchases = df.copy()
    no_purchases["Transaction Type"] = "Return"
    no_purchases["Quantity"] = -1

    with pytest.raises(
        ValueError,
        match="No qualifying purchase transactions",
    ):
        create_return_rate_features(
            no_purchases,
            cutoff_date=pd.Timestamp("2026-03-15"),
        )

    # 12. Excluded purchases and nonnegative return quantities
    # must not increase the calculated return rate.
    filtered_df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "Customer ID": [101, 101],
                    "Invoice": ["P_EXCLUDED", "R_NONNEGATIVE"],
                    "InvoiceDate": ["2026-03-14", "2026-03-14"],
                    "Quantity": [100, 2],
                    "Transaction Type": ["Purchase", "Return"],
                    "Include In Sales Analysis": [False, True],
                }
            ),
        ],
        ignore_index=True,
    )

    filtered_result = create_return_rate_features(
        filtered_df,
        cutoff_date=pd.Timestamp("2026-03-15"),
    )
    filtered_rates = filtered_result.set_index("Customer ID")["Return_Rate"]

    assert filtered_rates.loc[101] == pytest.approx(0.5)
    assert filtered_rates.loc[202] == pytest.approx(0.5)
    assert filtered_rates.loc[303] == pytest.approx(0.0)
    assert filtered_rates.loc[404] == pytest.approx(1.0)

def test_validate_return_rate_features(sample_return_rate_transactions):
    """Test return-rate validation and failure scenarios."""
    import warnings

    import numpy as np
    import pandas as pd
    import pytest

    warnings.filterwarnings(
        "ignore",
        message="Could not infer format.*",
        category=UserWarning,
    )

    # 1. Use the same transaction fixture as the creation test.
    transactions_df = sample_return_rate_transactions.copy(deep=True)
    cutoff = pd.Timestamp("2026-03-15")

    # Generate expected feature values using the production creation function.
    # This keeps the validation test consistent with the feature definition.
    expected_features = create_return_rate_features(
        transactions_df,
        cutoff_date=cutoff,
    )

    result = validate_return_rate_features(
        expected_features,
        transactions_df,
        cutoff_date=cutoff,
    )

    # 2. Validate successful checks.
    expected_true_checks = [
        "customer_id_column_present",
        "return_rate_column_present",
        "customer_id_is_unique",
        "no_missing_customer_id",
        "no_missing_return_rate",
        "return_rate_is_numeric",
        "no_negative_return_rate",
        "customer_count_matches",
        "customer_population_matches",
        "all_expected_return_rates_present",
        "return_rates_match",
        "no_return_rate_above_expected_domain",
        "cutoff_before_max_invoice_date",
        "no_source_transaction_after_cutoff_used",
        "passed",
    ]

    for check in expected_true_checks:
        assert bool(result["validation"][check]) is True, (
            f"Expected validation check '{check}' to pass; "
            f"failed checks: {result['failed_checks']}"
        )

    assert result["failed_checks"] == []

    # 3. Validate summary statistics against the created feature table.
    expected_rates = expected_features["Return_Rate"]

    assert result["summary"]["customer_count"] == len(expected_features)
    assert result["summary"]["expected_customer_count"] == len(expected_features)

    assert result["summary"]["mean_return_rate"] == pytest.approx(
        expected_rates.mean()
    )
    assert result["summary"]["median_return_rate"] == pytest.approx(
        expected_rates.median()
    )
    assert result["summary"]["minimum_return_rate"] == pytest.approx(
        expected_rates.min()
    )
    assert result["summary"]["maximum_return_rate"] == pytest.approx(
        expected_rates.max()
    )

    assert result["summary"]["max_invoice_date"] == pd.Timestamp(
        transactions_df["InvoiceDate"].max()
    )
    assert result["summary"]["cutoff_date"] == cutoff

    # 4. Missing required source columns raise ValueError.
    with pytest.raises(
        ValueError,
        match="Missing required columns in df_cleaned",
    ):
        validate_return_rate_features(
            expected_features,
            transactions_df.drop(columns=["Invoice"]),
            cutoff_date=cutoff,
        )

    # 5. Missing required feature columns raise ValueError.
    with pytest.raises(
        ValueError,
        match="Missing required columns in return_rate_features",
    ):
        validate_return_rate_features(
            expected_features.drop(columns=["Return_Rate"]),
            transactions_df,
            cutoff_date=cutoff,
        )

    # 6. Invalid source dates raise ValueError.
    invalid_dates_df = transactions_df.copy()
    invalid_dates_df["InvoiceDate"] = "not-a-date"

    with pytest.raises(
        ValueError,
        match="InvoiceDate contains missing or invalid values",
    ):
        validate_return_rate_features(
            expected_features,
            invalid_dates_df,
            cutoff_date=cutoff,
        )

    # 7. Missing feature customer IDs currently raise an exception.
    # The current validator casts IDs to int64 before completing its checks.
    missing_id_df = expected_features.copy()
    missing_id_df.loc[missing_id_df.index[0], "Customer ID"] = np.nan

    with pytest.raises(
        (ValueError, TypeError, OverflowError),
    ):
        validate_return_rate_features(
            missing_id_df,
            transactions_df,
            cutoff_date=cutoff,
        )

    # 8. Duplicate customer IDs currently cause the one-to-one merge to fail.
    duplicate_id_df = pd.concat(
        [expected_features, expected_features.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        (ValueError, pd.errors.MergeError),
    ):
        validate_return_rate_features(
            duplicate_id_df,
            transactions_df,
            cutoff_date=cutoff,
        )

    # 9. Missing return rates are detected.
    missing_rate_df = expected_features.copy()
    missing_rate_df.loc[missing_rate_df.index[0], "Return_Rate"] = np.nan

    result = validate_return_rate_features(
        missing_rate_df,
        transactions_df,
        cutoff_date=cutoff,
    )

    assert bool(result["validation"]["no_missing_return_rate"]) is False
    assert bool(result["validation"]["all_expected_return_rates_present"]) is False
    assert bool(result["validation"]["return_rates_match"]) is False
    assert bool(result["validation"]["passed"]) is False

    # 10. Negative return rates are detected.
    negative_rate_df = expected_features.copy()
    negative_rate_df.loc[negative_rate_df.index[0], "Return_Rate"] = -0.5

    result = validate_return_rate_features(
        negative_rate_df,
        transactions_df,
        cutoff_date=cutoff,
    )

    assert bool(result["validation"]["no_negative_return_rate"]) is False
    assert bool(
        result["validation"]["no_return_rate_above_expected_domain"]
    ) is False
    assert bool(result["validation"]["passed"]) is False

    # 11. Incorrect return-rate values are detected.
    incorrect_rate_df = expected_features.copy()
    original_rate = incorrect_rate_df.loc[
        incorrect_rate_df.index[0], "Return_Rate"
    ]
    incorrect_rate_df.loc[
        incorrect_rate_df.index[0], "Return_Rate"
    ] = original_rate + 0.25

    result = validate_return_rate_features(
        incorrect_rate_df,
        transactions_df,
        cutoff_date=cutoff,
    )

    assert bool(result["validation"]["return_rates_match"]) is False
    assert bool(result["validation"]["passed"]) is False
    assert "return_rates_match" in result["failed_checks"]

    # 12. Customer population mismatches are detected.
    missing_customer_df = expected_features.iloc[:-1].copy()

    result = validate_return_rate_features(
        missing_customer_df,
        transactions_df,
        cutoff_date=cutoff,
    )

    assert bool(result["validation"]["customer_count_matches"]) is False
    assert bool(result["validation"]["customer_population_matches"]) is False
    assert bool(result["validation"]["passed"]) is False

    # 13. Return rates greater than one are valid when the source data
    # contains more distinct return orders than purchase orders.
    high_rate_transactions = pd.DataFrame(
        {
            "Customer ID": [201] * 5,
            "InvoiceDate": pd.to_datetime(
                [
                    "2026-03-01",
                    "2026-03-02",
                    "2026-03-03",
                    "2026-03-04",
                    "2026-03-05",
                ]
            ),
            "Quantity": [1, -1, -1, -1, -1],
            "Transaction Type": [
                "Purchase",
                "Return",
                "Return",
                "Return",
                "Return",
            ],
            "Include In Sales Analysis": [True] * 5,
            "Invoice": ["P201", "R201", "R202", "R203", "R204"],
        }
    )

    high_rate_features = pd.DataFrame(
        {
            "Customer ID": [201],
            "Return_Rate": [4.0],
        }
    )

    high_rate_result = validate_return_rate_features(
       high_rate_features,
       high_rate_transactions,
       cutoff_date=pd.Timestamp("2026-03-05"),
     )

    assert bool(high_rate_result["validation"]["no_negative_return_rate"]) is True
    assert bool(high_rate_result["validation"]["return_rates_match"]) is True
    print("High-rate validation:", high_rate_result["validation"])
    print("High-rate failed checks:", high_rate_result["failed_checks"])
    assert bool(high_rate_result["validation"]["passed"]) is True
    assert high_rate_result["summary"]["maximum_return_rate"] == pytest.approx(4.0)

    # 14. Automatic cutoff is max invoice date minus 90 days.
    automatic_result = validate_return_rate_features(
        expected_features,
        transactions_df,
    )

    expected_automatic_cutoff = (
        pd.to_datetime(transactions_df["InvoiceDate"]).max()
        - pd.Timedelta(days=90)
    )

    assert (
        automatic_result["summary"]["cutoff_date"]
        == expected_automatic_cutoff
    )


#================================================================================================================
#-------------------- 9.Testing create_discount_markdown_sensitivity_features------------------------------------
#================================================================================================================
from src.features.features import create_discount_markdown_sensitivity_features
from src.features.features import validate_discount_markdown_sensitivity_features
@pytest.fixture
def sample_discount_sensitivity_transactions():
    return pd.DataFrame(
        {
            "Customer ID": [
                101, 101, 101, 101,
                202, 202,
                303, 303,
                404, 0,
            ],
            "Invoice": [
                "I001", "I002", "I003", "I004",
                "I005", "I006",
                "I007", "I008",
                "I009", "I010",
            ],
            "StockCode": [
                "P001", "P001", "P001", "P002",
                "P001", "P001",
                "D", "D",
                "P002", "P001",
            ],
            "InvoiceDate": [
                "2026-01-01",
                "2026-01-05",
                "2026-01-10",
                "2026-01-15",
                "2026-01-03",
                "2026-01-08",
                "2026-01-02",
                "2026-01-12",
                "2026-01-20",
                "2026-01-21",
            ],
            "Price": [
                10.0, 8.0, 5.0, 20.0,
                7.0, 10.0,
                5.0, 3.0,
                20.0, 1.0,
            ],
            "Quantity": [1] * 10,
            "Transaction Type": [
                "Purchase", "Purchase", "Purchase", "Purchase",
                "Purchase", "Purchase",
                "Purchase", "Purchase",
                "Purchase", "Purchase",
            ],
            "Include In Sales Analysis": [True] * 10,
        }
    )

def test_create_discount_markdown_sensitivity_features(
    sample_discount_sensitivity_transactions,
):
    df = sample_discount_sensitivity_transactions
    original_df = df.copy(deep=True)
    cutoff = pd.Timestamp("2026-01-20")

    result = create_discount_markdown_sensitivity_features(
        df,
        cutoff_date=cutoff,
    )

    # 1. Output structure
    assert list(result.columns) == [
        "Customer ID",
        "Discount_Markdown_Sensitivity",
    ]
    assert not result.empty
    assert result["Customer ID"].is_unique
    assert result["Customer ID"].notna().all()
    assert set(result["Customer ID"]) == {101, 202, 303, 404}

    # 2. Independently verify customer-level calculations.
    # P001 median price is 8.0.
    # Customer 101 deviations: 0, 0, 0.30 after clipping.
    # P002 has only one observation, so its deviation is zero.
    # Customer 101's mean sensitivity is 0.075.
    sensitivities = result.set_index("Customer ID")[
        "Discount_Markdown_Sensitivity"
    ]

    assert sensitivities.loc[101] == pytest.approx(0.075)
    assert sensitivities.loc[202] == pytest.approx(0.0625)
    assert sensitivities.loc[303] == pytest.approx(0.0)
    assert sensitivities.loc[404] == pytest.approx(0.0)

    # 3. Sensitivity values must be finite and within [0, 0.30].
    values = result["Discount_Markdown_Sensitivity"]
    assert values.notna().all()
    assert np.isfinite(values.to_numpy()).all()
    assert values.ge(0).all()
    assert values.le(0.30).all()

    # 4. Customer IDs must be integers and sensitivity values floats.
    assert pd.api.types.is_integer_dtype(result["Customer ID"])
    assert pd.api.types.is_float_dtype(
        result["Discount_Markdown_Sensitivity"]
    )

    # 5. Input data must remain unchanged.
    pd.testing.assert_frame_equal(df, original_df)

    # 6. Missing required column must raise ValueError.
    invalid_df = df.drop(columns=["Price"])

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        create_discount_markdown_sensitivity_features(
            invalid_df,
            cutoff_date=cutoff,
        )

    # 7. A cutoff equal to the maximum invoice date is invalid.
    with pytest.raises(
        ValueError,
        match="cutoff_date must be earlier",
    ):
        create_discount_markdown_sensitivity_features(
            df,
            cutoff_date=pd.Timestamp("2026-01-21"),
        )

    # 8. Invalid dates become NaT; if no valid date remains,
    # the function must raise its documented ValueError.
    import warnings
    invalid_dates = df.copy()
    invalid_dates["InvoiceDate"] = "not-a-date"

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=( r"Could not infer format, so each element will be parsed "
                    r"individually, falling back to `dateutil`"), category=UserWarning,)

        with pytest.raises( ValueError, match="InvoiceDate contains no valid dates",):
            create_discount_markdown_sensitivity_features(invalid_dates,cutoff_date=cutoff,)

    # 9. No qualifying historical customer transactions.
    with pytest.raises(
        ValueError,
        match="No qualifying historical customer transactions",
    ):
        create_discount_markdown_sensitivity_features(
            df,
            cutoff_date=pd.Timestamp("2025-01-01"),
        )

    # 10. Non-product stock codes must not contribute to price analysis.
    non_product_df = df.copy()
    non_product_df.loc[
        non_product_df["Customer ID"] == 101,
        "StockCode",
    ] = "D"

    non_product_result = (
        create_discount_markdown_sensitivity_features(
            non_product_df,
            cutoff_date=cutoff,
        )
    )
    non_product_sensitivities = non_product_result.set_index(
        "Customer ID"
    )["Discount_Markdown_Sensitivity"]

    assert non_product_sensitivities.loc[101] == pytest.approx(0.0)

    # 11. Zero or negative prices must not contribute to price analysis.
    invalid_prices_df = df.copy()
    invalid_prices_df.loc[
        invalid_prices_df["Customer ID"] == 101,
        "Price",
    ] = 0.0

    invalid_prices_result = (
        create_discount_markdown_sensitivity_features(
            invalid_prices_df,
            cutoff_date=cutoff,
        )
    )
    invalid_prices_sensitivities = invalid_prices_result.set_index(
        "Customer ID"
    )["Discount_Markdown_Sensitivity"]

    assert invalid_prices_sensitivities.loc[101] == pytest.approx(0.0)

def test_validate_discount_markdown_sensitivity_features(
    sample_discount_sensitivity_transactions,
):
    """Test discount/markdown sensitivity validation and failure scenarios."""
    import numpy as np
    import pandas as pd
    import pytest
    import warnings

    warnings.filterwarnings("ignore", message="Could not infer format.*",category=UserWarning,)

    # 1. Prepare source data and cutoff.
    transactions_df = (
        sample_discount_sensitivity_transactions.copy(deep=True)
    )

    maximum_date = pd.to_datetime(
        transactions_df["InvoiceDate"]
    ).max()

    cutoff = maximum_date - pd.Timedelta(days=1)

    # Generate expected features using the production function.
    expected_features = create_discount_markdown_sensitivity_features(
        transactions_df,
        cutoff_date=cutoff,
    )

    # 2. Validate correctly generated features.
    result = validate_discount_markdown_sensitivity_features(
        expected_features,
        transactions_df,
        cutoff_date=cutoff,
    )

    expected_true_checks = [
        "customer_id_column_present",
        "discount_markdown_sensitivity_column_present",
        "feature_output_not_empty",
        "customer_id_is_unique",
        "no_missing_customer_id",
        "no_missing_sensitivity",
        "sensitivity_is_numeric",
        "cutoff_before_max_invoice_date",
        "customer_count_matches",
        "customer_population_matches",
        "no_source_transaction_after_cutoff_used",
        "qualifying_price_transactions_present",
        "product_median_prices_present",
        "product_median_prices_positive",
        "customer_ids_aligned",
        "sensitivity_values_match",
        "no_negative_sensitivity",
        "sensitivity_within_30pct_cap",
        "customers_without_price_analysis_are_zero",
        "expected_customer_feature_count_matches",
        "passed",
    ]

    for check in expected_true_checks:
        assert bool(result["validation"][check]) is True, (
            f"Expected '{check}' to pass. "
            f"Failed checks: {result['failed_checks']}"
        )

    assert result["failed_checks"] == []

    # 3. Validate summary statistics.
    actual_values = expected_features[
        "Discount_Markdown_Sensitivity"
    ]

    summary = result["summary"]

    assert summary["customer_count"] == len(expected_features)
    assert summary["expected_customer_count"] == len(expected_features)

    assert summary["minimum_sensitivity"] == pytest.approx(
        actual_values.min()
    )
    assert summary["median_sensitivity"] == pytest.approx(
        actual_values.median()
    )
    assert summary["mean_sensitivity"] == pytest.approx(
        actual_values.mean()
    )
    assert summary["maximum_sensitivity"] == pytest.approx(
        actual_values.max()
    )

    assert summary["cutoff_date"] == cutoff
    assert summary["maximum_absolute_difference"] == pytest.approx(
        0.0, abs=1e-12
    )

    # 4. Missing required source column.
    with pytest.raises(
        ValueError,
        match="df_cleaned is missing required columns",
    ):
        validate_discount_markdown_sensitivity_features(
            expected_features,
            transactions_df.drop(columns=["Invoice"]),
            cutoff_date=cutoff,
        )

    # 5. Missing required feature column.
    missing_feature_column = expected_features.drop(
        columns=["Discount_Markdown_Sensitivity"]
    )

    result = validate_discount_markdown_sensitivity_features(
        missing_feature_column,
        transactions_df,
        cutoff_date=cutoff,
    )

    assert bool(result["validation"]["passed"]) is False
    assert result["summary"] == {}
    assert any(
        "Missing output columns" in item
        for item in result["failed_checks"]
    )

    # 6. Invalid source dates.
    invalid_dates_df = transactions_df.copy()
    invalid_dates_df["InvoiceDate"] = "not-a-date"

    with pytest.raises(
        ValueError,
        match="InvoiceDate contains no valid dates",
    ):
        validate_discount_markdown_sensitivity_features(
            expected_features,
            invalid_dates_df,
            cutoff_date=cutoff,
        )

    # 7. Missing customer ID.
    missing_id_df = expected_features.copy()
    missing_id_df.loc[
        missing_id_df.index[0], "Customer ID"
    ] = np.nan

    with pytest.raises((ValueError, TypeError, OverflowError)):
        validate_discount_markdown_sensitivity_features(
            missing_id_df,
            transactions_df,
            cutoff_date=cutoff,
        )

    # 8. Duplicate customer IDs.
    duplicate_id_df = pd.concat(
        [expected_features, expected_features.iloc[[0]]],
        ignore_index=True,
    )

    result = validate_discount_markdown_sensitivity_features(
        duplicate_id_df,
        transactions_df,
        cutoff_date=cutoff,
    )

    assert bool(result["validation"]["customer_id_is_unique"]) is False
    assert bool(result["validation"]["customer_count_matches"]) is False
    assert bool(result["validation"]["passed"]) is False

    # 9. Missing sensitivity values.
    missing_sensitivity_df = expected_features.copy()
    missing_sensitivity_df.loc[
        missing_sensitivity_df.index[0],
        "Discount_Markdown_Sensitivity",
    ] = np.nan

    result = validate_discount_markdown_sensitivity_features(
        missing_sensitivity_df,
        transactions_df,
        cutoff_date=cutoff,
    )

    assert bool(
        result["validation"]["no_missing_sensitivity"]
    ) is False
    assert bool(
        result["validation"]["sensitivity_values_match"]
    ) is False
    assert bool(result["validation"]["passed"]) is False

    # 10. Negative sensitivity.
    negative_sensitivity_df = expected_features.copy()
    negative_sensitivity_df.loc[
        negative_sensitivity_df.index[0],
        "Discount_Markdown_Sensitivity",
    ] = -0.01

    result = validate_discount_markdown_sensitivity_features(
        negative_sensitivity_df,
        transactions_df,
        cutoff_date=cutoff,
    )

    assert bool(
        result["validation"]["no_negative_sensitivity"]
    ) is False
    assert bool(
        result["validation"]["sensitivity_values_match"]
    ) is False
    assert bool(result["validation"]["passed"]) is False

    # 11. Sensitivity above the 30% cap.
    excessive_sensitivity_df = expected_features.copy()
    excessive_sensitivity_df.loc[
        excessive_sensitivity_df.index[0],
        "Discount_Markdown_Sensitivity",
    ] = 0.31

    result = validate_discount_markdown_sensitivity_features(
        excessive_sensitivity_df,
        transactions_df,
        cutoff_date=cutoff,
    )

    assert bool(
        result["validation"]["sensitivity_within_30pct_cap"]
    ) is False
    assert bool(
        result["validation"]["sensitivity_values_match"]
    ) is False
    assert bool(result["validation"]["passed"]) is False

    # 12. Incorrect sensitivity values.
    incorrect_sensitivity_df = expected_features.copy()
    first_index = incorrect_sensitivity_df.index[0]

    original_value = float(
        incorrect_sensitivity_df.loc[
            first_index, "Discount_Markdown_Sensitivity"
        ]
    )

    # Change the value while keeping it within the permitted range.
    if original_value < 0.30:
        changed_value = min(original_value + 0.05, 0.30)
    else:
        changed_value = max(original_value - 0.05, 0.0)

    incorrect_sensitivity_df.loc[
        first_index, "Discount_Markdown_Sensitivity"
    ] = changed_value

    assert not np.isclose(changed_value, original_value), (
        "Test setup error: the sensitivity value was not changed."
    )

    result = validate_discount_markdown_sensitivity_features(
        incorrect_sensitivity_df,
        transactions_df,
        cutoff_date=cutoff,
    )

    assert bool(
        result["validation"]["sensitivity_values_match"]
    ) is False
    assert bool(result["validation"]["passed"]) is False
    assert "sensitivity_values_match" in result["failed_checks"]

    # 13. Missing customer population.
    missing_customer_df = expected_features.iloc[:-1].copy()

    result = validate_discount_markdown_sensitivity_features(
        missing_customer_df,
        transactions_df,
        cutoff_date=cutoff,
    )

    assert bool(
        result["validation"]["customer_count_matches"]
    ) is False
    assert bool(
        result["validation"]["customer_population_matches"]
    ) is False
    assert bool(result["validation"]["passed"]) is False

    # 14. Automatic cutoff calculation.
    automatic_result = validate_discount_markdown_sensitivity_features(
        expected_features,
        transactions_df,
    )

    expected_automatic_cutoff = (
        maximum_date - pd.Timedelta(days=90)
    )

    assert (
        automatic_result["summary"]["cutoff_date"]
        == expected_automatic_cutoff
    )
    assert bool(
        automatic_result["validation"]["cutoff_before_max_invoice_date"]
    ) is True

    # 15. Cutoff equal to the maximum invoice date must fail validation.
    maximum_date_result = (
        validate_discount_markdown_sensitivity_features(
            expected_features,
            transactions_df,
            cutoff_date=maximum_date,
        )
    )

    assert bool(
        maximum_date_result["validation"]["cutoff_before_max_invoice_date"]
    ) is False
    assert bool(maximum_date_result["validation"]["passed"]) is False
    assert (
        "cutoff_before_max_invoice_date"
        in maximum_date_result["failed_checks"]
    )

def test_analyze_discount_markdown_sensitivity(
    sample_discount_sensitivity_transactions,
):

    import pandas as pd
    import numpy as np
    import pytest
    import warnings

    from src.features.features import analyze_discount_markdown_sensitivity

    # ---------------------------------------------------------
    # Helper: execute function with targeted warning suppression
    # ---------------------------------------------------------
    def run_analysis(data, cutoff_date=None):

        with warnings.catch_warnings():

            warnings.filterwarnings(
                "ignore",
                message="Could not infer format.*",
                category=UserWarning,
            )

            return analyze_discount_markdown_sensitivity(
                df_cleaned=data,
                cutoff_date=cutoff_date,
            )

    # ---------------------------------------------------------
    # 1. Prepare test data
    # ---------------------------------------------------------
    df = sample_discount_sensitivity_transactions.copy(deep=True)

    # Required production column, absent from supplied fixture.
    df["Analytical Transaction Value"] = (
        df["Price"] * df["Quantity"]
    )

    cutoff_date = pd.Timestamp("2026-01-20")

    # ---------------------------------------------------------
    # 2. Execute production function
    # ---------------------------------------------------------
    result = run_analysis(
        data=df,
        cutoff_date=cutoff_date,
    )

    # ---------------------------------------------------------
    # 3. Check returned package structure
    # ---------------------------------------------------------
    expected_keys = {
        "summary",
        "validation",
        "failed_checks",
        "customer_behavior",
        "product_price_benchmark",
        "transaction_price_analysis",
        "threshold_summary",
        "behavioral_band_summary",
        "transaction_value_relationship",
    }

    assert set(result.keys()) == expected_keys

    # ---------------------------------------------------------
    # 4. Check summary
    # ---------------------------------------------------------
    summary = result["summary"]

    assert summary["max_invoice_date"] == pd.Timestamp("2026-01-21")

    assert summary["cutoff_date"] == cutoff_date

    assert summary["historical_customer_population"] == 4

    assert summary["customers_with_price_analysis"] == 3

    assert summary["customers_without_price_analysis"] == 1

    assert summary["product_count"] == 2

    assert summary["products_with_multiple_prices"] == 1

    assert summary["transactions_analyzed"] == 7

    assert summary["customers_with_any_below_median"] == 2

    assert summary["customers_with_any_20pct_exposure"] == 1

    assert summary["customers_with_any_30pct_exposure"] == 1

    assert summary["customers_with_repeated_20pct_exposure"] == 0

    assert summary["customers_with_repeated_30pct_exposure"] == 0

    assert summary["mean_capped_deviation"] == pytest.approx(
        0.034375
    )

    assert summary["median_capped_deviation"] == pytest.approx(
        0.03125
    )

    assert summary["maximum_capped_deviation"] == pytest.approx(
        0.075
    )

    # ---------------------------------------------------------
    # 5. Check product-level price benchmarks
    # ---------------------------------------------------------
    benchmark = result["product_price_benchmark"]

    assert len(benchmark) == 2

    assert set(benchmark["StockCode_Key"]) == {
        "P001",
        "P002",
    }

    p001 = benchmark.loc[
        benchmark["StockCode_Key"] == "P001"
    ].iloc[0]

    assert p001["Price_Observation_Count"] == 5

    assert p001["Distinct_Price_Count"] == 4

    assert p001["Product_Median_Price"] == pytest.approx(8.0)

    assert p001["Product_Minimum_Price"] == pytest.approx(5.0)

    assert p001["Product_Maximum_Price"] == pytest.approx(10.0)

    p002 = benchmark.loc[
        benchmark["StockCode_Key"] == "P002"
    ].iloc[0]

    assert p002["Price_Observation_Count"] == 2

    assert p002["Distinct_Price_Count"] == 1

    assert p002["Product_Median_Price"] == pytest.approx(20.0)

    assert p002["Product_Minimum_Price"] == pytest.approx(20.0)

    assert p002["Product_Maximum_Price"] == pytest.approx(20.0)

    # ---------------------------------------------------------
    # 6. Check transaction-level price analysis
    # ---------------------------------------------------------
    transactions = result["transaction_price_analysis"]

    assert len(transactions) == 7

    assert set(transactions["Customer ID"]) == {
        101,
        202,
        404,
    }

    # Administrative StockCode D must be excluded.
    assert "D" not in transactions["StockCode_Key"].tolist()

    # Customer ID 0 must be excluded.
    assert 0 not in transactions["Customer ID"].tolist()

    assert bool(
        transactions["StockCode_Key"].notna().all()
    ) is True

    # All retained transactions must have positive prices.
    assert bool(
        (transactions["Price"] > 0).all()
    ) is True

    # ---------------------------------------------------------
    # 7. Check price deviation calculations
    # ---------------------------------------------------------
    customer_101_low_price = transactions.loc[
        (transactions["Customer ID"] == 101)
        & (transactions["Price"] == 5.0)
    ].iloc[0]

    expected_deviation = (8.0 - 5.0) / 8.0

    assert customer_101_low_price[
        "Price_Deviation_From_Median"
    ] == pytest.approx(expected_deviation)

    assert customer_101_low_price[
        "Below_Median_Price_Deviation"
    ] == pytest.approx(expected_deviation)

    # Positive deviation is capped at 30%.
    assert customer_101_low_price[
        "Capped_Below_Median_Deviation"
    ] == pytest.approx(0.30)

    assert bool(
        customer_101_low_price["Below_10pct_Median"]
    ) is True

    assert bool(
        customer_101_low_price["Below_20pct_Median"]
    ) is True

    assert bool(
        customer_101_low_price["Below_30pct_Median"]
    ) is True

    # ---------------------------------------------------------
    # 8. Check price equal to median
    # ---------------------------------------------------------
    customer_101_median_price = transactions.loc[
        (transactions["Customer ID"] == 101)
        & (transactions["Price"] == 8.0)
    ].iloc[0]

    assert customer_101_median_price[
        "Product_Median_Price"
    ] == pytest.approx(8.0)

    assert customer_101_median_price[
        "Below_Median_Price_Deviation"
    ] == pytest.approx(0.0)

    assert customer_101_median_price[
        "Capped_Below_Median_Deviation"
    ] == pytest.approx(0.0)

    assert bool(
        customer_101_median_price["Below_10pct_Median"]
    ) is False

    assert bool(
        customer_101_median_price["Below_20pct_Median"]
    ) is False

    assert bool(
        customer_101_median_price["Below_30pct_Median"]
    ) is False

    # ---------------------------------------------------------
    # 9. Check price above median
    # ---------------------------------------------------------
    customer_101_high_price = transactions.loc[
        (transactions["Customer ID"] == 101)
        & (transactions["Price"] == 10.0)
    ].iloc[0]

    assert customer_101_high_price[
        "Below_Median_Price_Deviation"
    ] == pytest.approx(0.0)

    assert customer_101_high_price[
        "Capped_Below_Median_Deviation"
    ] == pytest.approx(0.0)

    assert bool(
        customer_101_high_price["Below_10pct_Median"]
    ) is False

    # ---------------------------------------------------------
    # 10. Check threshold value calculations
    # ---------------------------------------------------------
    assert customer_101_low_price[
        "Below_20pct_Value"
    ] == pytest.approx(5.0)

    assert customer_101_low_price[
        "Below_30pct_Value"
    ] == pytest.approx(5.0)

    assert customer_101_low_price[
        "Capped_Deviation_Value"
    ] == pytest.approx(1.5)

    # ---------------------------------------------------------
    # 11. Check customer-level population
    # ---------------------------------------------------------
    customers = result["customer_behavior"]

    assert len(customers) == 4

    assert customers["Customer ID"].is_unique

    assert set(customers["Customer ID"]) == {
        101,
        202,
        303,
        404,
    }

    # ---------------------------------------------------------
    # 12. Check customer 101
    # ---------------------------------------------------------
    customer_101 = customers.loc[
        customers["Customer ID"] == 101
    ].iloc[0]

    assert customer_101["Price_Transaction_Count"] == 4

    assert customer_101["Price_Purchase_Order_Count"] == 4

    assert customer_101["Total_Purchase_Value"] == pytest.approx(43.0)

    assert customer_101["Mean_Below_Median_Deviation"] == pytest.approx(
        0.09375
    )

    assert customer_101[
        "Mean_Capped_Below_Median_Deviation"
    ] == pytest.approx(0.075)

    assert customer_101["Median_Below_Median_Deviation"] == pytest.approx(
        0.0
    )

    assert customer_101["Transactions_Below_Median"] == 1

    assert customer_101["Transactions_Below_10pct"] == 1

    assert customer_101["Transactions_Below_20pct"] == 1

    assert customer_101["Transactions_Below_30pct"] == 1

    assert customer_101["Below_Median_Transaction_Share"] == pytest.approx(
        0.25
    )

    assert customer_101["Below_10pct_Transaction_Share"] == pytest.approx(
        0.25
    )

    assert customer_101["Below_20pct_Transaction_Share"] == pytest.approx(
        0.25
    )

    assert customer_101["Below_30pct_Transaction_Share"] == pytest.approx(
        0.25
    )

    assert customer_101["Value_Below_20pct"] == pytest.approx(5.0)

    assert customer_101["Value_Below_30pct"] == pytest.approx(5.0)

    assert customer_101["Below_20pct_Value_Share"] == pytest.approx(
        5.0 / 43.0
    )

    assert customer_101["Below_30pct_Value_Share"] == pytest.approx(
        5.0 / 43.0
    )

    assert customer_101["Capped_Deviation_Value"] == pytest.approx(1.5)

    assert customer_101["Capped_Deviation_Value_Share"] == pytest.approx(
        1.5 / 43.0
    )

    assert bool(customer_101["Has_Price_Analysis"]) is True

    assert bool(
        customer_101["Repeated_20pct_Below_Median"]
    ) is False

    assert bool(
        customer_101["Repeated_30pct_Below_Median"]
    ) is False

    # ---------------------------------------------------------
    # 13. Check customer 202
    # ---------------------------------------------------------
    customer_202 = customers.loc[
        customers["Customer ID"] == 202
    ].iloc[0]

    assert customer_202["Price_Transaction_Count"] == 2

    assert customer_202["Price_Purchase_Order_Count"] == 2

    assert customer_202["Total_Purchase_Value"] == pytest.approx(17.0)

    assert customer_202["Mean_Below_Median_Deviation"] == pytest.approx(
        0.0625
    )

    assert customer_202[
        "Mean_Capped_Below_Median_Deviation"
    ] == pytest.approx(0.0625)

    assert customer_202["Transactions_Below_Median"] == 1

    assert customer_202["Transactions_Below_10pct"] == 1

    assert customer_202["Transactions_Below_20pct"] == 0

    assert customer_202["Transactions_Below_30pct"] == 0

    assert customer_202["Below_Median_Transaction_Share"] == pytest.approx(
        0.50
    )

    assert customer_202["Below_10pct_Transaction_Share"] == pytest.approx(
        0.50
    )

    assert customer_202["Below_20pct_Transaction_Share"] == pytest.approx(
        0.0
    )

    assert customer_202["Below_30pct_Transaction_Share"] == pytest.approx(
        0.0
    )

    assert customer_202["Below_20pct_Value_Share"] == pytest.approx(0.0)

    assert bool(customer_202["Has_Price_Analysis"]) is True

    # ---------------------------------------------------------
    # 14. Check customer 303 without eligible price analysis
    # ---------------------------------------------------------
    customer_303 = customers.loc[
        customers["Customer ID"] == 303
    ].iloc[0]

    assert customer_303["Price_Transaction_Count"] == 0

    assert customer_303["Price_Purchase_Order_Count"] == 0

    assert customer_303["Total_Purchase_Value"] == pytest.approx(0.0)

    assert customer_303["Below_Median_Transaction_Share"] == pytest.approx(
        0.0
    )

    assert customer_303["Below_20pct_Transaction_Share"] == pytest.approx(
        0.0
    )

    assert customer_303["Below_20pct_Value_Share"] == pytest.approx(0.0)

    assert bool(customer_303["Has_Price_Analysis"]) is False

    assert bool(
        customer_303["Repeated_20pct_Below_Median"]
    ) is False

    # ---------------------------------------------------------
    # 15. Check customer 404
    # ---------------------------------------------------------
    customer_404 = customers.loc[
        customers["Customer ID"] == 404
    ].iloc[0]

    assert customer_404["Price_Transaction_Count"] == 1

    assert customer_404["Price_Purchase_Order_Count"] == 1

    assert customer_404["Total_Purchase_Value"] == pytest.approx(20.0)

    assert customer_404["Transactions_Below_Median"] == 0

    assert customer_404["Below_20pct_Transaction_Share"] == pytest.approx(
        0.0
    )

    assert bool(customer_404["Has_Price_Analysis"]) is True

    # ---------------------------------------------------------
    # 16. Check capped deviation validation
    # ---------------------------------------------------------
    assert bool(
        customers["Mean_Capped_Below_Median_Deviation"]
        .between(0, 0.30)
        .all()
    ) is True

    assert bool(
        customers["Capped_Deviation_Value_Share"]
        .between(0, 1)
        .all()
    ) is True

    # ---------------------------------------------------------
    # 17. Check repeated behaviour flags
    # ---------------------------------------------------------
    assert bool(
        customers.loc[
            customers["Customer ID"] == 101,
            "Repeated_20pct_Below_Median",
        ].iloc[0]
    ) is False

    assert bool(
        customers.loc[
            customers["Customer ID"] == 202,
            "Repeated_30pct_Below_Median",
        ].iloc[0]
    ) is False

    # ---------------------------------------------------------
    # 18. Check behavioral bands
    # ---------------------------------------------------------
    assert customer_101["Below_20pct_Behavior_Band"] == (
        "Moderate Exposure (10-25%)"
    )

    assert customer_202["Below_20pct_Behavior_Band"] == (
        "No Exposure"
    )

    assert customer_303["Below_20pct_Behavior_Band"] == (
        "No Exposure"
    )

    assert customer_404["Below_20pct_Behavior_Band"] == (
        "No Exposure"
    )

    # ---------------------------------------------------------
    # 19. Check threshold summary
    # ---------------------------------------------------------
    threshold_summary = result["threshold_summary"]

    assert threshold_summary["Threshold"].tolist() == [
        "Below Median",
        "At Least 10% Below Median",
        "At Least 20% Below Median",
        "At Least 30% Below Median",
    ]

    assert threshold_summary["Customers_With_Exposure"].tolist() == [
        2,
        2,
        1,
        1,
    ]

    assert threshold_summary["Customer_Percentage"].tolist() == pytest.approx(
        [50.0, 50.0, 25.0, 25.0]
    )

    # ---------------------------------------------------------
    # 20. Check behavioral band summary
    # ---------------------------------------------------------
    band_summary = result["behavioral_band_summary"]

    band_counts = band_summary.set_index(
        "Behavior_Band"
    )["Customer_Count"]

    assert band_counts["No Exposure"] == 3

    assert band_counts["Moderate Exposure (10-25%)"] == 1

    assert band_summary["Customer_Count"].sum() == 4

    assert band_summary["Customer_Percentage"].sum() == pytest.approx(
        100.0
    )

    # ---------------------------------------------------------
    # 21. Check transaction-value relationships
    # ---------------------------------------------------------
    relationships = result["transaction_value_relationship"]

    assert set(relationships.keys()) == {
        "below_20pct_transaction_share_vs_value_share",
        "below_30pct_transaction_share_vs_value_share",
        "capped_deviation_vs_20pct_transaction_share",
    }

    for value in relationships.values():

        assert pd.isna(value) or -1 <= value <= 1

    # ---------------------------------------------------------
    # 22. Check validation
    # ---------------------------------------------------------
    validation = result["validation"]

    expected_validation_keys = {
        "customer_population_is_5281_or_expected",
        "customer_ids_unique",
        "customer_ids_not_missing",
        "customer_population_matches",
        "price_analysis_customers_subset_of_population",
        "capped_deviation_valid",
        "transaction_shares_valid",
        "value_shares_valid",
        "threshold_counts_consistent",
        "no_missing_behavior_values",
        "passed",
    }

    assert set(validation.keys()) == expected_validation_keys

    for key, value in validation.items():

        assert isinstance(value, (bool, np.bool_))

    assert bool(
        validation["customer_population_is_5281_or_expected"]
    ) is True

    assert bool(validation["customer_ids_unique"]) is True

    assert bool(validation["customer_ids_not_missing"]) is True

    assert bool(validation["customer_population_matches"]) is True

    assert bool(
        validation["price_analysis_customers_subset_of_population"]
    ) is True

    assert bool(validation["capped_deviation_valid"]) is True

    assert bool(validation["transaction_shares_valid"]) is True

    assert bool(validation["value_shares_valid"]) is True

    assert bool(validation["threshold_counts_consistent"]) is True

    assert bool(validation["no_missing_behavior_values"]) is True

    assert bool(validation["passed"]) is True

    assert result["failed_checks"] == []

    # ---------------------------------------------------------
    # 23. Check automatic cutoff calculation
    # ---------------------------------------------------------
    automatic_transactions = df.copy(deep=True)

    automatic_transactions = pd.concat(
        [
            automatic_transactions,
            pd.DataFrame(
                {
                    "Customer ID": [505],
                    "Invoice": ["I011"],
                    "StockCode": ["P003"],
                    "InvoiceDate": ["2025-10-01"],
                    "Price": [6.0],
                    "Quantity": [1],
                    "Transaction Type": ["Purchase"],
                    "Include In Sales Analysis": [True],
                    "Analytical Transaction Value": [6.0],
                }
            ),
        ],
        ignore_index=True,
    )

    automatic_result = run_analysis(
        data=automatic_transactions,
        cutoff_date=None,
    )

    expected_automatic_cutoff = (
        pd.Timestamp("2026-01-21")
        - pd.Timedelta(days=90)
    )

    assert automatic_result["summary"]["cutoff_date"] == (
        expected_automatic_cutoff
    )

    assert automatic_result["summary"][
        "historical_customer_population"
    ] == 1

    assert automatic_result["summary"]["transactions_analyzed"] == 1

    assert automatic_result["summary"][
        "customers_with_price_analysis"
    ] == 1

    assert bool(
        automatic_result["validation"]["passed"]
    ) is True

    # ---------------------------------------------------------
    # 24. Check missing required columns
    # ---------------------------------------------------------
    invalid_df = df.drop(
        columns=["Price"]
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):

        run_analysis(
            data=invalid_df,
            cutoff_date=cutoff_date,
        )

    # ---------------------------------------------------------
    # 25. Check invalid dates
    # ---------------------------------------------------------
    invalid_dates_df = df.copy(deep=True)

    invalid_dates_df["InvoiceDate"] = "invalid-date"

    with pytest.raises(
        ValueError,
        match="InvoiceDate contains no valid dates",
    ):

        run_analysis(
            data=invalid_dates_df,
            cutoff_date=cutoff_date,
        )

    # ---------------------------------------------------------
    # 26. Check cutoff date validation
    # ---------------------------------------------------------
    with pytest.raises(
        ValueError,
        match="cutoff_date must be earlier than max InvoiceDate",
    ):

        run_analysis(
            data=df,
            cutoff_date=pd.Timestamp("2026-01-21"),
        )

    # ---------------------------------------------------------
    # 27. Check no qualifying historical customer transactions
    # ---------------------------------------------------------
    no_customer_transactions = df.copy(deep=True)

    no_customer_transactions["Customer ID"] = 0

    with pytest.raises(
        ValueError,
        match="No qualifying historical customer transactions found",
    ):

        run_analysis(
            data=no_customer_transactions,
            cutoff_date=cutoff_date,
        )

    # ---------------------------------------------------------
    # 28. Check no valid product-price transactions
    # ---------------------------------------------------------
    no_valid_prices = df.copy(deep=True)

    no_valid_prices["StockCode"] = "D"

    with pytest.raises(
        ValueError,
        match="No valid product-price transactions remain",
    ):

        run_analysis(
            data=no_valid_prices,
            cutoff_date=cutoff_date,
        )

def test_analyze_customer_price_deviation(sample_discount_sensitivity_transactions):
    import warnings
    import pandas as pd
    import pytest

    from src.features.features import analyze_customer_price_deviation

    # ============================================================
    # Helper
    # ============================================================

    def run_analysis(data, cutoff=None):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Could not infer format.*",
                category=UserWarning,
            )
            return analyze_customer_price_deviation(
                df_cleaned=data,
                cutoff_date=cutoff,
            )

    # ============================================================
    # 1. Prepare fixture
    # ============================================================

    df = sample_discount_sensitivity_transactions.copy(deep=True)

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce",
    )

    df["Transaction Type"] = "Purchase"
    df["Include In Sales Analysis"] = True

    if "Analytical Transaction Value" not in df.columns:
        df["Analytical Transaction Value"] = (
            df["Price"] * df["Quantity"]
        )

    cutoff = pd.Timestamp("2026-01-20")

    # ============================================================
    # 2. Main analysis
    # ============================================================

    result = run_analysis(df, cutoff)

    assert isinstance(result, dict)

    expected_keys = {
        "summary",
        "validation",
        "failed_checks",
        "product_price_benchmark",
        "transaction_price_analysis",
        "customer_price_exposure",
    }

    assert expected_keys.issubset(result.keys())

    summary = result["summary"]
    validation = result["validation"]
    benchmark = result["product_price_benchmark"]
    transaction_analysis = result["transaction_price_analysis"]
    customer_exposure = result["customer_price_exposure"]

    assert isinstance(summary, dict)
    assert isinstance(validation, dict)
    assert isinstance(benchmark, pd.DataFrame)
    assert isinstance(transaction_analysis, pd.DataFrame)
    assert isinstance(customer_exposure, pd.DataFrame)

    # ============================================================
    # 3. Summary assertions
    # ============================================================

    assert summary["max_invoice_date"] == pd.Timestamp("2026-01-21")
    assert summary["cutoff_date"] == cutoff

    assert summary["qualifying_transaction_count"] == 7
    assert summary["customer_count"] == 3
    assert summary["product_count"] == 2

    assert summary["products_with_multiple_prices"] == 1
    assert summary["products_with_single_price"] == 1

    assert summary["transactions_below_product_median"] == 2
    assert summary["transactions_below_median_percentage"] == pytest.approx(
        2 / 7 * 100
    )

    assert summary["transactions_below_10pct"] == 2
    assert summary["transactions_below_10pct_percentage"] == pytest.approx(
        2 / 7 * 100
    )

    assert summary["transactions_below_20pct"] == 1
    assert summary["transactions_below_20pct_percentage"] == pytest.approx(
        1 / 7 * 100
    )

    assert summary["transactions_below_30pct"] == 1
    assert summary["transactions_below_30pct_percentage"] == pytest.approx(
        1 / 7 * 100
    )

    assert summary["median_below_median_deviation"] == pytest.approx(0)
    assert summary["mean_below_median_deviation"] == pytest.approx(
        0.07142857142857142
    )
    assert summary["maximum_below_median_deviation"] == pytest.approx(0.375)

    assert summary["customers_with_below_median_purchases"] == 2
    assert summary["customers_with_20pct_below_median_purchases"] == 1
    assert summary["customers_with_30pct_below_median_purchases"] == 1

    # ============================================================
    # 4. Product price benchmarks
    # ============================================================

    p001 = benchmark[
        benchmark["StockCode_Key"] == "P001"
    ].iloc[0]

    assert p001["Price_Observation_Count"] == 5
    assert p001["Distinct_Price_Count"] == 4
    assert p001["Product_Median_Price"] == pytest.approx(8)
    assert p001["Product_Minimum_Price"] == pytest.approx(5)
    assert p001["Product_Maximum_Price"] == pytest.approx(10)

    p002 = benchmark[
        benchmark["StockCode_Key"] == "P002"
    ].iloc[0]

    assert p002["Price_Observation_Count"] == 2
    assert p002["Distinct_Price_Count"] == 1
    assert p002["Product_Median_Price"] == pytest.approx(20)

    # ============================================================
    # 5. Transaction-level analysis
    # ============================================================

    assert len(transaction_analysis) == 7
    assert transaction_analysis["Customer ID"].nunique() == 3

    non_product_stock_codes = {
        "23444",
        "23595",
        "ADJUST",
        "ADJUST2",
        "AMAZONFEE",
        "BANK CHARGES",
        "C2",
        "D",
        "DOT",
        "M",
        "POST",
    }

    assert not transaction_analysis["StockCode_Key"].isin(
        non_product_stock_codes
    ).any()

    assert (
        transaction_analysis["InvoiceDate"] <= cutoff
    ).all()

    assert (
        transaction_analysis["Product_Median_Price"] > 0
    ).all()

    assert (
        transaction_analysis["Below_Median_Price_Deviation"] >= 0
    ).all()

    # ============================================================
    # 6. Customer 101 exposure
    # ============================================================

    customer101 = customer_exposure[
        customer_exposure["Customer ID"] == 101
    ].iloc[0]

    assert customer101["Purchase_Transaction_Count"] == 4
    assert customer101["Purchase_Order_Count"] == 4
    assert customer101["Total_Purchase_Value"] == pytest.approx(43)

    assert customer101["Mean_Below_Median_Price_Deviation"] == pytest.approx(
        0.09375
    )

    assert customer101["Median_Below_Median_Price_Deviation"] == pytest.approx(0)

    assert customer101["Transactions_Below_Median"] == 1
    assert customer101["Transactions_Below_10pct"] == 1
    assert customer101["Transactions_Below_20pct"] == 1
    assert customer101["Transactions_Below_30pct"] == 1

    assert customer101["Below_Median_Transaction_Share"] == pytest.approx(0.25)
    assert customer101["Below_10pct_Transaction_Share"] == pytest.approx(0.25)
    assert customer101["Below_20pct_Transaction_Share"] == pytest.approx(0.25)
    assert customer101["Below_30pct_Transaction_Share"] == pytest.approx(0.25)

    assert customer101["Below_20pct_Value_Share"] == pytest.approx(5 / 43)
    assert customer101["Below_30pct_Value_Share"] == pytest.approx(5 / 43)

    # ============================================================
    # 7. Customer 202 exposure
    # ============================================================

    customer202 = customer_exposure[
        customer_exposure["Customer ID"] == 202
    ].iloc[0]

    assert customer202["Purchase_Transaction_Count"] == 2
    assert customer202["Purchase_Order_Count"] == 2
    assert customer202["Total_Purchase_Value"] == pytest.approx(17)

    assert customer202["Mean_Below_Median_Price_Deviation"] == pytest.approx(
        0.0625
    )

    assert customer202["Median_Below_Median_Price_Deviation"] == pytest.approx(
        0.0625
    )

    assert customer202["Transactions_Below_Median"] == 1
    assert customer202["Transactions_Below_10pct"] == 1
    assert customer202["Transactions_Below_20pct"] == 0
    assert customer202["Transactions_Below_30pct"] == 0

    assert customer202["Below_Median_Transaction_Share"] == pytest.approx(0.5)
    assert customer202["Below_10pct_Transaction_Share"] == pytest.approx(0.5)
    assert customer202["Below_20pct_Transaction_Share"] == pytest.approx(0)
    assert customer202["Below_30pct_Transaction_Share"] == pytest.approx(0)

    assert customer202["Below_20pct_Value_Share"] == pytest.approx(0)
    assert customer202["Below_30pct_Value_Share"] == pytest.approx(0)

    # ============================================================
    # 8. Customer 404 exposure
    # ============================================================

    customer404 = customer_exposure[
        customer_exposure["Customer ID"] == 404
    ].iloc[0]

    assert customer404["Purchase_Transaction_Count"] == 1
    assert customer404["Purchase_Order_Count"] == 1
    assert customer404["Total_Purchase_Value"] == pytest.approx(20)

    assert customer404["Transactions_Below_Median"] == 0
    assert customer404["Mean_Below_Median_Price_Deviation"] == pytest.approx(0)
    assert customer404["Median_Below_Median_Price_Deviation"] == pytest.approx(0)

    assert customer404["Below_20pct_Value_Share"] == pytest.approx(0)
    assert customer404["Below_30pct_Value_Share"] == pytest.approx(0)

    assert 303 not in customer_exposure["Customer ID"].tolist()
    assert 0 not in customer_exposure["Customer ID"].tolist()

    # ============================================================
    # 9. Validation checks
    # ============================================================

    expected_validation_keys = [
        "qualifying_transactions_present",
        "customer_ids_present",
        "positive_prices_only",
        "product_median_prices_present",
        "product_median_prices_positive",
        "price_deviation_present",
        "below_median_deviation_non_negative",
        "customer_exposure_present",
        "all_transactions_within_cutoff",
        "passed",
    ]

    for key in expected_validation_keys:
        assert key in validation

    assert all(
        bool(validation[key])
        for key in expected_validation_keys
    )

    assert result["failed_checks"] == []

    # ============================================================
    # 10. Automatic cutoff calculation
    # ============================================================

    auto_df = pd.DataFrame(
        [
            {
                "Invoice": "AUTO001",
                "StockCode": "P003",
                "InvoiceDate": pd.Timestamp("2025-10-01"),
                "Price": 6.0,
                "Quantity": 1,
                "Customer ID": 505,
                "Transaction Type": "Purchase",
                "Include In Sales Analysis": True,
                "Analytical Transaction Value": 6.0,
            },
            {
                "Invoice": "FUTURE001",
                "StockCode": "P004",
                "InvoiceDate": pd.Timestamp("2026-01-21"),
                "Price": 9.0,
                "Quantity": 1,
                "Customer ID": 606,
                "Transaction Type": "Purchase",
                "Include In Sales Analysis": True,
                "Analytical Transaction Value": 9.0,
            },
        ]
    )

    auto_result = run_analysis(auto_df)

    assert auto_result["summary"]["max_invoice_date"] == pd.Timestamp(
        "2026-01-21"
    )

    assert auto_result["summary"]["cutoff_date"] == pd.Timestamp(
        "2025-10-23"
    )

    assert auto_result["summary"]["qualifying_transaction_count"] == 1
    assert auto_result["summary"]["customer_count"] == 1
    assert auto_result["summary"]["product_count"] == 1

    assert auto_result["transaction_price_analysis"]["StockCode_Key"].tolist() == [
        "P003"
    ]

    # ============================================================
    # 11. Invalid cutoff date
    # ============================================================

    with pytest.raises(
        ValueError,
        match="cutoff_date must be earlier than the maximum InvoiceDate",
    ):
        run_analysis(df, pd.Timestamp("2026-01-21"))

    # ============================================================
    # 12. No qualifying purchase transactions
    # ============================================================

    no_purchase_df = df.copy(deep=True)
    no_purchase_df["Transaction Type"] = "Return"

    with pytest.raises(
        ValueError,
        match="No qualifying purchase transactions were found",
    ):
        run_analysis(no_purchase_df, cutoff)

    # ============================================================
    # 13. No qualifying product StockCodes
    # ============================================================

    administrative_df = df.copy(deep=True)
    administrative_df["StockCode"] = "D"

    with pytest.raises(
        ValueError,
        match="No qualifying product transactions remain",
    ):
        run_analysis(administrative_df, cutoff)

    # ============================================================
    # 14. Missing required column
    # ============================================================

    missing_column_df = df.drop(
        columns=["Transaction Type"]
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        run_analysis(missing_column_df, cutoff)

    # ============================================================
    # 15. Invalid invoice dates
    # ============================================================

    invalid_dates_df = df.copy(deep=True)
    invalid_dates_df["InvoiceDate"] = "invalid-date"

    with pytest.raises(
        ValueError,
        match="InvoiceDate contains no valid dates",
    ):
        run_analysis(invalid_dates_df, cutoff)

def test_analyze_discount_sensitivity_behavior(sample_discount_sensitivity_transactions):
    import warnings
    import pandas as pd
    import pytest

    from src.features.features import (
        analyze_customer_price_deviation,
        analyze_discount_sensitivity_behavior,
    )

    # ============================================================
    # Helper: Run upstream price deviation analysis
    # ============================================================

    def run_price_analysis(data, cutoff=None):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Could not infer format.*",
                category=UserWarning,
            )
            return analyze_customer_price_deviation(
                df_cleaned=data,
                cutoff_date=cutoff,
            )

    # ============================================================
    # 1. Prepare fixture
    # ============================================================

    df = sample_discount_sensitivity_transactions.copy(deep=True)

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce",
    )

    df["Transaction Type"] = "Purchase"
    df["Include In Sales Analysis"] = True

    if "Analytical Transaction Value" not in df.columns:
        df["Analytical Transaction Value"] = (
            df["Price"] * df["Quantity"]
        )

    cutoff = pd.Timestamp("2026-01-20")

    # ============================================================
    # 2. Generate input analysis package
    # ============================================================

    discount_price_analysis = run_price_analysis(
        df,
        cutoff,
    )

    assert isinstance(discount_price_analysis, dict)

    assert "customer_price_exposure" in discount_price_analysis
    assert "transaction_price_analysis" in discount_price_analysis

    original_customer_exposure = (
        discount_price_analysis["customer_price_exposure"].copy(deep=True)
    )

    original_transaction_analysis = (
        discount_price_analysis["transaction_price_analysis"].copy(deep=True)
    )

    # ============================================================
    # 3. Execute behavioral analysis
    # ============================================================

    result = analyze_discount_sensitivity_behavior(
        discount_price_analysis=discount_price_analysis,
        df_cleaned=df,
    )

    assert isinstance(result, dict)

    expected_result_keys = {
        "summary",
        "validation",
        "failed_checks",
        "customer_behavior",
        "behavioral_band_summary",
        "threshold_summary",
        "repeated_behavior_summary",
        "transaction_value_relationship",
        "high_exposure_customers",
        "repeated_low_price_customers",
        "high_value_exposure_customers",
        "customers_without_price_analysis",
    }

    assert expected_result_keys.issubset(result.keys())

    summary = result["summary"]
    validation = result["validation"]
    customer_behavior = result["customer_behavior"]
    band_summary = result["behavioral_band_summary"]
    threshold_summary = result["threshold_summary"]
    repeated_summary = result["repeated_behavior_summary"]
    relationship = result["transaction_value_relationship"]

    high_exposure = result["high_exposure_customers"]
    repeated_customers = result["repeated_low_price_customers"]
    high_value_customers = result["high_value_exposure_customers"]

    # ============================================================
    # 4. Verify output types
    # ============================================================

    assert isinstance(summary, dict)
    assert isinstance(validation, dict)
    assert isinstance(customer_behavior, pd.DataFrame)
    assert isinstance(band_summary, pd.DataFrame)
    assert isinstance(threshold_summary, pd.DataFrame)
    assert isinstance(repeated_summary, dict)
    assert isinstance(relationship, dict)

    assert isinstance(high_exposure, pd.DataFrame)
    assert isinstance(repeated_customers, pd.DataFrame)
    assert isinstance(high_value_customers, pd.DataFrame)

    assert isinstance(
        result["customers_without_price_analysis"],
        set,
    )

    # ============================================================
    # 5. Customer population assertions
    # ============================================================

    assert len(customer_behavior) == 3

    assert set(customer_behavior["Customer ID"]) == {
        101,
        202,
        404,
    }

    assert customer_behavior["Customer ID"].is_unique

    assert summary["full_customer_population"] == 4
    assert summary["customers_with_price_analysis"] == 3
    assert summary["customers_without_price_analysis"] == 1

    assert result["customers_without_price_analysis"] == {303}

    # Customer 303 has only administrative StockCode D.
    # Customer ID zero is excluded from the full population.

    assert 303 not in customer_behavior["Customer ID"].tolist()
    assert 0 not in customer_behavior["Customer ID"].tolist()

    # ============================================================
    # 6. Behavioral band creation
    # ============================================================

    expected_band_columns = [
        "Below_20pct_Behavior_Band",
        "Below_30pct_Behavior_Band",
    ]

    for column in expected_band_columns:
        assert column in customer_behavior.columns
        assert customer_behavior[column].notna().all()

    customer101 = customer_behavior[
        customer_behavior["Customer ID"] == 101
    ].iloc[0]

    customer202 = customer_behavior[
        customer_behavior["Customer ID"] == 202
    ].iloc[0]

    customer404 = customer_behavior[
        customer_behavior["Customer ID"] == 404
    ].iloc[0]

    assert str(customer101["Below_20pct_Behavior_Band"]) == (
        "Moderate Exposure (10-25%)"
    )

    assert str(customer101["Below_30pct_Behavior_Band"]) == (
        "Moderate Exposure (10-25%)"
    )

    assert str(customer202["Below_20pct_Behavior_Band"]) == "No Exposure"
    assert str(customer404["Below_20pct_Behavior_Band"]) == "No Exposure"

    # ============================================================
    # 7. Behavioral band summary
    # ============================================================

    assert "Behavior_Band" in band_summary.columns
    assert "Customer_Count" in band_summary.columns
    assert "Customer_Percentage" in band_summary.columns

    band_counts = dict(
        zip(
            band_summary["Behavior_Band"].astype(str),
            band_summary["Customer_Count"],
        )
    )

    assert band_counts["No Exposure"] == 2
    assert band_counts["Moderate Exposure (10-25%)"] == 1

    assert band_counts["Low Exposure (0-10%)"] == 0
    assert band_counts["High Exposure (25-50%)"] == 0
    assert band_counts["Very High Exposure (>50%)"] == 0

    assert band_summary["Customer_Count"].sum() == 3

    assert band_summary["Customer_Percentage"].sum() == pytest.approx(100)

    # ============================================================
    # 8. Threshold exposure summary
    # ============================================================

    assert len(threshold_summary) == 4

    expected_thresholds = [
        "Below Median",
        "At Least 10% Below Median",
        "At Least 20% Below Median",
        "At Least 30% Below Median",
    ]

    assert threshold_summary["Threshold"].tolist() == expected_thresholds

    assert threshold_summary[
        "Customers_With_Exposure"
    ].tolist() == [2, 2, 1, 1]

    assert threshold_summary[
        "Customer_Percentage"
    ].tolist() == pytest.approx([
        200 / 3,
        200 / 3,
        100 / 3,
        100 / 3,
    ])

    # ============================================================
    # 9. Repeated low-price behavior
    # ============================================================

    assert "Repeated_20pct_Below_Median" in customer_behavior.columns
    assert "Repeated_30pct_Below_Median" in customer_behavior.columns

    assert not customer_behavior[
        "Repeated_20pct_Below_Median"
    ].any()

    assert not customer_behavior[
        "Repeated_30pct_Below_Median"
    ].any()

    assert repeated_summary[
        "customers_with_2plus_20pct_below_transactions"
    ] == 0

    assert repeated_summary[
        "customers_with_2plus_30pct_below_transactions"
    ] == 0

    assert repeated_summary[
        "customers_with_5plus_20pct_below_transactions"
    ] == 0

    assert repeated_summary[
        "customers_with_5plus_30pct_below_transactions"
    ] == 0

    assert repeated_customers.empty

    # ============================================================
    # 10. Transaction-share versus value-share relationship
    # ============================================================

    assert (
        "below_20pct_transaction_share_vs_value_share"
        in relationship
    )

    assert (
        "below_30pct_transaction_share_vs_value_share"
        in relationship
    )

    assert relationship[
        "below_20pct_transaction_share_vs_value_share"
    ] == pytest.approx(1.0)

    assert relationship[
        "below_30pct_transaction_share_vs_value_share"
    ] == pytest.approx(1.0)

    # ============================================================
    # 11. High-exposure customers
    # ============================================================

    assert len(high_exposure) == 1

    assert high_exposure["Customer ID"].tolist() == [101]

    assert (
        high_exposure.iloc[0]["Below_20pct_Transaction_Share"]
        == pytest.approx(0.25)
    )

    # ============================================================
    # 12. High-value-exposure customers
    # ============================================================

    assert len(high_value_customers) == 1

    assert high_value_customers["Customer ID"].tolist() == [101]

    assert (
        high_value_customers.iloc[0]["Below_20pct_Value_Share"]
        == pytest.approx(5 / 43)
    )

    # ============================================================
    # 13. Transaction-value exposure differences
    # ============================================================

    expected_difference_columns = [
        "Transaction_Value_Exposure_Difference_20pct",
        "Transaction_Value_Exposure_Difference_30pct",
    ]

    for column in expected_difference_columns:
        assert column in customer_behavior.columns
        assert customer_behavior[column].notna().all()

    assert customer101[
        "Transaction_Value_Exposure_Difference_20pct"
    ] == pytest.approx(0.25 - 5 / 43)

    assert customer101[
        "Transaction_Value_Exposure_Difference_30pct"
    ] == pytest.approx(0.25 - 5 / 43)

    # ============================================================
    # 14. Summary statistics
    # ============================================================

    assert summary["full_customer_population"] == 4
    assert summary["customers_with_price_analysis"] == 3
    assert summary["customers_without_price_analysis"] == 1

    assert summary["below_20pct_share_mean"] == pytest.approx(1 / 12)
    assert summary["below_20pct_share_median"] == pytest.approx(0)
    assert summary["below_20pct_share_max"] == pytest.approx(0.25)

    assert summary["below_30pct_share_mean"] == pytest.approx(1 / 12)
    assert summary["below_30pct_share_median"] == pytest.approx(0)
    assert summary["below_30pct_share_max"] == pytest.approx(0.25)

    assert summary["customers_with_any_20pct_exposure"] == 1
    assert summary["customers_with_any_30pct_exposure"] == 1

    assert summary["customers_with_repeated_20pct_exposure"] == 0
    assert summary["customers_with_repeated_30pct_exposure"] == 0

    # ============================================================
    # 15. Validation checks
    # ============================================================

    expected_validation_keys = [
        "customer_exposure_not_empty",
        "customer_ids_unique",
        "customer_ids_not_missing",
        "purchase_counts_positive",
        "transaction_shares_valid",
        "value_shares_valid",
        "threshold_counts_consistent",
        "full_population_identified",
        "behavioral_bands_created",
        "passed",
    ]

    for key in expected_validation_keys:
        assert key in validation

    assert all(
        bool(validation[key])
        for key in expected_validation_keys
    )

    assert validation["passed"] is True
    assert result["failed_checks"] == []

    # ============================================================
    # 16. Verify input DataFrames are not modified
    # ============================================================

    pd.testing.assert_frame_equal(
        discount_price_analysis["customer_price_exposure"],
        original_customer_exposure,
    )

    pd.testing.assert_frame_equal(
        discount_price_analysis["transaction_price_analysis"],
        original_transaction_analysis,
    )

    # ============================================================
    # 17. Missing required analysis keys
    # ============================================================

    with pytest.raises(
        ValueError,
        match="discount_price_analysis is missing required keys",
    ):
        analyze_discount_sensitivity_behavior(
            discount_price_analysis={},
            df_cleaned=df,
        )

    # ============================================================
    # 18. Missing required customer exposure column
    # ============================================================

    missing_customer_column_analysis = {
        key: value.copy() if isinstance(value, pd.DataFrame) else value
        for key, value in discount_price_analysis.items()
    }

    missing_customer_column_analysis[
        "customer_price_exposure"
    ] = (
        missing_customer_column_analysis[
            "customer_price_exposure"
        ].drop(
            columns=["Below_20pct_Value_Share"]
        )
    )

    with pytest.raises(
        ValueError,
        match="Customer exposure analysis is missing required columns",
    ):
        analyze_discount_sensitivity_behavior(
            discount_price_analysis=missing_customer_column_analysis,
            df_cleaned=df,
        )

    # ============================================================
    # 19. Missing Customer ID in cleaned transaction data
    # ============================================================

    missing_customer_id_df = df.drop(
        columns=["Customer ID"]
    )

    with pytest.raises(
        ValueError,
        match="Missing required column: Customer ID",
    ):
        analyze_discount_sensitivity_behavior(
            discount_price_analysis=discount_price_analysis,
            df_cleaned=missing_customer_id_df,
        )






#================================================================================================================
#----------------------------- 10 .testing engagement_rmf_features ----------------------------------------------
#================================================================================================================   

from src.features.features import create_engagement_score_features
from src.features.features import validate_engagement_score_features
@pytest.fixture
def sample_engagement_rfm_features():
    return pd.DataFrame(
        {
            "Customer ID": [101, 102, 103, 104, 105],
            "Recency": [10, 20, 30, 40, 50],
            "Frequency": [50, 40, 30, 20, 10],
            "Total_Monetary_Value": [
                500.0, 400.0, 300.0, 200.0, 100.0
            ],
        }
    )

def test_create_engagement_score_features(
    sample_engagement_rfm_features,
):
    df = sample_engagement_rfm_features
    original_df = df.copy(deep=True)

    result = create_engagement_score_features(df)

    # 1. Output structure
    assert list(result.columns) == [
        "Customer ID",
        "Engagement_Score",
        "Engagement_Band",
    ]
    assert len(result) == 5
    assert result["Customer ID"].is_unique
    assert result["Customer ID"].notna().all()

    # 2. Independently verify percentile-based calculations
    scores = result.set_index("Customer ID")["Engagement_Score"]

    expected_scores = {
        101: 280 / 3,
        102: 220 / 3,
        103: 160 / 3,
        104: 100 / 3,
        105: 40 / 3,
    }

    for customer_id, expected_score in expected_scores.items():
        assert scores.loc[customer_id] == pytest.approx(
            expected_score
        )

    # 3. Scores must be finite and within [0, 100]
    values = result["Engagement_Score"]
    assert values.notna().all()
    assert np.isfinite(values.to_numpy()).all()
    assert values.ge(0).all()
    assert values.le(100).all()

    # 4. Verify engagement bands
    bands = result.set_index("Customer ID")["Engagement_Band"]

    expected_bands = {
        101: "Very High",
        102: "High",
        103: "Moderate",
        104: "Low",
        105: "Very Low",
    }

    for customer_id, expected_band in expected_bands.items():
        assert bands.loc[customer_id] == expected_band

    # 5. Check output data types
    assert pd.api.types.is_numeric_dtype(
        result["Engagement_Score"]
    )

    # 6. Input data must remain unchanged
    pd.testing.assert_frame_equal(df, original_df)

    # 7. Missing required column must raise ValueError
    invalid_df = df.drop(columns=["Frequency"])

    with pytest.raises(
        ValueError,
        match="rfm_features is missing required columns",
    ):
        create_engagement_score_features(invalid_df)

    # 8. Empty input must raise ValueError
    with pytest.raises(
        ValueError,
        match="rfm_features is empty",
    ):
        create_engagement_score_features(df.iloc[0:0].copy())

    # 9. Missing Customer ID must raise ValueError
    missing_customer_df = df.copy()
    missing_customer_df.loc[0, "Customer ID"] = np.nan

    with pytest.raises(
        ValueError,
        match="missing Customer ID values",
    ):
        create_engagement_score_features(missing_customer_df)

    # 10. Duplicate Customer IDs must raise ValueError
    duplicate_customer_df = df.copy()
    duplicate_customer_df.loc[1, "Customer ID"] = 101

    with pytest.raises(
        ValueError,
        match="duplicate Customer IDs",
    ):
        create_engagement_score_features(duplicate_customer_df)

    # 11. Missing RFM values must raise ValueError
    missing_rfm_df = df.copy()
    missing_rfm_df.loc[0, "Frequency"] = np.nan

    with pytest.raises(
        ValueError,
        match="RFM features contain missing values",
    ):
        create_engagement_score_features(missing_rfm_df)

    # 12. Negative Recency is not explicitly prohibited
    negative_recency_df = df.copy()
    negative_recency_df.loc[0, "Recency"] = -1

    negative_recency_result = create_engagement_score_features(
        negative_recency_df
    )

    assert negative_recency_result[
        "Engagement_Score"
    ].notna().all()

def test_validate_engagement_score_features(sample_rfm_features):
    """Test engagement score validation, summaries, and failure scenarios."""
    import numpy as np
    import pandas as pd
    import pytest

    # 1. Prepare RFM data and generate expected engagement features.
    rfm_df = sample_rfm_features.copy(deep=True)

    # The existing fixture does not contain Recency.
    if "Recency" not in rfm_df.columns:
        rfm_df["Recency"] = [10, 30] if len(rfm_df) == 2 else range(
            1, len(rfm_df) + 1
        )

    expected_features = create_engagement_score_features(rfm_df)

    # 2. Validate correctly generated engagement features.
    result = validate_engagement_score_features(
        expected_features,
        rfm_df,
    )

    expected_true_checks = [
        "customer_id_column_present",
        "engagement_score_column_present",
        "engagement_band_column_present",
        "rfm_output_not_empty",
        "rfm_customer_ids_unique",
        "rfm_customer_ids_not_missing",
        "rfm_values_not_missing",
        "engagement_output_not_empty",
        "customer_ids_unique",
        "no_missing_customer_id",
        "no_missing_engagement_score",
        "engagement_score_is_numeric",
        "customer_count_matches",
        "customer_population_matches",
        "recency_values_non_negative",
        "frequency_values_positive",
        "monetary_values_positive",
        "recency_scores_valid",
        "frequency_scores_valid",
        "monetary_scores_valid",
        "expected_engagement_score_valid",
        "actual_engagement_score_valid",
        "customer_ids_aligned",
        "engagement_scores_match",
        "engagement_bands_match",
        "engagement_band_categories_valid",
        "passed",
    ]

    for check in expected_true_checks:
        assert bool(result["validation"][check]) is True, (
            f"Expected '{check}' to pass. "
            f"Failed checks: {result['failed_checks']}"
        )

    assert result["failed_checks"] == []

    # 3. Validate summary statistics.
    scores = expected_features["Engagement_Score"]
    summary = result["summary"]

    assert summary["customer_count"] == len(expected_features)
    assert summary["expected_customer_count"] == len(rfm_df)

    assert summary["minimum_engagement_score"] == pytest.approx(
        scores.min()
    )
    assert summary["median_engagement_score"] == pytest.approx(
        scores.median()
    )
    assert summary["mean_engagement_score"] == pytest.approx(
        scores.mean()
    )
    assert summary["maximum_engagement_score"] == pytest.approx(
        scores.max()
    )
    assert summary["maximum_absolute_difference"] == pytest.approx(
        0.0, abs=1e-12
    )

    # 4. Validate engagement band summary.
    band_summary = result["engagement_band_summary"]

    assert set(band_summary.columns) == {
        "Engagement_Band",
        "Customer_Count",
        "Customer_Percentage",
    }
    assert band_summary["Customer_Count"].sum() == len(expected_features)
    assert band_summary["Customer_Percentage"].sum() == pytest.approx(
        100.0
    )

    band_counts = {
        "very_low_customer_count": "Very Low",
        "low_customer_count": "Low",
        "moderate_customer_count": "Moderate",
        "high_customer_count": "High",
        "very_high_customer_count": "Very High",
    }

    for summary_key, band_name in band_counts.items():
        expected_count = (
            expected_features["Engagement_Band"].astype(str) == band_name
        ).sum()
        assert summary[summary_key] == expected_count

    # 5. Missing required RFM source column must raise ValueError.
    with pytest.raises(
        ValueError,
        match="rfm_features is missing required columns",
    ):
        validate_engagement_score_features(
            expected_features,
            rfm_df.drop(columns=["Recency"]),
        )

    # 6. Missing Engagement_Score output column.
    missing_score_column = expected_features.drop(
        columns=["Engagement_Score"]
    )

    result = validate_engagement_score_features(
        missing_score_column,
        rfm_df,
    )

    assert bool(
        result["validation"]["engagement_score_column_present"]
    ) is False
    assert bool(result["validation"]["passed"]) is False
    assert result["summary"] == {}
    assert any(
        "Missing output columns" in item
        for item in result["failed_checks"]
    )

    # 7. Missing Engagement_Band output column.
    missing_band_column = expected_features.drop(
        columns=["Engagement_Band"]
    )

    result = validate_engagement_score_features(
        missing_band_column,
        rfm_df,
    )

    assert bool(
        result["validation"]["engagement_band_column_present"]
    ) is False
    assert bool(result["validation"]["passed"]) is False
    assert result["summary"] == {}

    # 8. Duplicate customer IDs in engagement features.
    duplicate_customer_df = pd.concat(
        [expected_features, expected_features.iloc[[0]]],
        ignore_index=True,
    )

    result = validate_engagement_score_features(
        duplicate_customer_df,
        rfm_df,
    )

    assert bool(result["validation"]["customer_ids_unique"]) is False
    assert bool(result["validation"]["customer_count_matches"]) is False
    assert bool(result["validation"]["customer_ids_aligned"]) is False
    assert bool(result["validation"]["passed"]) is False

    # 9. Missing engagement score.
    missing_score_df = expected_features.copy(deep=True)
    missing_score_df.loc[
        missing_score_df.index[0], "Engagement_Score"
    ] = np.nan

    result = validate_engagement_score_features(
        missing_score_df,
        rfm_df,
    )

    assert bool(
        result["validation"]["no_missing_engagement_score"]
    ) is False
    assert bool(
        result["validation"]["engagement_scores_match"]
    ) is False
    assert bool(result["validation"]["passed"]) is False

    # 10. Engagement score below the valid range.
    negative_score_df = expected_features.copy(deep=True)
    negative_score_df.loc[
        negative_score_df.index[0], "Engagement_Score"
    ] = -1.0

    result = validate_engagement_score_features(
        negative_score_df,
        rfm_df,
    )

    assert bool(
        result["validation"]["actual_engagement_score_valid"]
    ) is False
    assert bool(
        result["validation"]["engagement_scores_match"]
    ) is False
    assert bool(result["validation"]["passed"]) is False

    # 11. Engagement score above the valid range.
    excessive_score_df = expected_features.copy(deep=True)
    excessive_score_df.loc[
        excessive_score_df.index[0], "Engagement_Score"
    ] = 101.0

    result = validate_engagement_score_features(
        excessive_score_df,
        rfm_df,
    )

    assert bool(
        result["validation"]["actual_engagement_score_valid"]
    ) is False
    assert bool(
        result["validation"]["engagement_scores_match"]
    ) is False
    assert bool(result["validation"]["passed"]) is False

    # 12. Incorrect engagement score within the valid range.
    incorrect_score_df = expected_features.copy(deep=True)
    first_index = incorrect_score_df.index[0]
    original_score = float(
        incorrect_score_df.loc[first_index, "Engagement_Score"]
    )

    changed_score = (
        original_score + 1.0
        if original_score < 100.0
        else original_score - 1.0
    )

    incorrect_score_df.loc[
        first_index, "Engagement_Score"
    ] = changed_score

    assert not np.isclose(changed_score, original_score)

    result = validate_engagement_score_features(
        incorrect_score_df,
        rfm_df,
    )

    assert bool(
        result["validation"]["engagement_scores_match"]
    ) is False
    assert bool(result["validation"]["passed"]) is False
    assert "engagement_scores_match" in result["failed_checks"]

    # 13. Invalid engagement band.
    # Convert the categorical column to object before assigning a new label.
    incorrect_band_df = expected_features.copy(deep=True)
    incorrect_band_df["Engagement_Band"] = (
        incorrect_band_df["Engagement_Band"].astype(object)
    )
    incorrect_band_df.loc[
        incorrect_band_df.index[0], "Engagement_Band"
    ] = "Invalid Band"

    result = validate_engagement_score_features(
        incorrect_band_df,
        rfm_df,
    )

    assert bool(
        result["validation"]["engagement_band_categories_valid"]
    ) is False
    assert bool(
        result["validation"]["engagement_bands_match"]
    ) is False
    assert bool(result["validation"]["passed"]) is False

    # 14. Missing customer from the engagement population.
    missing_customer_df = expected_features.iloc[:-1].copy(deep=True)

    result = validate_engagement_score_features(
        missing_customer_df,
        rfm_df,
    )

    assert bool(result["validation"]["customer_count_matches"]) is False
    assert bool(
        result["validation"]["customer_population_matches"]
    ) is False
    assert bool(result["validation"]["customer_ids_aligned"]) is False
    assert bool(result["validation"]["passed"]) is False

    # 15. Invalid RFM values: negative Recency.
    invalid_rfm_df = rfm_df.copy(deep=True)
    invalid_rfm_df.loc[invalid_rfm_df.index[0], "Recency"] = -1

    result = validate_engagement_score_features(
        expected_features,
        invalid_rfm_df,
    )

    assert bool(
        result["validation"]["recency_values_non_negative"]
    ) is False
    assert bool(result["validation"]["passed"]) is False

    # 16. Invalid RFM values: zero Frequency.
    invalid_frequency_df = rfm_df.copy(deep=True)
    invalid_frequency_df.loc[
        invalid_frequency_df.index[0], "Frequency"
    ] = 0

    result = validate_engagement_score_features(
        expected_features,
        invalid_frequency_df,
    )

    assert bool(
        result["validation"]["frequency_values_positive"]
    ) is False
    assert bool(result["validation"]["passed"]) is False

    # 17. Invalid RFM values: zero monetary value.
    invalid_monetary_df = rfm_df.copy(deep=True)
    invalid_monetary_df.loc[
        invalid_monetary_df.index[0], "Total_Monetary_Value"
    ] = 0

    result = validate_engagement_score_features(
        expected_features,
        invalid_monetary_df,
    )

    assert bool(
        result["validation"]["monetary_values_positive"]
    ) is False
    assert bool(result["validation"]["passed"]) is False

    # 18. Missing values in RFM inputs.
    missing_rfm_value_df = rfm_df.copy(deep=True)
    missing_rfm_value_df.loc[
        missing_rfm_value_df.index[0], "Recency"
    ] = np.nan

    result = validate_engagement_score_features(
        expected_features,
        missing_rfm_value_df,
    )

    assert bool(
        result["validation"]["rfm_values_not_missing"]
    ) is False
    assert bool(result["validation"]["passed"]) is False


def test_analyze_engagement_score(sample_discount_sensitivity_transactions):
    import warnings

    import numpy as np
    import pandas as pd
    import pytest

    from src.features.features import analyze_engagement_score

    # ---------------------------------------------------------
    # Helper: compare floating-point values without pytest.approx
    # ---------------------------------------------------------

    def assert_numeric_equal(actual, expected):
        assert round(float(actual), 10) == round(float(expected), 10)

    # ---------------------------------------------------------
    # 1. Prepare a copy of the existing fixture
    # ---------------------------------------------------------

    df = sample_discount_sensitivity_transactions.copy(deep=True)

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce"
    )

    df["Transaction Type"] = "Purchase"
    df["Include In Sales Analysis"] = True

    df["Analytical Transaction Value"] = (
        pd.to_numeric(df["Price"], errors="coerce")
        * pd.to_numeric(df["Quantity"], errors="coerce")
    )

    cutoff_date = pd.Timestamp("2026-01-20")

    # ---------------------------------------------------------
    # 2. Execute engagement analysis
    # ---------------------------------------------------------

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)

        result = analyze_engagement_score(
            df_cleaned=df,
            cutoff_date=cutoff_date
        )

    # ---------------------------------------------------------
    # 3. Validate returned structure
    # ---------------------------------------------------------

    expected_keys = {
        "summary",
        "validation",
        "failed_checks",
        "engagement_base",
        "rfm_distribution",
        "score_distribution",
        "engagement_band_summary",
        "component_correlation",
        "component_summary",
        "highest_engagement_customers",
        "lowest_engagement_customers",
    }

    assert isinstance(result, dict)
    assert expected_keys.issubset(result.keys())

    summary = result["summary"]
    validation = result["validation"]
    engagement = result["engagement_base"]

    # ---------------------------------------------------------
    # 4. Validate qualifying transactions and customers
    # ---------------------------------------------------------

    assert summary["qualifying_transaction_count"] == 9
    assert summary["customer_count"] == 4

    assert set(engagement["Customer ID"].tolist()) == {
        101, 202, 303, 404
    }

    assert engagement["Customer ID"].is_unique
    assert engagement["Customer ID"].notna().all()

    # ---------------------------------------------------------
    # 5. Validate point-in-time cutoff and dates
    # ---------------------------------------------------------

    assert isinstance(summary["max_invoice_date"], pd.Timestamp)
    assert isinstance(summary["cutoff_date"], pd.Timestamp)

    assert summary["max_invoice_date"] == pd.Timestamp("2026-01-21")
    assert summary["cutoff_date"] == cutoff_date

    assert summary["cutoff_date"] < summary["max_invoice_date"]

    assert (engagement["Last_Purchase_Date"] <= cutoff_date).all()
    assert engagement["Last_Purchase_Date"].notna().all()

    # ---------------------------------------------------------
    # 6. Validate customer-level RFM calculations
    # ---------------------------------------------------------

    expected_rfm = {
        101: {
            "Recency": 5.0,
            "Frequency": 4,
            "Monetary": 43.0,
        },
        202: {
            "Recency": 12.0,
            "Frequency": 2,
            "Monetary": 17.0,
        },
        303: {
            "Recency": 8.0,
            "Frequency": 2,
            "Monetary": 8.0,
        },
        404: {
            "Recency": 0.0,
            "Frequency": 1,
            "Monetary": 20.0,
        },
    }

    for customer_id, expected in expected_rfm.items():

        customer = engagement.loc[
            engagement["Customer ID"] == customer_id
        ].iloc[0]

        assert_numeric_equal(
            customer["Recency"],
            expected["Recency"]
        )

        assert customer["Frequency"] == expected["Frequency"]

        assert_numeric_equal(
            customer["Monetary"],
            expected["Monetary"]
        )

    # ---------------------------------------------------------
    # 7. Validate percentile-based component scores
    # ---------------------------------------------------------

    expected_scores = {
        101: {
            "Recency_Score": 50.0,
            "Frequency_Score": 100.0,
            "Monetary_Score": 100.0,
            "Candidate_Engagement_Score": 250 / 3,
        },
        202: {
            "Recency_Score": 0.0,
            "Frequency_Score": 62.5,
            "Monetary_Score": 50.0,
            "Candidate_Engagement_Score": 37.5,
        },
        303: {
            "Recency_Score": 25.0,
            "Frequency_Score": 62.5,
            "Monetary_Score": 25.0,
            "Candidate_Engagement_Score": 37.5,
        },
        404: {
            "Recency_Score": 75.0,
            "Frequency_Score": 25.0,
            "Monetary_Score": 75.0,
            "Candidate_Engagement_Score": 175 / 3,
        },
    }

    score_columns = [
        "Recency_Score",
        "Frequency_Score",
        "Monetary_Score",
        "Candidate_Engagement_Score",
    ]

    for customer_id, expected in expected_scores.items():

        customer = engagement.loc[
            engagement["Customer ID"] == customer_id
        ].iloc[0]

        for column in score_columns:

            assert_numeric_equal(
                customer[column],
                expected[column]
            )

    # ---------------------------------------------------------
    # 8. Validate engagement bands
    # ---------------------------------------------------------

    expected_bands = {
        101: "Very High",
        202: "Low",
        303: "Low",
        404: "Moderate",
    }

    for customer_id, expected_band in expected_bands.items():

        customer = engagement.loc[
            engagement["Customer ID"] == customer_id
        ].iloc[0]

        assert str(customer["Candidate_Engagement_Band"]) == (
            expected_band
        )

    band_summary = result["engagement_band_summary"]

    expected_band_counts = {
        "Very Low": 0,
        "Low": 2,
        "Moderate": 1,
        "High": 0,
        "Very High": 1,
    }

    actual_band_counts = dict(
        zip(
            band_summary["Engagement_Band"].astype(str),
            band_summary["Customer_Count"]
        )
    )

    assert actual_band_counts == expected_band_counts

    assert_numeric_equal(
        band_summary["Customer_Percentage"].sum(),
        100.0
    )

    # ---------------------------------------------------------
    # 9. Validate summary statistics
    # ---------------------------------------------------------

    expected_summary = {
        "minimum_recency_days": 0.0,
        "median_recency_days": 6.5,
        "maximum_recency_days": 12.0,
        "minimum_frequency": 1,
        "median_frequency": 2.0,
        "maximum_frequency": 4,
        "minimum_monetary": 8.0,
        "median_monetary": 18.5,
        "maximum_monetary": 43.0,
        "candidate_engagement_mean": 325 / 6,
        "candidate_engagement_median": 575 / 12,
        "candidate_engagement_minimum": 37.5,
        "candidate_engagement_maximum": 250 / 3,
    }

    for key, expected in expected_summary.items():

        assert_numeric_equal(
            summary[key],
            expected
        )

    # ---------------------------------------------------------
    # 10. Validate score ranges and numerical data types
    # ---------------------------------------------------------

    for column in score_columns:

        assert pd.api.types.is_numeric_dtype(
            engagement[column]
        )

        assert engagement[column].notna().all()

        assert engagement[column].between(0, 100).all()

    assert np.isfinite(
        engagement[score_columns].to_numpy(dtype=float)
    ).all()

    # ---------------------------------------------------------
    # 11. Validate RFM and score distributions
    # ---------------------------------------------------------

    rfm_distribution = result["rfm_distribution"]
    score_distribution = result["score_distribution"]

    assert set([
        "Recency",
        "Frequency",
        "Monetary",
    ]).issubset(rfm_distribution.columns)

    assert set(score_columns).issubset(
        score_distribution.columns
    )

    assert rfm_distribution.notna().all().all()
    assert score_distribution.notna().all().all()

    # ---------------------------------------------------------
    # 12. Validate component correlation
    # ---------------------------------------------------------

    correlation = result["component_correlation"]

    assert isinstance(correlation, pd.DataFrame)
    assert correlation.shape == (7, 7)
    assert correlation.notna().all().all()
    assert correlation.equals(correlation.T)

    assert np.isfinite(
        correlation.to_numpy(dtype=float)
    ).all()

    # ---------------------------------------------------------
    # 13. Validate component summary
    # ---------------------------------------------------------

    component_summary = result["component_summary"]

    assert component_summary["Component"].tolist() == [
        "Recency",
        "Frequency",
        "Monetary",
    ]

    assert len(component_summary) == 3

    component_score_columns = [
        "Mean_Score",
        "Median_Score",
        "Minimum_Score",
        "Maximum_Score",
    ]

    assert component_summary[
        component_score_columns
    ].notna().all().all()

    # ---------------------------------------------------------
    # 14. Validate highest and lowest engagement customers
    # ---------------------------------------------------------

    highest = result["highest_engagement_customers"]
    lowest = result["lowest_engagement_customers"]

    assert len(highest) == 4
    assert len(lowest) == 4

    assert highest[
        "Candidate_Engagement_Score"
    ].is_monotonic_decreasing

    assert lowest[
        "Candidate_Engagement_Score"
    ].is_monotonic_increasing

    assert highest.iloc[0]["Customer ID"] == 101

    assert lowest.iloc[0]["Customer ID"] == 202

    # ---------------------------------------------------------
    # 15. Validate function validation results
    # ---------------------------------------------------------

    expected_validation_checks = {
        "qualifying_transactions_present",
        "customer_ids_present",
        "customer_ids_unique",
        "recency_values_valid",
        "frequency_values_positive",
        "monetary_values_positive",
        "recency_scores_valid",
        "frequency_scores_valid",
        "monetary_scores_valid",
        "candidate_engagement_scores_valid",
        "no_missing_component_scores",
        "no_missing_candidate_scores",
        "all_transactions_within_cutoff",
        "cutoff_before_max_invoice_date",
        "passed",
    }

    assert expected_validation_checks.issubset(
        validation.keys()
    )

    for check in expected_validation_checks:

        assert bool(validation[check]) is True, (
            f"Validation check failed: {check}"
        )

    assert bool(validation["passed"]) is True
    assert result["failed_checks"] == []

    # ---------------------------------------------------------
    # 16. Test automatic cutoff date calculation
    # ---------------------------------------------------------

    df_auto = df.copy(deep=True)

    anchor_row = df_auto.iloc[[0]].copy()

    anchor_row["Invoice"] = "AUTO_CUTOFF_ANCHOR"

    anchor_row["InvoiceDate"] = (
        df_auto["InvoiceDate"].max() + pd.Timedelta(days=100)
    )

    anchor_row["Customer ID"] = 505

    df_auto = pd.concat(
        [df_auto, anchor_row],
        ignore_index=True
    )

    auto_result = analyze_engagement_score(
        df_cleaned=df_auto,
        cutoff_date=None
    )

    expected_max_date = df_auto["InvoiceDate"].max()

    expected_auto_cutoff = (
        expected_max_date - pd.Timedelta(days=90)
    )

    assert auto_result["summary"]["max_invoice_date"] == (
        expected_max_date
    )

    assert auto_result["summary"]["cutoff_date"] == (
        expected_auto_cutoff
    )

    assert (
        auto_result["summary"]["qualifying_transaction_count"] == 9
    )

    # ---------------------------------------------------------
    # 17. Test missing required columns
    # ---------------------------------------------------------

    df_missing = df.drop(
        columns=["Invoice"]
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns"
    ):
        analyze_engagement_score(
            df_cleaned=df_missing,
            cutoff_date=cutoff_date
        )

    # ---------------------------------------------------------
    # 18. Test dataframe containing no valid dates
    # ---------------------------------------------------------

    df_invalid_dates = df.copy(deep=True)

    df_invalid_dates["InvoiceDate"] = "invalid-date"

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Could not infer format",
            category=UserWarning,
        )

        with pytest.raises(
            ValueError,
            match="InvoiceDate contains no valid dates"
        ):
            analyze_engagement_score(
                df_cleaned=df_invalid_dates,
                cutoff_date=cutoff_date
            )

    # ---------------------------------------------------------
    # 19. Test cutoff date equal to maximum invoice date
    # ---------------------------------------------------------

    with pytest.raises(
        ValueError,
        match="cutoff_date must be earlier than max InvoiceDate"
    ):
        analyze_engagement_score(
            df_cleaned=df,
            cutoff_date=pd.Timestamp("2026-01-21")
        )

    # ---------------------------------------------------------
    # 20. Test cutoff date later than maximum invoice date
    # ---------------------------------------------------------

    with pytest.raises(
        ValueError,
        match="cutoff_date must be earlier than max InvoiceDate"
    ):
        analyze_engagement_score(
            df_cleaned=df,
            cutoff_date=pd.Timestamp("2026-02-01")
        )

    # ---------------------------------------------------------
    # 21. Test no qualifying historical purchases
    # ---------------------------------------------------------

    df_no_purchases = df.copy(deep=True)

    df_no_purchases["Include In Sales Analysis"] = False

    with pytest.raises(
        ValueError,
        match="No qualifying historical purchase transactions found"
    ):
        analyze_engagement_score(
            df_cleaned=df_no_purchases,
            cutoff_date=cutoff_date
        )

    # ---------------------------------------------------------
    # 22. Test transactions after cutoff are excluded
    # ---------------------------------------------------------

    df_after_cutoff = df.copy(deep=True)

    df_after_cutoff["InvoiceDate"] = pd.Timestamp("2026-01-22")

    with pytest.raises(
        ValueError,
        match="No qualifying historical purchase transactions found"
    ):
        analyze_engagement_score(
            df_cleaned=df_after_cutoff,
            cutoff_date=cutoff_date
        )



################################################################################################################################################
################################################################################################################################################################
#-------------------------------------------- TESTING OF DATA CLEANING ------------------------------------------------------------
###############################################################################################################################################
################################################################################################################################################################################################

def test_clean_duplicates():
    """Test exact duplicate removal and preservation of distinct transactions."""
    import pandas as pd
    from pandas.testing import assert_frame_equal

    from src.data.clean import clean_duplicates

    # 1. Exact duplicates should be removed.
    transactions = pd.DataFrame(
        {
            "Invoice": ["INV001", "INV001", "INV002", "INV003"],
            "StockCode": ["A101", "A101", "B202", "C303"],
            "Quantity": [2, 2, 5, 1],
            "Price": [10.0, 10.0, 20.0, 15.0],
        },
        index=[10, 20, 30, 40],
    )

    original_df = transactions.copy(deep=True)
    result = clean_duplicates(transactions)

    expected_df = pd.DataFrame(
        {
            "Invoice": ["INV001", "INV002", "INV003"],
            "StockCode": ["A101", "B202", "C303"],
            "Quantity": [2, 5, 1],
            "Price": [10.0, 20.0, 15.0],
        }
    )

    assert_frame_equal(result, expected_df)

    # 2. The input DataFrame must remain unchanged.
    assert_frame_equal(transactions, original_df)

    # 3. Distinct transactions must be preserved even when their
    # invoice and product are the same.
    repeated_purchases = pd.DataFrame(
        {
            "Invoice": ["INV010", "INV010", "INV010"],
            "StockCode": ["A101", "A101", "A101"],
            "Quantity": [2, 3, 2],
            "Price": [10.0, 10.0, 12.0],
        }
    )

    result = clean_duplicates(repeated_purchases)

    assert len(result) == 3
    assert not result.duplicated().any()

    # 4. DataFrames without duplicates should remain unchanged.
    unique_df = pd.DataFrame(
        {
            "Invoice": ["INV101", "INV102"],
            "StockCode": ["A101", "B202"],
            "Quantity": [1, 4],
            "Price": [5.0, 8.0],
        },
        index=[5, 9],
    )

    result = clean_duplicates(unique_df)

    assert_frame_equal(result, unique_df)

    # 5. Empty DataFrames should be handled.
    empty_df = pd.DataFrame(
        columns=["Invoice", "StockCode", "Quantity", "Price"]
    )

    result = clean_duplicates(empty_df)

    assert_frame_equal(result, empty_df)

    # 6. A single-row DataFrame should remain unchanged.
    single_row_df = pd.DataFrame(
        {
            "Invoice": ["INV201"],
            "StockCode": ["A101"],
            "Quantity": [1],
            "Price": [10.0],
        },
        index=[7],
    )

    result = clean_duplicates(single_row_df)

    assert_frame_equal(result, single_row_df)

    # 7. Duplicate removal should reset the index.
    indexed_df = pd.DataFrame(
        {
            "Invoice": ["INV301", "INV301", "INV302"],
            "Quantity": [1, 1, 2],
        },
        index=[10, 20, 30],
    )

    result = clean_duplicates(indexed_df)

    assert result.index.tolist() == [0, 1]
    assert len(result) == 2

def test_handle_quantity_and_price():
    """Test quantity and price anomaly handling and business rules."""
    import pandas as pd
    import pytest
    from pandas.testing import assert_frame_equal

    from src.data.clean import handle_quantity_and_price

    # 1. Build data covering purchases, returns, cancellations,
    # price adjustments, and extreme quantities.
    df = pd.DataFrame(
        {
            "Invoice": [
                "1001", "1002", "1003", "C1004",
                "1005", "1006", "1007", "1008",
                "1009", "1010", "1011", "1012",
            ],
            "StockCode": [
                "A", "A", "B", "B", "C", "C",
                "D", "E", "F", "G", "H", "I",
            ],
            "Quantity": [
                1, 2, 3, -2, -1, 4,
                5, 6, 7, 8, 15000, 2,
            ],
            "Price": [
                10.0, 12.0, 15.0, 15.0, 10.0, 20.0,
                0.0, -5.0, 25.0, 30.0, 100.0, 12.0,
            ],
            "Customer ID": [
                101, 101, 102, 102, 103, 104,
                105, 106, 107, 108, 109, 110,
            ],
        }
    )
    original_df = df.copy(deep=True)

    result = handle_quantity_and_price(
        df,
        extreme_quantity_threshold=10000,
        extreme_value_quantile=0.75,
    )

    # 2. Check output structure and ensure the input is unchanged.
    required_output_columns = [
        "Quantity IQR Outlier",
        "Price IQR Outlier",
        "Transaction Type",
        "Quantity Treatment",
        "Transaction Value",
        "Price Treatment",
        "Include In Sales Analysis",
        "Analytical Transaction Value",
    ]

    for column in required_output_columns:
        assert column in result.columns

    assert_frame_equal(df, original_df)
    assert len(result) == len(df)

    # 3. Original quantity and price values must be preserved.
    assert result["Quantity"].tolist() == original_df["Quantity"].tolist()
    assert result["Price"].tolist() == original_df["Price"].tolist()

    # 4. Verify transaction type classification.
    assert result.loc[0, "Transaction Type"] == "Purchase"
    assert result.loc[3, "Transaction Type"] == "Cancellation"
    assert result.loc[4, "Transaction Type"] == "Return"

    # 5. Verify returns and cancellations are retained as such.
    assert result.loc[3, "Quantity Treatment"] == "Return/Cancellation"
    assert result.loc[4, "Quantity Treatment"] == "Return/Cancellation"

    # 6. Transaction values must equal Quantity multiplied by Price.
    expected_transaction_values = (
        result["Quantity"] * result["Price"]
    )
    assert result["Transaction Value"].equals(
        expected_transaction_values
    )

    # 7. Zero-price transactions are excluded from sales analysis.
    zero_price_rows = result["Price"] == 0
    assert zero_price_rows.any()

    assert (
        result.loc[zero_price_rows, "Price Treatment"]
        == "Zero Price - Excluded"
    ).all()

    assert (
        ~result.loc[zero_price_rows, "Include In Sales Analysis"]
    ).all()

    assert (
        result.loc[zero_price_rows, "Analytical Transaction Value"] == 0
    ).all()

    # 8. Negative-price transactions are administrative adjustments.
    negative_price_rows = result["Price"] < 0
    assert negative_price_rows.any()

    assert (
        result.loc[negative_price_rows, "Price Treatment"]
        == "Negative Price - Administrative Adjustment"
    ).all()

    assert (
        ~result.loc[negative_price_rows, "Include In Sales Analysis"]
    ).all()

    assert (
        result.loc[negative_price_rows, "Analytical Transaction Value"]
        == 0
    ).all()

    # 9. Positive prices, including price outliers, are retained
    # unless another rule excludes the transaction.
    positive_price_rows = result["Price"] > 0
    assert (
        result.loc[positive_price_rows, "Price Treatment"]
        != "Zero Price - Excluded"
    ).all()

    # 10. Analytical transaction value must be zero for excluded rows
    # and equal transaction value for included rows.
    excluded_rows = ~result["Include In Sales Analysis"]
    included_rows = result["Include In Sales Analysis"]

    assert (
        result.loc[excluded_rows, "Analytical Transaction Value"] == 0
    ).all()

    assert (
        result.loc[included_rows, "Analytical Transaction Value"]
        == result.loc[included_rows, "Transaction Value"]
    ).all()

    # 11. IQR statistics and domain-rule metadata must be recorded.
    assert "quantity_iqr" in result.attrs
    assert "price_iqr" in result.attrs
    assert "domain_rules" in result.attrs

    for key in ["q1", "q3", "iqr", "lower_bound", "upper_bound"]:
        assert key in result.attrs["quantity_iqr"]
        assert key in result.attrs["price_iqr"]

    assert (
        result.attrs["domain_rules"]["extreme_quantity_threshold"]
        == 10000
    )
    assert result.attrs["domain_rules"]["extreme_value_quantile"] == 0.75

    # 12. Missing required columns should raise ValueError.
    with pytest.raises(ValueError, match="Missing required columns"):
        handle_quantity_and_price(df.drop(columns=["Invoice"]))

    with pytest.raises(ValueError, match="Missing required columns"):
        handle_quantity_and_price(df.drop(columns=["StockCode"]))

    with pytest.raises(ValueError, match="Missing required columns"):
        handle_quantity_and_price(df.drop(columns=["Quantity"]))

    with pytest.raises(ValueError, match="Missing required columns"):
        handle_quantity_and_price(df.drop(columns=["Price"]))

    # 13. Numeric strings should be converted to numeric values.
    numeric_string_df = pd.DataFrame(
        {
            "Invoice": ["2001", "2002", "2003", "2004"],
            "StockCode": ["A", "B", "C", "D"],
            "Quantity": ["1", "2", "3", "4"],
            "Price": ["10.0", "20.0", "30.0", "40.0"],
        }
    )

    numeric_result = handle_quantity_and_price(numeric_string_df)

    assert pd.api.types.is_numeric_dtype(numeric_result["Quantity"])
    assert pd.api.types.is_numeric_dtype(numeric_result["Price"])

    # 14. Nonnumeric quantity and price values should be coerced to NaN.
    nonnumeric_df = pd.DataFrame(
        {
            "Invoice": ["3001", "3002", "3003", "3004"],
            "StockCode": ["A", "B", "C", "D"],
            "Quantity": ["1", "bad", "3", "4"],
            "Price": ["10.0", "20.0", "bad", "40.0"],
        }
    )

    nonnumeric_result = handle_quantity_and_price(nonnumeric_df)

    assert pd.isna(nonnumeric_result.loc[1, "Quantity"])
    assert pd.isna(nonnumeric_result.loc[2, "Price"])

    # 15. Extreme-quantity behavior: repeated bulk lines on one invoice
    # are classified as legitimate bulk purchases.
    bulk_invoice_df = pd.DataFrame(
        {
            "Invoice": ["4001", "4001", "4002", "4003", "4004"],
            "StockCode": ["A", "B", "C", "D", "E"],
            "Quantity": [15000, 16000, 1, 2, 3],
            "Price": [1.0, 1.0, 10.0, 20.0, 30.0],
            "Customer ID": [201, 202, 203, 204, 205],
        }
    )

    bulk_result = handle_quantity_and_price(
        bulk_invoice_df,
        extreme_quantity_threshold=10000,
        extreme_value_quantile=0.75,
    )

    assert (
        bulk_result.loc[0, "Quantity Treatment"]
        == "Legitimate Bulk Purchase"
    )
    assert (
        bulk_result.loc[1, "Quantity Treatment"]
        == "Legitimate Bulk Purchase"
    )

    # 16. Repeated extreme quantities by the same customer are
    # classified as legitimate bulk behavior, even on separate invoices.
    repeated_customer_df = pd.DataFrame(
        {
            "Invoice": ["5001", "5002", "5003", "5004", "5005"],
            "StockCode": ["A", "B", "C", "D", "E"],
            "Quantity": [15000, 16000, 1, 2, 3],
            "Price": [1.0, 1.0, 10.0, 20.0, 30.0],
            "Customer ID": [301, 301, 302, 303, 304],
        }
    )

    repeated_result = handle_quantity_and_price(
        repeated_customer_df,
        extreme_quantity_threshold=10000,
        extreme_value_quantile=0.75,
    )

    assert (
        repeated_result.loc[0, "Quantity Treatment"]
        == "Legitimate Bulk Purchase"
    )
    assert (
        repeated_result.loc[1, "Quantity Treatment"]
        == "Legitimate Bulk Purchase"
    )

    # 17. Without Customer ID, the function should still process
    # the DataFrame and use invoice-level bulk detection.
    no_customer_df = bulk_invoice_df.drop(columns=["Customer ID"])

    no_customer_result = handle_quantity_and_price(
        no_customer_df,
        extreme_quantity_threshold=10000,
        extreme_value_quantile=0.75,
    )

    assert len(no_customer_result) == len(no_customer_df)
    assert "Quantity Treatment" in no_customer_result.columns

    # 18. Empty input with the required columns should be processed.
    empty_df = pd.DataFrame(
        columns=["Invoice", "StockCode", "Quantity", "Price"]
    )

    empty_result = handle_quantity_and_price(empty_df)

    assert empty_result.empty
    assert "Quantity Treatment" in empty_result.columns
    assert "Price Treatment" in empty_result.columns

def test_standardize_descriptions():
    """Test product description standardization and canonical lookup."""
    import pandas as pd
    import pytest
    import warnings

    warnings.filterwarnings(
    "ignore",
    message=r".*This pattern.*has match groups.*",
    category=UserWarning,)
    from pandas.testing import assert_frame_equal

    from src.data.clean import standardize_descriptions

    # 1. Prepare data with inconsistent descriptions, missing values,
    # whitespace, and data-quality notes.
    df = pd.DataFrame(
        {
            "StockCode": [
                "A101", "A101", "A101",
                "B202", "B202",
                "C303", "C303",
                "D404", "E505",
            ],
            "Description": [
                "  red   cotton shirt ",
                "RED COTTON SHIRT",
                "wrongly marked item",
                "blue ceramic mug",
                " BLUE   CERAMIC MUG ",
                None,
                "green glass bottle",
                "FOUND DAMAGED ITEM",
                "   ",
            ],
            "Quantity": [1, 2, 1, 1, 3, 1, 2, 1, 1],
        }
    )
    original_df = df.copy(deep=True)

    result = standardize_descriptions(df)

    # 2. Input must remain unchanged.
    assert_frame_equal(df, original_df)

    # 3. Verify required output columns.
    expected_columns = [
        "Original Description",
        "Description Data Quality Note",
        "Description Present Flag",
        "Description Standardized",
    ]
    for column in expected_columns:
        assert column in result.columns

    assert len(result) == len(df)

    # 4. Verify whitespace and case normalization.
    assert result.loc[0, "Description"] == "RED COTTON SHIRT"
    assert result.loc[3, "Description"] == "BLUE CERAMIC MUG"
    assert result.loc[4, "Description"] == "BLUE CERAMIC MUG"

    # 5. Verify the canonical description for A101.
    assert result.loc[0, "Description"] == result.loc[1, "Description"]
    assert result.loc[1, "Description"] == "RED COTTON SHIRT"

    # 6. Data-quality notes should be identified.
    assert int(result.loc[2, "Description Data Quality Note"]) == 1
    assert int(result.loc[7, "Description Data Quality Note"]) == 1

    # 7. A data-quality note must not become the canonical description
    # when a valid description exists for that StockCode.
    assert result.loc[2, "Description"] == "RED COTTON SHIRT"

    # 8. Original descriptions must be preserved.
    assert result.loc[0, "Original Description"] == "  red   cotton shirt "
    assert result.loc[2, "Original Description"] == "wrongly marked item"
    assert pd.isna(result.loc[5, "Original Description"])

    # 9. Description Present Flag reflects the original input.
    assert int(result.loc[5, "Description Present Flag"]) == 0
    assert int(result.loc[6, "Description Present Flag"]) == 1
    assert int(result.loc[0, "Description Present Flag"]) == 1

    # 10. A missing description uses the canonical description for
    # its StockCode when a valid description exists.
    assert result.loc[5, "Description"] == "GREEN GLASS BOTTLE"

    # 11. Whitespace-only descriptions normalize to an empty string
    # under the current implementation; fillna does not replace "".
    assert result.loc[8, "Description"] == ""

    # 12. StockCode values must remain unchanged.
    assert result["StockCode"].equals(df["StockCode"])

    # 13. Description Standardized compares the normalized original
    # description with the final canonical description.
    # Formatting-only changes do not set this flag.
    assert bool(result.loc[0, "Description Standardized"]) is False

    # A data-quality note replaced by a canonical description is changed.
    assert bool(result.loc[2, "Description Standardized"]) is True

    # A missing original description replaced by a canonical description
    # is also marked as standardized.
    assert bool(result.loc[5, "Description Standardized"]) is True

    # 14. Verify the output dtypes.
    assert str(result["Description Data Quality Note"].dtype) == "int8"
    assert str(result["Description Present Flag"].dtype) == "int8"
    assert isinstance(result["Description"].dtype, pd.StringDtype)
    assert isinstance(result["Original Description"].dtype, pd.StringDtype)

    # 15. Missing StockCode column must raise ValueError.
    with pytest.raises(ValueError, match="Missing required columns"):
        standardize_descriptions(df.drop(columns=["StockCode"]))

    # 16. Missing Description column must raise ValueError.
    with pytest.raises(ValueError, match="Missing required columns"):
        standardize_descriptions(df.drop(columns=["Description"]))

    # 17. Empty DataFrame with required columns should be supported.
    empty_df = pd.DataFrame(columns=["StockCode", "Description"])
    empty_result = standardize_descriptions(empty_df)

    assert empty_result.empty
    assert "Original Description" in empty_result.columns
    assert "Description Present Flag" in empty_result.columns
    assert "Description Data Quality Note" in empty_result.columns
    assert "Description Standardized" in empty_result.columns

    # 18. StockCode lookup should ignore case and surrounding whitespace.
    stockcode_df = pd.DataFrame(
        {
            "StockCode": [" abc ", "ABC", "Abc"],
            "Description": [
                "silver key ring",
                "SILVER KEY RING",
                "wrongly coded product",
            ],
        }
    )

    stockcode_result = standardize_descriptions(stockcode_df)

    assert stockcode_result["Description"].tolist() == [
        "SILVER KEY RING",
        "SILVER KEY RING",
        "SILVER KEY RING",
    ]

    # The data-quality note is flagged even though its description
    # is replaced by the canonical value.
    assert int(stockcode_result.loc[2, "Description Data Quality Note"]) == 1
    assert bool(
        stockcode_result.loc[2, "Description Standardized"]
    ) is True

def test_handle_customer_id():
    """Test missing Customer ID handling while preserving all transactions."""
    import pandas as pd
    import pytest
    from pandas.testing import assert_frame_equal

    from src.data.clean import handle_customer_id

    # 1. Mixed data: existing IDs, missing IDs, and duplicate transactions.
    input_df = pd.DataFrame(
        {
            "Invoice": ["10001", "10002", "10002", "10003"],
            "StockCode": ["A101", "B202", "B202", "C303"],
            "Customer ID": [12345, None, 67890, float("nan")],
            "Quantity": [2, 1, 1, 3],
        }
    )
    original_df = input_df.copy(deep=True)

    result = handle_customer_id(input_df)

    # 2. Verify the input DataFrame was not modified.
    assert_frame_equal(input_df, original_df)

    # 3. Verify all transaction rows and their order are preserved.
    assert len(result) == len(input_df)
    assert result["Invoice"].tolist() == input_df["Invoice"].tolist()
    assert result["StockCode"].tolist() == input_df["StockCode"].tolist()
    assert result["Quantity"].tolist() == input_df["Quantity"].tolist()

    # 4. Verify existing IDs are preserved and missing IDs become zero.
    assert result["Customer ID"].tolist() == [12345, 0, 67890, 0]
    assert result["Customer ID"].dtype == "int64"

    # 5. Verify the missing-ID flag reflects the original missing values.
    assert result["Customer ID Missing Flag"].tolist() == [0, 1, 0, 1]
    assert result["Customer ID Missing Flag"].dtype == "int8"

    # 6. Verify the new flag is added without removing existing columns.
    assert "Invoice" in result.columns
    assert "StockCode" in result.columns
    assert "Quantity" in result.columns
    assert "Customer ID Missing Flag" in result.columns

    # 7. Verify duplicate transaction rows are retained.
    duplicate_rows = result[
        (result["Invoice"] == "10002")
        & (result["StockCode"] == "B202")
    ]
    assert len(duplicate_rows) == 2

    # 8. Verify missing Customer ID column raises the expected error.
    df_without_customer_id = pd.DataFrame(
        {"Invoice": ["10001"], "Quantity": [2]}
    )

    with pytest.raises(
        ValueError,
        match="Missing required column: Customer ID",
    ):
        handle_customer_id(df_without_customer_id)

    # 9. Verify an empty DataFrame with the required column works.
    empty_df = pd.DataFrame(
        {
            "Invoice": pd.Series(dtype="object"),
            "Customer ID": pd.Series(dtype="float64"),
        }
    )

    empty_result = handle_customer_id(empty_df)

    assert empty_result.empty
    assert "Customer ID Missing Flag" in empty_result.columns
    assert empty_result["Customer ID"].dtype == "int64"
    assert empty_result["Customer ID Missing Flag"].dtype == "int8"

def test_clean_transaction_data():
    """Test the complete transaction-data cleaning pipeline."""
    import pandas as pd
    import pytest
    import warnings

    warnings.filterwarnings(
    "ignore",
    message=r".*This pattern.*has match groups.*",
    category=UserWarning,)
    from pandas.testing import assert_frame_equal

    from src.data.clean import clean_transaction_data

    # 1. Create sample data covering all four cleaning steps.
    input_df = pd.DataFrame(
        {
            "Invoice": ["10001", "10002", "10002", "C10003", "10004"],
            "StockCode": ["A101", "B202", "B202", "C303", "D404"],
            "Description": [
                "  white   product  ",
                "blue product",
                "blue product",
                "returned product",
                None,
            ],
            "Quantity": [2, 1, 1, -2, 3],
            "Price": [10.0, 20.0, 20.0, 5.0, 15.0],
            "Customer ID": [12345, None, 67890, 12345, None],
            "Country": ["UK", " France ", " France ", "UK", "Germany"],
        }
    )

    # Add an exact duplicate row to test duplicate removal.
    input_df = pd.concat(
        [input_df, input_df.iloc[[0]]],
        ignore_index=True,
    )
    original_df = input_df.copy(deep=True)

    # 2. Execute the complete cleaning pipeline.
    result = clean_transaction_data(input_df)

    # 3. Verify the original DataFrame was not modified.
    assert_frame_equal(input_df, original_df)

    # 4. Verify exact duplicates are removed.
    assert len(result) == len(input_df) - 1
    assert not result.duplicated().any()

    # 5. Verify standardized column types.
    assert isinstance(result["Invoice"].dtype, pd.StringDtype)
    assert isinstance(result["StockCode"].dtype, pd.StringDtype)
    assert isinstance(result["Description"].dtype, pd.StringDtype)
    assert isinstance(result["Country"].dtype, pd.StringDtype)

    # 6. Verify quantity and price treatment columns exist.
    expected_columns = [
        "Quantity IQR Outlier",
        "Price IQR Outlier",
        "Transaction Type",
        "Quantity Treatment",
        "Transaction Value",
        "Price Treatment",
        "Include In Sales Analysis",
        "Analytical Transaction Value",
    ]
    for column in expected_columns:
        assert column in result.columns

    # 7. Verify description standardization and tracking columns.
    assert result.loc[0, "Description"] == "WHITE PRODUCT"
    assert result.loc[0, "Original Description"] == "  white   product  "
    assert "Description Standardized" in result.columns
    assert "Description Data Quality Note" in result.columns

    # 8. Verify missing Customer IDs are replaced and flagged.
    assert result["Customer ID"].tolist() == [
        12345, 0, 67890, 12345, 0
    ]
    assert result["Customer ID"].dtype == "int64"
    assert result["Customer ID Missing Flag"].tolist() == [
        0, 1, 0, 0, 1
    ]
    assert result["Customer ID Missing Flag"].dtype == "int8"

    # 9. Verify cancellation classification.
    cancellation = result.loc[result["Invoice"] == "C10003"]
    assert len(cancellation) == 1
    assert cancellation.iloc[0]["Transaction Type"] == "Cancellation"

    # 10. Verify transaction values are calculated correctly.
    assert result.loc[0, "Transaction Value"] == 20.0
    assert result.loc[1, "Transaction Value"] == 20.0

    # 11. Verify missing Description currently raises KeyError.
    # The master function accesses Description before validating columns.
    df_without_description = pd.DataFrame(
        {
            "Invoice": ["10001"],
            "StockCode": ["A101"],
        }
    )

    with pytest.raises(KeyError, match="Description"):
        clean_transaction_data(df_without_description)

    # 12. Verify missing required columns for quantity/price processing
    # are rejected when the earlier required columns are available.
    df_without_quantity = pd.DataFrame(
        {
            "Invoice": ["10001"],
            "StockCode": ["A101"],
            "Description": ["WHITE PRODUCT"],
            "Country": ["UK"],
            "Customer ID": [12345],
            "Price": [10.0],
        }
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        clean_transaction_data(df_without_quantity)


#######################################################################################################
#---------------------------- testing database engine -----------------------------------------
######################################################################################################

def test_get_database_engine(monkeypatch):
    import pytest

    from sqlalchemy.engine import URL
    import src.utils.database as database

    # ---------------------------------------------------------
    # 1. Prepare environment variables
    # ---------------------------------------------------------

    monkeypatch.setenv("DB_PASSWORD", "test_password")
    monkeypatch.setenv("DB_USER", "test_user")
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_PORT", "5433")
    monkeypatch.setenv("DB_NAME", "test_database")

    captured = {}

    # ---------------------------------------------------------
    # 2. Mock create_engine
    # ---------------------------------------------------------

    def mock_create_engine(database_url, **kwargs):
        captured["database_url"] = database_url
        captured["kwargs"] = kwargs

        return "mock_database_engine"

    monkeypatch.setattr(
        database,
        "create_engine",
        mock_create_engine
    )

    # ---------------------------------------------------------
    # 3. Execute function
    # ---------------------------------------------------------

    engine = database.get_database_engine()

    # ---------------------------------------------------------
    # 4. Validate returned engine
    # ---------------------------------------------------------

    assert engine == "mock_database_engine"

    # ---------------------------------------------------------
    # 5. Validate database URL
    # ---------------------------------------------------------

    database_url = captured["database_url"]

    assert isinstance(database_url, URL)

    assert database_url.drivername == "postgresql+psycopg"
    assert database_url.username == "test_user"
    assert database_url.password == "test_password"
    assert database_url.host == "localhost"
    assert database_url.port == 5433
    assert database_url.database == "test_database"

    # ---------------------------------------------------------
    # 6. Validate connection configuration
    # ---------------------------------------------------------

    engine_kwargs = captured["kwargs"]

    assert engine_kwargs["pool_pre_ping"] is True

    assert engine_kwargs["connect_args"] == {
        "connect_timeout": 5
    }

    # ---------------------------------------------------------
    # 7. Test default environment variables
    # ---------------------------------------------------------

    monkeypatch.delenv("DB_USER")
    monkeypatch.delenv("DB_HOST")
    monkeypatch.delenv("DB_PORT")
    monkeypatch.delenv("DB_NAME")

    captured.clear()

    database.get_database_engine()

    default_url = captured["database_url"]

    assert default_url.username == "postgres"
    assert default_url.host == "127.0.0.1"
    assert default_url.port == 5432
    assert default_url.database == "vantara_db"

    # ---------------------------------------------------------
    # 8. Test missing database password
    # ---------------------------------------------------------

    monkeypatch.delenv("DB_PASSWORD")

    with pytest.raises(
        ValueError,
        match="DB_PASSWORD environment variable is not set"
    ):
        database.get_database_engine()

def test_database_connection(monkeypatch):
    from unittest.mock import MagicMock

    import pytest

    import src.utils.database as database

    # ---------------------------------------------------------
    # 1. Create mock database objects
    # ---------------------------------------------------------

    mock_engine = MagicMock()
    mock_connection = MagicMock()
    mock_result = MagicMock()

    # ---------------------------------------------------------
    # 2. Configure successful database response
    # ---------------------------------------------------------

    mock_result.scalar_one.return_value = "vantara_db"

    mock_connection.execute.return_value = mock_result

    mock_engine.connect.return_value.__enter__.return_value = (
        mock_connection
    )

    # ---------------------------------------------------------
    # 3. Mock get_database_engine
    # ---------------------------------------------------------

    monkeypatch.setattr(
        database,
        "get_database_engine",
        lambda: mock_engine
    )

    # ---------------------------------------------------------
    # 4. Execute database connection test
    # ---------------------------------------------------------

    result = database.test_database_connection()

    # ---------------------------------------------------------
    # 5. Validate returned database name
    # ---------------------------------------------------------

    assert result == "vantara_db"

    # ---------------------------------------------------------
    # 6. Validate connection execution
    # ---------------------------------------------------------

    mock_engine.connect.assert_called_once()

    mock_connection.execute.assert_called_once()

    executed_query = mock_connection.execute.call_args[0][0]

    assert str(executed_query) == "SELECT current_database();"

    mock_result.scalar_one.assert_called_once()

    # ---------------------------------------------------------
    # 7. Validate engine disposal after successful execution
    # ---------------------------------------------------------

    mock_engine.dispose.assert_called_once()

    # ---------------------------------------------------------
    # 8. Test connection failure
    # ---------------------------------------------------------

    mock_engine.reset_mock()

    mock_engine.connect.side_effect = (
        Exception("Database connection failed")
    )

    with pytest.raises(
        Exception,
        match="Database connection failed"
    ):
        database.test_database_connection()

    mock_engine.dispose.assert_called_once()

    # ---------------------------------------------------------
    # 9. Test query execution failure
    # ---------------------------------------------------------

    mock_engine.reset_mock()

    # Clear previous connection exception
    mock_engine.connect.side_effect = None

    mock_connection.reset_mock()

    mock_connection.execute.side_effect = (
        Exception("Database query execution failed")
    )

    mock_engine.connect.return_value.__enter__.return_value = (
        mock_connection
    )

    with pytest.raises(
        Exception,
        match="Database query execution failed"
    ):
        database.test_database_connection()

    mock_engine.dispose.assert_called_once()

    mock_connection.execute.assert_called_once()

    # ---------------------------------------------------------
    # 10. Test scalar result retrieval failure
    # ---------------------------------------------------------

    mock_engine.reset_mock()
    mock_connection.reset_mock()

    # Clear previous query execution exception
    mock_connection.execute.side_effect = None

    mock_connection.execute.return_value = mock_result

    mock_result.scalar_one.side_effect = (
        Exception("Database result retrieval failed")
    )

    mock_engine.connect.return_value.__enter__.return_value = (
        mock_connection
    )

    with pytest.raises(
        Exception,
        match="Database result retrieval failed"
    ):
        database.test_database_connection()

    mock_engine.dispose.assert_called_once()

    mock_result.scalar_one.assert_called_once()




########################################################################################################
#--------------------------- test_prepare_lstm_product_taxonomy
########################################################################################################

def test_prepare_lstm_product_taxonomy(sample_discount_sensitivity_transactions):
    import pandas as pd
    import pytest

    from src.features.features import prepare_lstm_product_taxonomy

    # ---------------------------------------------------------
    # 1. Prepare transaction fixture
    # ---------------------------------------------------------

    df = sample_discount_sensitivity_transactions.copy(deep=True)

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce"
    )

    df["Transaction Type"] = "Purchase"
    df["Include In Sales Analysis"] = True

    df["Analytical Transaction Value"] = (
        pd.to_numeric(df["Price"], errors="coerce")
        * pd.to_numeric(df["Quantity"], errors="coerce")
    )

    # ---------------------------------------------------------
    # 2. Prepare explicit product taxonomy
    # ---------------------------------------------------------

    product_catalog = pd.DataFrame({
        "StockCode_Key": ["P001", "P002"],
        "Product_Category": [
            "Category A",
            "Category B",
        ],
    })

    taxonomy_result = {
        "product_catalog": product_catalog
    }

    # ---------------------------------------------------------
    # 3. Execute function
    # ---------------------------------------------------------

    result = prepare_lstm_product_taxonomy(
        df_transactions=df,
        taxonomy_result=taxonomy_result
    )

    # ---------------------------------------------------------
    # 4. Validate returned structure
    # ---------------------------------------------------------

    expected_keys = {
        "product_catalog",
        "lstm_transactions",
        "unmatched_products",
        "excluded_admin_transactions",
        "taxonomy_audit",
    }

    assert isinstance(result, dict)
    assert expected_keys.issubset(result.keys())

    catalog = result["product_catalog"]
    lstm_transactions = result["lstm_transactions"]
    unmatched_products = result["unmatched_products"]
    excluded_admin = result["excluded_admin_transactions"]
    audit = result["taxonomy_audit"]

    # ---------------------------------------------------------
    # 5. Validate product catalog
    # ---------------------------------------------------------

    assert len(catalog) == 2

    assert catalog["StockCode_Key"].tolist() == [
        "P001",
        "P002",
    ]

    assert catalog["StockCode_Key"].is_unique
    assert catalog["StockCode_Key"].notna().all()
    assert catalog["Product_Category"].notna().all()

    # ---------------------------------------------------------
    # 6. Validate eligible LSTM transactions
    # ---------------------------------------------------------

    assert len(lstm_transactions) == 7

    assert set(lstm_transactions["Customer ID"].tolist()) == {
        101, 202, 404
    }

    assert lstm_transactions["Customer ID"].notna().all()
    assert lstm_transactions["Customer ID"].ne(0).all()

    assert (
        lstm_transactions["Transaction Type"] == "Purchase"
    ).all()

    assert (
        lstm_transactions["Include In Sales Analysis"] == True
    ).all()

    assert lstm_transactions["InvoiceDate"].notna().all()

    # ---------------------------------------------------------
    # 7. Validate stock-code standardization
    # ---------------------------------------------------------

    assert "StockCode_Key" in lstm_transactions.columns

    assert lstm_transactions["StockCode_Key"].notna().all()

    assert (
        lstm_transactions["StockCode_Key"]
        == lstm_transactions["StockCode_Key"].str.strip().str.upper()
    ).all()

    admin_stock_codes = {
        "23444",
        "23595",
        "ADJUST",
        "ADJUST2",
        "AMAZONFEE",
        "BANK CHARGES",
        "C2",
        "D",
        "DOT",
        "M",
        "POST",
    }

    assert not lstm_transactions["StockCode_Key"].isin(
        admin_stock_codes
    ).any()

    # ---------------------------------------------------------
    # 8. Validate taxonomy matching
    # ---------------------------------------------------------

    assert "Product_Category" in lstm_transactions.columns

    assert "Taxonomy_Match_Status" in lstm_transactions.columns

    assert (
        lstm_transactions["Taxonomy_Match_Status"] == "Matched"
    ).all()

    assert lstm_transactions["Product_Category"].notna().all()

    assert set(
        lstm_transactions["Product_Category"].tolist()
    ) == {
        "Category A",
        "Category B",
    }

    assert len(unmatched_products) == 0

    assert unmatched_products.columns.tolist() == [
        "StockCode_Key"
    ]

    # ---------------------------------------------------------
    # 9. Validate administrative transaction exclusion
    # ---------------------------------------------------------

    assert len(excluded_admin) == 2

    assert excluded_admin["StockCode_Key"].isin(
        ["D"]
    ).all()

    assert excluded_admin["Is_Admin_StockCode"].all()

    assert excluded_admin["Customer ID"].notna().all()

    assert excluded_admin["Customer ID"].ne(0).all()

    # ---------------------------------------------------------
    # 10. Validate chronological sorting
    # ---------------------------------------------------------

    assert lstm_transactions.index.equals(
        pd.RangeIndex(len(lstm_transactions))
    )

    expected_sorted = lstm_transactions.sort_values(
        [
            "Customer ID",
            "InvoiceDate",
            "StockCode_Key",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        lstm_transactions,
        expected_sorted
    )

    # ---------------------------------------------------------
    # 11. Validate audit summary
    # ---------------------------------------------------------

    assert audit["input_transaction_rows"] == 10

    assert (
        audit["eligible_purchase_rows_before_admin_exclusion"]
        == 9
    )

    assert audit["eligible_admin_rows_excluded"] == 2

    assert audit["eligible_retail_rows_after_exclusion"] == 7

    assert audit["matched_transaction_rows"] == 7

    assert audit["unmatched_transaction_rows"] == 0

    assert audit["unmatched_unique_products"] == 0

    assert audit["unique_customers"] == 3

    assert audit["unique_products_in_transactions"] == 2

    assert audit["unique_products_in_catalog"] == 2

    # Corrected counts based on the actual fixture mapping
    assert audit["category_counts"] == {
        "Category A": 5,
        "Category B": 2,
    }

    assert sum(
        audit["category_counts"].values()
    ) == audit["eligible_retail_rows_after_exclusion"]

    # ---------------------------------------------------------
    # 12. Test unmatched products
    # ---------------------------------------------------------

    df_unmatched = df.copy(deep=True)

    unmatched_row = df_unmatched.iloc[[0]].copy()

    unmatched_row["Invoice"] = "UNMATCHED001"
    unmatched_row["StockCode"] = "UNKNOWN_PRODUCT"

    df_unmatched = pd.concat(
        [df_unmatched, unmatched_row],
        ignore_index=True
    )

    unmatched_result = prepare_lstm_product_taxonomy(
        df_transactions=df_unmatched,
        taxonomy_result=taxonomy_result
    )

    unmatched_transactions = unmatched_result["lstm_transactions"]

    unmatched_audit = unmatched_result["taxonomy_audit"]

    assert len(unmatched_transactions) == 8

    assert unmatched_audit["unmatched_transaction_rows"] == 1

    assert unmatched_audit["unmatched_unique_products"] == 1

    assert unmatched_audit["matched_transaction_rows"] == 7

    assert unmatched_result["unmatched_products"][
        "StockCode_Key"
    ].tolist() == ["UNKNOWN_PRODUCT"]

    unmatched_row_result = unmatched_transactions.loc[
        unmatched_transactions["StockCode_Key"] == "UNKNOWN_PRODUCT"
    ].iloc[0]

    assert unmatched_row_result["Taxonomy_Match_Status"] == "Unmatched"

    assert pd.isna(unmatched_row_result["Product_Category"])

    # ---------------------------------------------------------
    # 13. Test missing required transaction columns
    # ---------------------------------------------------------

    df_missing = df.drop(
        columns=["Customer ID"]
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns"
    ):
        prepare_lstm_product_taxonomy(
            df_transactions=df_missing,
            taxonomy_result=taxonomy_result
        )

    # ---------------------------------------------------------
    # 14. Test missing product_catalog in taxonomy result
    # ---------------------------------------------------------

    with pytest.raises(
        KeyError,
        match="Taxonomy result must contain 'product_catalog'"
    ):
        prepare_lstm_product_taxonomy(
            df_transactions=df,
            taxonomy_result={}
        )

    # ---------------------------------------------------------
    # 15. Test missing catalog columns
    # ---------------------------------------------------------

    invalid_catalog = pd.DataFrame({
        "StockCode_Key": ["P001", "P002"]
    })

    with pytest.raises(
        ValueError,
        match="Product catalog is missing columns"
    ):
        prepare_lstm_product_taxonomy(
            df_transactions=df,
            taxonomy_result={
                "product_catalog": invalid_catalog
            }
        )

    # ---------------------------------------------------------
    # 16. Test missing stock-code keys in catalog
    # ---------------------------------------------------------

    catalog_missing_keys = pd.DataFrame({
        "StockCode_Key": ["P001", None],
        "Product_Category": ["Category A", "Category B"],
    })

    with pytest.raises(
        ValueError,
        match="Product catalog contains missing stock-code keys"
    ):
        prepare_lstm_product_taxonomy(
            df_transactions=df,
            taxonomy_result={
                "product_catalog": catalog_missing_keys
            }
        )

    # ---------------------------------------------------------
    # 17. Test duplicate stock-code keys in catalog
    # ---------------------------------------------------------

    catalog_duplicate = pd.DataFrame({
        "StockCode_Key": ["P001", "P001"],
        "Product_Category": ["Category A", "Category B"],
    })

    with pytest.raises(
        ValueError,
        match="Product catalog must contain one row per product"
    ):
        prepare_lstm_product_taxonomy(
            df_transactions=df,
            taxonomy_result={
                "product_catalog": catalog_duplicate
            }
        )

    # ---------------------------------------------------------
    # 18. Test missing product categories
    # ---------------------------------------------------------

    catalog_missing_category = pd.DataFrame({
        "StockCode_Key": ["P001", "P002"],
        "Product_Category": ["Category A", None],
    })

    with pytest.raises(
        ValueError,
        match="Product catalog contains missing product categories"
    ):
        prepare_lstm_product_taxonomy(
            df_transactions=df,
            taxonomy_result={
                "product_catalog": catalog_missing_category
            }
        )

def test_build_lstm_purchase_events(sample_discount_sensitivity_transactions):
    import pandas as pd
    import pytest

    from src.features.features import (
        prepare_lstm_product_taxonomy,
        build_lstm_purchase_events,
    )

    # ---------------------------------------------------------
    # 1. Prepare transaction fixture
    # ---------------------------------------------------------

    df = sample_discount_sensitivity_transactions.copy(deep=True)

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce"
    )

    df["Transaction Type"] = "Purchase"
    df["Include In Sales Analysis"] = True

    df["Analytical Transaction Value"] = (
        pd.to_numeric(df["Price"], errors="coerce")
        * pd.to_numeric(df["Quantity"], errors="coerce")
    )

    # ---------------------------------------------------------
    # 2. Prepare explicit product taxonomy
    # ---------------------------------------------------------

    product_catalog = pd.DataFrame({
        "StockCode_Key": ["P001", "P002"],
        "Product_Category": [
            "Category A",
            "Category B",
        ],
    })

    taxonomy_result = {
        "product_catalog": product_catalog
    }

    # ---------------------------------------------------------
    # 3. Prepare eligible LSTM transactions
    # ---------------------------------------------------------

    lstm_taxonomy_result = prepare_lstm_product_taxonomy(
        df_transactions=df,
        taxonomy_result=taxonomy_result
    )

    lstm_transactions = lstm_taxonomy_result[
        "lstm_transactions"
    ]

    assert len(lstm_transactions) == 7

    # ---------------------------------------------------------
    # 4. Execute purchase-event aggregation
    # ---------------------------------------------------------

    result = build_lstm_purchase_events(
        lstm_taxonomy_result=lstm_taxonomy_result
    )

    # ---------------------------------------------------------
    # 5. Validate returned structure
    # ---------------------------------------------------------

    expected_keys = {
        "purchase_events",
        "event_category_features",
        "event_audit",
    }

    assert isinstance(result, dict)
    assert expected_keys.issubset(result.keys())

    purchase_events = result["purchase_events"]
    event_category_features = result["event_category_features"]
    audit = result["event_audit"]

    # ---------------------------------------------------------
    # 6. Validate purchase-event structure
    # ---------------------------------------------------------

    expected_event_columns = {
        "Customer ID",
        "Invoice_ID",
        "InvoiceDate",
        "Purchase_Value",
        "Product_Line_Count",
        "Unique_Product_Count",
        "Total_Quantity",
    }

    assert expected_event_columns.issubset(
        purchase_events.columns
    )

    assert len(purchase_events) == 7

    assert purchase_events["Customer ID"].notna().all()
    assert purchase_events["Invoice_ID"].notna().all()
    assert purchase_events["InvoiceDate"].notna().all()

    assert purchase_events["Customer ID"].is_unique is False or (
        purchase_events[
            ["Customer ID", "Invoice_ID"]
        ].duplicated().sum() == 0
    )

    assert not purchase_events[
        ["Customer ID", "Invoice_ID"]
    ].duplicated().any()

    # ---------------------------------------------------------
    # 7. Validate invoice-level aggregation
    # ---------------------------------------------------------

    expected_invoice_values = {
        "I001": 10.0,
        "I002": 8.0,
        "I003": 5.0,
        "I004": 20.0,
        "I005": 7.0,
        "I006": 10.0,
        "I009": 20.0,
    }

    for invoice_id, expected_value in expected_invoice_values.items():

        event = purchase_events.loc[
            purchase_events["Invoice_ID"] == invoice_id
        ]

        assert len(event) == 1

        assert round(
            float(event.iloc[0]["Purchase_Value"]), 10
        ) == round(expected_value, 10)

        assert event.iloc[0]["Product_Line_Count"] == 1

        assert event.iloc[0]["Unique_Product_Count"] == 1

    # ---------------------------------------------------------
    # 8. Validate customer and invoice relationships
    # ---------------------------------------------------------

    expected_invoice_customers = {
        "I001": 101,
        "I002": 101,
        "I003": 101,
        "I004": 101,
        "I005": 202,
        "I006": 202,
        "I009": 404,
    }

    for invoice_id, expected_customer in expected_invoice_customers.items():

        event = purchase_events.loc[
            purchase_events["Invoice_ID"] == invoice_id
        ].iloc[0]

        assert event["Customer ID"] == expected_customer

    assert set(purchase_events["Customer ID"].tolist()) == {
        101, 202, 404
    }

    # ---------------------------------------------------------
    # 9. Validate purchase dates and timestamps
    # ---------------------------------------------------------

    assert pd.api.types.is_datetime64_any_dtype(
        purchase_events["InvoiceDate"]
    )

    assert purchase_events["InvoiceDate"].notna().all()

    expected_dates = {
        "I001": pd.Timestamp("2026-01-01"),
        "I002": pd.Timestamp("2026-01-05"),
        "I003": pd.Timestamp("2026-01-10"),
        "I004": pd.Timestamp("2026-01-15"),
        "I005": pd.Timestamp("2026-01-03"),
        "I006": pd.Timestamp("2026-01-08"),
        "I009": pd.Timestamp("2026-01-20"),
    }

    for invoice_id, expected_date in expected_dates.items():

        event = purchase_events.loc[
            purchase_events["Invoice_ID"] == invoice_id
        ].iloc[0]

        assert event["InvoiceDate"] == expected_date

    # ---------------------------------------------------------
    # 10. Validate product line and quantity aggregation
    # ---------------------------------------------------------

    assert (
        purchase_events["Product_Line_Count"].sum() == 7
    )

    assert (
        purchase_events["Unique_Product_Count"].sum() == 7
    )

    assert purchase_events["Total_Quantity"].notna().all()

    expected_total_quantity = int(
        lstm_transactions["Quantity"].sum()
    )

    assert (
        purchase_events["Total_Quantity"].sum()
        == expected_total_quantity
    )

    # ---------------------------------------------------------
    # 11. Validate category-level event features
    # ---------------------------------------------------------

    expected_category_columns = {
        "Customer ID",
        "Invoice_ID",
        "Category_Category_A_Line_Count",
        "Category_Category_B_Line_Count",
        "Category_Category_A_Value",
        "Category_Category_B_Value",
    }

    assert expected_category_columns.issubset(
        event_category_features.columns
    )

    assert len(event_category_features) == 7

    assert not event_category_features[
        ["Customer ID", "Invoice_ID"]
    ].duplicated().any()

    assert event_category_features[
        [
            "Category_Category_A_Line_Count",
            "Category_Category_B_Line_Count",
            "Category_Category_A_Value",
            "Category_Category_B_Value",
        ]
    ].notna().all().all()

    # ---------------------------------------------------------
    # 12. Validate category line counts
    # ---------------------------------------------------------

    category_a_lines = (
        event_category_features[
            "Category_Category_A_Line_Count"
        ].sum()
    )

    category_b_lines = (
        event_category_features[
            "Category_Category_B_Line_Count"
        ].sum()
    )

    assert category_a_lines == 5
    assert category_b_lines == 2

    assert (
        category_a_lines + category_b_lines == 7
    )

    # ---------------------------------------------------------
    # 13. Validate category purchase values
    # ---------------------------------------------------------

    category_a_value = (
        event_category_features[
            "Category_Category_A_Value"
        ].sum()
    )

    category_b_value = (
        event_category_features[
            "Category_Category_B_Value"
        ].sum()
    )

    assert round(float(category_a_value), 10) == 40.0

    assert round(float(category_b_value), 10) == 40.0

    # ---------------------------------------------------------
    # 14. Validate merged category features
    # ---------------------------------------------------------

    category_columns = [
        column
        for column in purchase_events.columns
        if column.startswith("Category_")
    ]

    assert len(category_columns) == 4

    assert purchase_events[category_columns].notna().all().all()

    assert (
        purchase_events["Category_Category_A_Line_Count"].sum()
        == 5
    )

    assert (
        purchase_events["Category_Category_B_Line_Count"].sum()
        == 2
    )

    assert round(
        float(purchase_events["Category_Category_A_Value"].sum()),
        10
    ) == 40.0

    assert round(
        float(purchase_events["Category_Category_B_Value"].sum()),
        10
    ) == 40.0

    # ---------------------------------------------------------
    # 15. Validate chronological sorting
    # ---------------------------------------------------------

    assert purchase_events.index.equals(
        pd.RangeIndex(len(purchase_events))
    )

    expected_sorted = purchase_events.sort_values(
        [
            "Customer ID",
            "InvoiceDate",
            "Invoice_ID",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        purchase_events,
        expected_sorted
    )

    # ---------------------------------------------------------
    # 16. Validate total purchase value reconciliation
    # ---------------------------------------------------------

    transaction_total = (
        lstm_transactions[
            "Analytical Transaction Value"
        ].sum()
    )

    event_total = purchase_events["Purchase_Value"].sum()

    assert round(float(transaction_total), 10) == 80.0

    assert round(float(event_total), 10) == 80.0

    assert round(float(transaction_total), 10) == (
        round(float(event_total), 10)
    )

    # ---------------------------------------------------------
    # 17. Validate event audit
    # ---------------------------------------------------------

    assert audit["input_transaction_rows"] == 7

    assert audit["purchase_event_count"] == 7

    assert audit["unique_customers"] == 3

    assert audit["unique_invoices"] == 7

    assert audit["duplicate_customer_invoice_events"] == 0

    assert audit["category_line_count_total"] == 7

    assert bool(
        audit["category_line_counts_reconcile"]
    ) is True

    assert round(
        audit["total_purchase_value_transactions"], 10
    ) == 80.0

    assert round(
        audit["total_purchase_value_events"], 10
    ) == 80.0

    assert audit["category_feature_column_count"] == 4

    # ---------------------------------------------------------
    # 18. Test missing lstm_transactions key
    # ---------------------------------------------------------

    with pytest.raises(
        KeyError,
        match="Input must contain 'lstm_transactions'"
    ):
        build_lstm_purchase_events(
            lstm_taxonomy_result={}
        )

    # ---------------------------------------------------------
    # 19. Test empty transaction dataframe
    # ---------------------------------------------------------

    empty_transactions = lstm_transactions.iloc[0:0].copy()

    with pytest.raises(
        ValueError,
        match="No transactions available for purchase-event creation"
    ):
        build_lstm_purchase_events(
            lstm_taxonomy_result={
                "lstm_transactions": empty_transactions
            }
        )

    # ---------------------------------------------------------
    # 20. Test invoice column not found
    # ---------------------------------------------------------

    df_no_invoice = lstm_transactions.drop(
        columns=["Invoice"]
    )

    with pytest.raises(
        ValueError,
        match="Invoice column not found"
    ):
        build_lstm_purchase_events(
            lstm_taxonomy_result={
                "lstm_transactions": df_no_invoice
            }
        )

    # ---------------------------------------------------------
    # 21. Test explicitly supplied invalid invoice column
    # ---------------------------------------------------------

    with pytest.raises(
        ValueError,
        match="Invoice column not found"
    ):
        build_lstm_purchase_events(
            lstm_taxonomy_result={
                "lstm_transactions": lstm_transactions
            },
            invoice_col="InvalidInvoiceColumn"
        )

    # ---------------------------------------------------------
    # 22. Test missing required transaction columns
    # ---------------------------------------------------------

    df_missing_column = lstm_transactions.drop(
        columns=["StockCode_Key"]
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns"
    ):
        build_lstm_purchase_events(
            lstm_taxonomy_result={
                "lstm_transactions": df_missing_column
            }
        )

    # ---------------------------------------------------------
    # 23. Test missing customer, invoice or date values
    # ---------------------------------------------------------

    df_missing_values = lstm_transactions.copy(deep=True)

    df_missing_values.loc[
        df_missing_values.index[0],
        "Customer ID"
    ] = pd.NA

    with pytest.raises(
        ValueError,
        match="Missing customer IDs, invoice IDs, or invoice dates"
    ):
        build_lstm_purchase_events(
            lstm_taxonomy_result={
                "lstm_transactions": df_missing_values
            }
        )

    # ---------------------------------------------------------
    # 24. Test missing product category
    # ---------------------------------------------------------

    df_missing_category = lstm_transactions.copy(deep=True)

    df_missing_category.loc[
        df_missing_category.index[0],
        "Product_Category"
    ] = None

    with pytest.raises(
        ValueError,
        match="Some transaction rows have no product category"
    ):
        build_lstm_purchase_events(
            lstm_taxonomy_result={
                "lstm_transactions": df_missing_category
            }
        )

    # ---------------------------------------------------------
    # 25. Test negative analytical transaction value
    # ---------------------------------------------------------

    df_negative_value = lstm_transactions.copy(deep=True)

    df_negative_value.loc[
        df_negative_value.index[0],
        "Analytical Transaction Value"
    ] = -10

    with pytest.raises(
        ValueError,
        match="Analytical Transaction Value contains missing or negative values"
    ):
        build_lstm_purchase_events(
            lstm_taxonomy_result={
                "lstm_transactions": df_negative_value
            }
        )

    # ---------------------------------------------------------
    # 26. Test missing analytical transaction value
    # ---------------------------------------------------------

    df_missing_value = lstm_transactions.copy(deep=True)

    df_missing_value.loc[
        df_missing_value.index[0],
        "Analytical Transaction Value"
    ] = float("nan")

    with pytest.raises(
        ValueError,
        match="Analytical Transaction Value contains missing or negative values"
    ):
        build_lstm_purchase_events(
            lstm_taxonomy_result={
                "lstm_transactions": df_missing_value
            }
        )

    # ---------------------------------------------------------
    # 27. Test invalid analytical transaction value
    # ---------------------------------------------------------

    df_invalid_value = lstm_transactions.copy(deep=True)

    df_invalid_value["Analytical Transaction Value"] = (
        df_invalid_value["Analytical Transaction Value"].astype(object)
    )

    df_invalid_value.loc[
        df_invalid_value.index[0],
        "Analytical Transaction Value"
    ] = "invalid-value"

    with pytest.raises(
        (ValueError, TypeError)
    ):
        build_lstm_purchase_events(
            lstm_taxonomy_result={
                "lstm_transactions": df_invalid_value
            }
        )

    # ---------------------------------------------------------
    # 28. Test missing quantity
    # ---------------------------------------------------------

    df_missing_quantity = lstm_transactions.copy(deep=True)

    df_missing_quantity.loc[
        df_missing_quantity.index[0],
        "Quantity"
    ] = float("nan")

    with pytest.raises(
        ValueError,
        match="Quantity contains missing values"
    ):
        build_lstm_purchase_events(
            lstm_taxonomy_result={
                "lstm_transactions": df_missing_quantity
            }
        )

    # ---------------------------------------------------------
    # 29. Test invalid quantity
    # ---------------------------------------------------------

    df_invalid_quantity = lstm_transactions.copy(deep=True)

    df_invalid_quantity["Quantity"] = (
        df_invalid_quantity["Quantity"].astype(object)
    )

    df_invalid_quantity.loc[
        df_invalid_quantity.index[0],
        "Quantity"
    ] = "invalid-quantity"

    with pytest.raises(
        (ValueError, TypeError)
    ):
        build_lstm_purchase_events(
            lstm_taxonomy_result={
                "lstm_transactions": df_invalid_quantity
            }
        )

    # ---------------------------------------------------------
    # 30. Test alternative supported invoice column
    # ---------------------------------------------------------

    df_alternative_invoice = lstm_transactions.rename(
        columns={
            "Invoice": "InvoiceNo"
        }
    )

    alternative_result = build_lstm_purchase_events(
        lstm_taxonomy_result={
            "lstm_transactions": df_alternative_invoice
        }
    )

    alternative_events = alternative_result["purchase_events"]

    assert len(alternative_events) == 7

    assert "Invoice_ID" in alternative_events.columns

    assert set(alternative_events["Invoice_ID"].tolist()) == {
        "I001",
        "I002",
        "I003",
        "I004",
        "I005",
        "I006",
        "I009",
    }

    # ---------------------------------------------------------
    # 31. Test explicitly supplied invoice column
    # ---------------------------------------------------------

    explicit_result = build_lstm_purchase_events(
        lstm_taxonomy_result={
            "lstm_transactions": df_alternative_invoice
        },
        invoice_col="InvoiceNo"
    )

    assert len(explicit_result["purchase_events"]) == 7

    assert (
        explicit_result["event_audit"]["purchase_event_count"]
        == 7
    )

def test_create_lstm_purchase_timing_targets(sample_discount_sensitivity_transactions):
    import pandas as pd
    import pytest
    import warnings

    from src.features.features import (
        prepare_lstm_product_taxonomy,
        build_lstm_purchase_events,
        create_lstm_purchase_timing_targets,
    )

    # ---------------------------------------------------------
    # 1. Prepare transaction fixture
    # ---------------------------------------------------------

    transactions = sample_discount_sensitivity_transactions.copy(deep=True)

    transactions["Transaction Type"] = "Purchase"
    transactions["Include In Sales Analysis"] = True
    transactions["Analytical Transaction Value"] = (
        transactions["Price"] * transactions["Quantity"]
    )

    product_catalog = pd.DataFrame({
        "StockCode_Key": ["P001", "P002"],
        "Product_Category": ["Category A", "Category B"],
    })

    taxonomy_input = {
        "product_catalog": product_catalog
    }

    taxonomy_result = prepare_lstm_product_taxonomy(
        transactions,
        taxonomy_input,
    )

    event_result = build_lstm_purchase_events(taxonomy_result)

    purchase_events = event_result["purchase_events"].copy(deep=True)

    # ---------------------------------------------------------
    # 2. Execute production function
    # ---------------------------------------------------------

    result = create_lstm_purchase_timing_targets(purchase_events)

    assert isinstance(result, dict)

    assert set(result.keys()) == {
        "purchase_events_with_targets",
        "modeling_targets",
        "target_audit",
    }

    events_with_targets = result["purchase_events_with_targets"]
    modeling_targets = result["modeling_targets"]
    audit = result["target_audit"]

    assert isinstance(events_with_targets, pd.DataFrame)
    assert isinstance(modeling_targets, pd.DataFrame)
    assert isinstance(audit, dict)

    # ---------------------------------------------------------
    # 3. Validate output columns
    # ---------------------------------------------------------

    expected_columns = {
        "Customer ID",
        "Invoice_ID",
        "InvoiceDate",
        "Next_Purchase_Date",
        "Next_Invoice_ID",
        "Days_Until_Next_Purchase",
    }

    assert expected_columns.issubset(events_with_targets.columns)
    assert expected_columns.issubset(modeling_targets.columns)

    # ---------------------------------------------------------
    # 4. Validate output row counts
    # ---------------------------------------------------------

    assert len(events_with_targets) == 7
    assert len(modeling_targets) == 4

    # ---------------------------------------------------------
    # 5. Validate datetime columns
    # ---------------------------------------------------------

    assert pd.api.types.is_datetime64_any_dtype(
        events_with_targets["InvoiceDate"]
    )

    assert pd.api.types.is_datetime64_any_dtype(
        events_with_targets["Next_Purchase_Date"]
    )

    assert pd.api.types.is_datetime64_any_dtype(
        modeling_targets["Next_Purchase_Date"]
    )

    # ---------------------------------------------------------
    # 6. Validate chronological ordering within each customer
    # ---------------------------------------------------------

    # Production sorts by Customer ID and then InvoiceDate.
    # Global chronological ordering is therefore not expected.

    for customer_id, customer_events in events_with_targets.groupby(
        "Customer ID"
    ):
        assert customer_events["InvoiceDate"].is_monotonic_increasing

    # ---------------------------------------------------------
    # 7. Validate expected next purchase mapping
    # ---------------------------------------------------------

    expected_next_purchases = {
        "I001": ("I002", pd.Timestamp("2026-01-05"), 4.0),
        "I002": ("I003", pd.Timestamp("2026-01-10"), 5.0),
        "I003": ("I004", pd.Timestamp("2026-01-15"), 5.0),
        "I005": ("I006", pd.Timestamp("2026-01-08"), 5.0),
    }

    for invoice_id, expected in expected_next_purchases.items():

        next_invoice, next_date, days = expected

        event = events_with_targets.loc[
            events_with_targets["Invoice_ID"] == invoice_id
        ]

        assert len(event) == 1
        assert event.iloc[0]["Next_Invoice_ID"] == next_invoice
        assert event.iloc[0]["Next_Purchase_Date"] == next_date

        actual_days = float(
            event.iloc[0]["Days_Until_Next_Purchase"]
        )

        assert round(actual_days, 10) == round(float(days), 10)

    # ---------------------------------------------------------
    # 8. Validate final events without next purchase
    # ---------------------------------------------------------

    final_invoice_ids = {"I004", "I006", "I009"}

    final_events = events_with_targets[
        events_with_targets["Invoice_ID"].isin(final_invoice_ids)
    ]

    assert len(final_events) == 3
    assert final_events["Next_Purchase_Date"].isna().all()
    assert final_events["Next_Invoice_ID"].isna().all()
    assert final_events["Days_Until_Next_Purchase"].isna().all()

    # ---------------------------------------------------------
    # 9. Validate modeling target selection
    # ---------------------------------------------------------

    assert set(modeling_targets["Invoice_ID"]) == {
        "I001",
        "I002",
        "I003",
        "I005",
    }

    assert modeling_targets["Next_Purchase_Date"].notna().all()
    assert modeling_targets["Next_Invoice_ID"].notna().all()
    assert modeling_targets["Days_Until_Next_Purchase"].notna().all()

    assert events_with_targets["InvoiceDate"].notna().all()

    # ---------------------------------------------------------
    # 10. Validate target values
    # ---------------------------------------------------------

    expected_days = {
        "I001": 4.0,
        "I002": 5.0,
        "I003": 5.0,
        "I005": 5.0,
    }

    for invoice_id, expected_days_value in expected_days.items():

        actual = modeling_targets.loc[
            modeling_targets["Invoice_ID"] == invoice_id,
            "Days_Until_Next_Purchase",
        ].iloc[0]

        assert round(float(actual), 10) == round(
            expected_days_value, 10
        )

    # ---------------------------------------------------------
    # 11. Validate target values are non-negative
    # ---------------------------------------------------------

    assert (
        modeling_targets["Days_Until_Next_Purchase"] >= 0
    ).all()

    # ---------------------------------------------------------
    # 12. Validate customer-specific target isolation
    # ---------------------------------------------------------

    customer_101_targets = modeling_targets[
        modeling_targets["Customer ID"] == 101
    ]

    customer_202_targets = modeling_targets[
        modeling_targets["Customer ID"] == 202
    ]

    customer_404_targets = modeling_targets[
        modeling_targets["Customer ID"] == 404
    ]

    assert len(customer_101_targets) == 3
    assert len(customer_202_targets) == 1
    assert len(customer_404_targets) == 0

    # Verify every next purchase belongs to the same customer.

    for _, event in modeling_targets.iterrows():

        next_customer = events_with_targets.loc[
            events_with_targets["Invoice_ID"] == event["Next_Invoice_ID"],
            "Customer ID",
        ]

        assert len(next_customer) == 1
        assert next_customer.iloc[0] == event["Customer ID"]

    # ---------------------------------------------------------
    # 13. Validate audit summary
    # ---------------------------------------------------------

    assert audit["input_purchase_events"] == 7
    assert audit["unique_customers"] == 3
    assert audit["invalid_dates"] == 0
    assert audit["missing_dates"] == 0

    assert audit["events_with_valid_next_purchase_target"] == 4
    assert audit["events_without_next_purchase_target"] == 3

    assert audit["customers_with_one_event"] == 1
    assert audit["zero_day_targets"] == 0
    assert audit["negative_targets"] == 0

    assert audit["target_min_days"] == 4.0
    assert audit["target_median_days"] == 5.0
    assert audit["target_max_days"] == 5.0

    # Support both pandas datetime precisions.

    assert audit["invoice_date_dtype"] in (
        "datetime64[ns]",
        "datetime64[us]",
    )

    # ---------------------------------------------------------
    # 14. Validate string date conversion
    # ---------------------------------------------------------

    string_dates = purchase_events.copy(deep=True)

    # Convert datetime values into strings.
    string_dates["InvoiceDate"] = (
        pd.to_datetime(string_dates["InvoiceDate"])
        .dt.strftime("%Y-%m-%d")
    )

    # Accept pandas StringDtype and traditional object dtype.
    assert pd.api.types.is_string_dtype(
        string_dates["InvoiceDate"].dtype
    )

    string_result = create_lstm_purchase_timing_targets(
        string_dates
    )

    string_events = string_result["purchase_events_with_targets"]

    assert len(string_events) == 7

    assert pd.api.types.is_datetime64_any_dtype(
        string_events["InvoiceDate"]
    )

    assert pd.api.types.is_datetime64_any_dtype(
        string_events["Next_Purchase_Date"]
    )

    # ---------------------------------------------------------
    # 15. Validate invalid input type
    # ---------------------------------------------------------

    with pytest.raises(
        TypeError,
        match="purchase_events must be a pandas DataFrame",
    ):
        create_lstm_purchase_timing_targets(None)

    with pytest.raises(
        TypeError,
        match="purchase_events must be a pandas DataFrame",
    ):
        create_lstm_purchase_timing_targets([])

    # ---------------------------------------------------------
    # 16. Validate missing required columns
    # ---------------------------------------------------------

    for missing_column in [
        "Customer ID",
        "Invoice_ID",
        "InvoiceDate",
    ]:

        invalid_df = purchase_events.drop(
            columns=[missing_column]
        )

        with pytest.raises(
            ValueError,
            match="Missing required columns",
        ):
            create_lstm_purchase_timing_targets(invalid_df)

    # ---------------------------------------------------------
    # 17. Validate empty DataFrame
    # ---------------------------------------------------------

    empty_df = purchase_events.iloc[0:0].copy()

    with pytest.raises(
        ValueError,
        match="purchase_events DataFrame is empty",
    ):
        create_lstm_purchase_timing_targets(empty_df)

    # ---------------------------------------------------------
    # 18. Validate invalid date strings
    # ---------------------------------------------------------

    invalid_dates_df = purchase_events.copy(deep=True)

    invalid_dates_df.loc[
        invalid_dates_df.index[0],
        "InvoiceDate",
    ] = "invalid-date"

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Could not infer format",
            category=UserWarning,
        )

        with pytest.raises(
            ValueError,
            match="InvoiceDate validation failed",
        ):
            create_lstm_purchase_timing_targets(invalid_dates_df)

    # ---------------------------------------------------------
    # 19. Validate missing dates
    # ---------------------------------------------------------

    missing_date_df = purchase_events.copy(deep=True)

    missing_date_df.loc[
        missing_date_df.index[0],
        "InvoiceDate",
    ] = pd.NaT

    with pytest.raises(
        ValueError,
        match="InvoiceDate validation failed",
    ):
        create_lstm_purchase_timing_targets(missing_date_df)

    # ---------------------------------------------------------
    # 20. Validate missing Customer ID
    # ---------------------------------------------------------

    missing_customer_df = purchase_events.copy(deep=True)

    missing_customer_df.loc[
        missing_customer_df.index[0],
        "Customer ID",
    ] = None

    with pytest.raises(
        ValueError,
        match="Customer ID and Invoice_ID must not be missing",
    ):
        create_lstm_purchase_timing_targets(missing_customer_df)

    # ---------------------------------------------------------
    # 21. Validate missing Invoice_ID
    # ---------------------------------------------------------

    missing_invoice_df = purchase_events.copy(deep=True)

    missing_invoice_df.loc[
        missing_invoice_df.index[0],
        "Invoice_ID",
    ] = None

    with pytest.raises(
        ValueError,
        match="Customer ID and Invoice_ID must not be missing",
    ):
        create_lstm_purchase_timing_targets(missing_invoice_df)

    # ---------------------------------------------------------
    # 22. Validate unknown customer ID = 0
    # ---------------------------------------------------------

    zero_customer_df = purchase_events.copy(deep=True)

    zero_customer_df.loc[
        zero_customer_df.index[0],
        "Customer ID",
    ] = 0

    with pytest.raises(
        ValueError,
        match="Customer ID = 0 found",
    ):
        create_lstm_purchase_timing_targets(zero_customer_df)

    # ---------------------------------------------------------
    # 23. Validate duplicate customer-invoice events
    # ---------------------------------------------------------

    duplicate_df = pd.concat(
        [
            purchase_events,
            purchase_events.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate customer-invoice events",
    ):
        create_lstm_purchase_timing_targets(duplicate_df)

    # ---------------------------------------------------------
    # 24. Validate zero-day purchase target
    # ---------------------------------------------------------

    zero_day_df = pd.DataFrame({
        "Customer ID": [101, 101, 202],
        "Invoice_ID": ["Z001", "Z002", "Z003"],
        "InvoiceDate": [
            pd.Timestamp("2026-01-01"),
            pd.Timestamp("2026-01-01"),
            pd.Timestamp("2026-01-02"),
        ],
    })

    zero_day_result = create_lstm_purchase_timing_targets(
        zero_day_df
    )

    zero_day_events = zero_day_result[
        "purchase_events_with_targets"
    ]

    zero_day_target = zero_day_events.loc[
        zero_day_events["Invoice_ID"] == "Z001",
        "Days_Until_Next_Purchase",
    ].iloc[0]

    assert float(zero_day_target) == 0.0

    assert zero_day_result["target_audit"]["zero_day_targets"] == 1

    # ---------------------------------------------------------
    # 25. Validate original input remains unchanged
    # ---------------------------------------------------------

    pd.testing.assert_frame_equal(
        purchase_events,
        event_result["purchase_events"],
    )

def test_create_lstm_sequences():

    import numpy as np
    import pandas as pd
    import pytest
    import warnings

    # ---------------------------------------------------------
    # 1. Prepare sample modeling targets
    # ---------------------------------------------------------

    modeling_targets = pd.DataFrame(
        {
            "Customer ID": [101, 101, 101, 202, 202],
            "Invoice_ID": ["I001", "I002", "I003", "I004", "I005"],
            "InvoiceDate": pd.to_datetime([
                "2026-01-01",
                "2026-01-05",
                "2026-01-10",
                "2026-01-02",
                "2026-01-08",
            ]),
            "Next_Purchase_Date": pd.to_datetime([
                "2026-01-05",
                "2026-01-10",
                "2026-01-15",
                "2026-01-08",
                "2026-01-12",
            ]),
            "Next_Invoice_ID": [
                "I002", "I003", "I006", "I005", "I007"
            ],
            "Days_Until_Next_Purchase": [4, 5, 5, 6, 4],
            "Purchase_Value": [10.0, 8.0, 5.0, 20.0, 7.0],
            "Product_Line_Count": [1, 2, 1, 1, 2],
            "Unique_Product_Count": [1, 2, 1, 1, 2],
            "Total_Quantity": [1, 3, 1, 2, 4],
            "Category_A_Line_Count": [1, 2, 1, 0, 2],
            "Category_A_Value": [10.0, 8.0, 5.0, 0.0, 7.0],
        }
    )

    feature_columns = [
        "Purchase_Value",
        "Product_Line_Count",
        "Unique_Product_Count",
        "Total_Quantity",
        "Category_A_Line_Count",
        "Category_A_Value",
    ]

    input_data = {
        "modeling_targets": modeling_targets.copy(deep=True)
    }

    original_input = input_data["modeling_targets"].copy(deep=True)

    # ---------------------------------------------------------
    # 2. Execute function
    # ---------------------------------------------------------

    result = create_lstm_sequences(
        input_data,
        sequence_length=3,
        feature_columns=feature_columns,
    )

    # ---------------------------------------------------------
    # 3. Validate return structure
    # ---------------------------------------------------------

    assert isinstance(result, dict)

    expected_keys = {
        "X",
        "X_mask",
        "y",
        "sequence_metadata",
        "feature_columns",
        "sequence_audit",
    }

    assert set(result.keys()) == expected_keys

    X = result["X"]
    X_mask = result["X_mask"]
    y = result["y"]
    metadata = result["sequence_metadata"]
    audit = result["sequence_audit"]

    assert isinstance(X, np.ndarray)
    assert isinstance(X_mask, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(metadata, pd.DataFrame)
    assert isinstance(audit, dict)

    # ---------------------------------------------------------
    # 4. Validate array dimensions and data types
    # ---------------------------------------------------------

    assert X.shape == (5, 3, 6)
    assert X_mask.shape == (5, 3)
    assert y.shape == (5,)

    assert X.dtype == np.float64
    assert y.dtype == np.float64
    assert X_mask.dtype == bool

    assert X.shape[0] == len(modeling_targets)
    assert X.shape[1] == 3
    assert X.shape[2] == len(feature_columns)

    # ---------------------------------------------------------
    # 5. Validate feature columns and metadata columns
    # ---------------------------------------------------------

    assert result["feature_columns"] == feature_columns

    assert list(metadata.columns) == [
        "Customer ID",
        "Invoice_ID",
        "InvoiceDate",
        "Next_Purchase_Date",
        "Days_Until_Next_Purchase",
        "History_Length",
        "Padding_Length",
    ]

    # ---------------------------------------------------------
    # 6. Validate target values
    # ---------------------------------------------------------

    expected_targets = np.array([4.0, 5.0, 5.0, 6.0, 4.0])

    np.testing.assert_array_equal(y, expected_targets)

    assert np.all(y >= 0)

    # ---------------------------------------------------------
    # 7. Validate customer and invoice ordering
    # ---------------------------------------------------------

    assert metadata["Customer ID"].tolist() == [
        101, 101, 101, 202, 202
    ]

    assert metadata["Invoice_ID"].tolist() == [
        "I001", "I002", "I003", "I004", "I005"
    ]

    expected_dates = pd.to_datetime([
        "2026-01-01",
        "2026-01-05",
        "2026-01-10",
        "2026-01-02",
        "2026-01-08",
    ])

    pd.testing.assert_series_equal(
        metadata["InvoiceDate"].reset_index(drop=True),
        pd.Series(expected_dates, name="InvoiceDate"),
        check_dtype=False,
    )

    # ---------------------------------------------------------
    # 8. Validate left-padding masks
    # ---------------------------------------------------------

    expected_mask = np.array(
        [
            [False, False, True],
            [False, True, True],
            [True, True, True],
            [False, False, True],
            [False, True, True],
        ],
        dtype=bool,
    )

    np.testing.assert_array_equal(X_mask, expected_mask)

    # ---------------------------------------------------------
    # 9. Validate history and padding lengths
    # ---------------------------------------------------------

    assert metadata["History_Length"].tolist() == [1, 2, 3, 1, 2]
    assert metadata["Padding_Length"].tolist() == [2, 1, 0, 2, 1]

    assert (
        metadata["History_Length"] + metadata["Padding_Length"]
    ).eq(3).all()

    # ---------------------------------------------------------
    # 10. Validate sequence feature values
    # ---------------------------------------------------------

    expected_sequences = np.array(
        [
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [10, 1, 1, 1, 1, 10],
            ],
            [
                [0, 0, 0, 0, 0, 0],
                [10, 1, 1, 1, 1, 10],
                [8, 2, 2, 3, 2, 8],
            ],
            [
                [10, 1, 1, 1, 1, 10],
                [8, 2, 2, 3, 2, 8],
                [5, 1, 1, 1, 1, 5],
            ],
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [20, 1, 1, 2, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0],
                [20, 1, 1, 2, 0, 0],
                [7, 2, 2, 4, 2, 7],
            ],
        ],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(X, expected_sequences)

    # ---------------------------------------------------------
    # 11. Validate audit structure
    # ---------------------------------------------------------

    expected_audit_keys = {
        "input_target_rows",
        "sequence_count",
        "sequence_length",
        "feature_count",
        "unique_customers",
        "X_shape",
        "X_mask_shape",
        "X_mask_dtype",
        "y_shape",
        "short_history_sequences",
        "sequences_with_padding",
        "sequences_without_padding",
        "missing_values_in_X",
        "missing_values_in_y",
        "padding_values_are_zero",
        "mask_validation_passed",
        "target_min_days",
        "target_median_days",
        "target_max_days",
        "feature_columns",
    }

    assert set(audit.keys()) == expected_audit_keys

    # ---------------------------------------------------------
    # 12. Validate audit counts
    # ---------------------------------------------------------

    assert audit["input_target_rows"] == 5
    assert audit["sequence_count"] == 5
    assert audit["sequence_length"] == 3
    assert audit["feature_count"] == 6
    assert audit["unique_customers"] == 2

    assert audit["short_history_sequences"] == 4
    assert audit["sequences_with_padding"] == 4
    assert audit["sequences_without_padding"] == 1

    assert audit["missing_values_in_X"] == 0
    assert audit["missing_values_in_y"] == 0

    # ---------------------------------------------------------
    # 13. Validate audit shapes and data types
    # ---------------------------------------------------------

    assert audit["X_shape"] == (5, 3, 6)
    assert audit["X_mask_shape"] == (5, 3)
    assert audit["X_mask_dtype"] == "bool"
    assert audit["y_shape"] == (5,)

    assert bool(audit["padding_values_are_zero"]) is True
    assert bool(audit["mask_validation_passed"]) is True

    # ---------------------------------------------------------
    # 14. Validate target statistics
    # ---------------------------------------------------------

    assert round(float(audit["target_min_days"]), 10) == 4.0
    assert round(float(audit["target_median_days"]), 10) == 5.0
    assert round(float(audit["target_max_days"]), 10) == 6.0

    assert audit["feature_columns"] == feature_columns

    # ---------------------------------------------------------
    # 15. Validate numerical integrity and zero padding
    # ---------------------------------------------------------

    assert np.all(X[~X_mask] == 0)
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()
    assert np.all(y >= 0)

    # ---------------------------------------------------------
    # 16. Validate input DataFrame remains unchanged
    # ---------------------------------------------------------

    pd.testing.assert_frame_equal(
        input_data["modeling_targets"],
        original_input,
    )

    # ---------------------------------------------------------
    # 17. Validate invalid input types
    # ---------------------------------------------------------

    with pytest.raises(
        TypeError,
        match="lstm_target_result must be a dictionary",
    ):
        create_lstm_sequences(None)

    with pytest.raises(
        TypeError,
        match="lstm_target_result must be a dictionary",
    ):
        create_lstm_sequences([])

    # ---------------------------------------------------------
    # 18. Validate missing dictionary key
    # ---------------------------------------------------------

    with pytest.raises(
        ValueError,
        match="lstm_target_result must contain",
    ):
        create_lstm_sequences({})

    # ---------------------------------------------------------
    # 19. Validate incorrect modeling_targets type
    # ---------------------------------------------------------

    with pytest.raises(
        TypeError,
        match=r"'modeling_targets' must be a pandas DataFrame",
    ):
        create_lstm_sequences(
            {"modeling_targets": ["invalid"]}
        )

    # ---------------------------------------------------------
    # 20. Validate empty modeling_targets DataFrame
    # ---------------------------------------------------------

    with pytest.raises(
        ValueError,
        match=r"'modeling_targets' is empty",
    ):
        create_lstm_sequences(
            {"modeling_targets": pd.DataFrame()}
        )

    # ---------------------------------------------------------
    # 21. Validate sequence_length
    # ---------------------------------------------------------

    for invalid_length in [0, -1, 1.5, "3", True, None]:

        with pytest.raises(
            ValueError,
            match="sequence_length must be a positive integer",
        ):
            create_lstm_sequences(
                input_data,
                sequence_length=invalid_length,
                feature_columns=feature_columns,
            )

    # ---------------------------------------------------------
    # 22. Validate missing required columns
    # ---------------------------------------------------------

    missing_required = modeling_targets.drop(
        columns=["Purchase_Value"]
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        create_lstm_sequences(
            {"modeling_targets": missing_required}
        )

    # ---------------------------------------------------------
    # 23. Validate invalid InvoiceDate
    # ---------------------------------------------------------

    invalid_dates = modeling_targets.copy(deep=True)
    invalid_dates["InvoiceDate"] = invalid_dates[
        "InvoiceDate"
    ].astype("object")
    invalid_dates.loc[0, "InvoiceDate"] = "invalid-date"

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Could not infer format",
            category=UserWarning,
        )

        with pytest.raises(
            ValueError,
            match="InvoiceDate contains missing or invalid dates",
        ):
            create_lstm_sequences(
                {"modeling_targets": invalid_dates}
            )

    # ---------------------------------------------------------
    # 24. Validate missing Customer ID
    # ---------------------------------------------------------

    missing_customer = modeling_targets.copy(deep=True)
    missing_customer.loc[0, "Customer ID"] = np.nan

    with pytest.raises(
        ValueError,
        match="Customer ID contains missing values",
    ):
        create_lstm_sequences(
            {"modeling_targets": missing_customer}
        )

    # ---------------------------------------------------------
    # 25. Validate excluded Customer ID 0
    # ---------------------------------------------------------

    zero_customer = modeling_targets.copy(deep=True)
    zero_customer.loc[0, "Customer ID"] = 0

    with pytest.raises(
        ValueError,
        match="Customer ID 0 must be excluded",
    ):
        create_lstm_sequences(
            {"modeling_targets": zero_customer}
        )

    # ---------------------------------------------------------
    # 26. Validate missing Invoice_ID
    # ---------------------------------------------------------

    missing_invoice = modeling_targets.copy(deep=True)
    missing_invoice.loc[0, "Invoice_ID"] = np.nan

    with pytest.raises(
        ValueError,
        match="Invoice_ID contains missing values",
    ):
        create_lstm_sequences(
            {"modeling_targets": missing_invoice}
        )

    # ---------------------------------------------------------
    # 27. Validate duplicate customer-invoice events
    # ---------------------------------------------------------

    duplicate_events = pd.concat(
        [
            modeling_targets,
            modeling_targets.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate customer-invoice events were found",
    ):
        create_lstm_sequences(
            {"modeling_targets": duplicate_events}
        )

    # ---------------------------------------------------------
    # 28. Validate missing prediction targets
    # ---------------------------------------------------------

    missing_target = modeling_targets.copy(deep=True)
    missing_target.loc[
        0, "Days_Until_Next_Purchase"
    ] = np.nan

    with pytest.raises(
        ValueError,
        match="Days_Until_Next_Purchase contains missing or non-numeric values",
    ):
        create_lstm_sequences(
            {"modeling_targets": missing_target}
        )

    # ---------------------------------------------------------
    # 29. Validate non-numeric prediction targets
    # ---------------------------------------------------------

    invalid_target = modeling_targets.copy(deep=True)
    invalid_target["Days_Until_Next_Purchase"] = (
        invalid_target["Days_Until_Next_Purchase"].astype(object)
    )
    invalid_target.loc[
        0, "Days_Until_Next_Purchase"
    ] = "invalid"

    with pytest.raises(
        ValueError,
        match="Days_Until_Next_Purchase contains missing or non-numeric values",
    ):
        create_lstm_sequences(
            {"modeling_targets": invalid_target}
        )

    # ---------------------------------------------------------
    # 30. Validate infinite prediction targets
    # ---------------------------------------------------------

    infinite_target = modeling_targets.copy(deep=True)

    # Convert int64 to float before assigning infinity.
    infinite_target["Days_Until_Next_Purchase"] = (
        infinite_target["Days_Until_Next_Purchase"].astype(float)
    )

    infinite_target.loc[
        0, "Days_Until_Next_Purchase"
    ] = np.inf

    with pytest.raises(
        ValueError,
        match="Prediction targets contain infinite values",
    ):
        create_lstm_sequences(
            {"modeling_targets": infinite_target}
        )

    # ---------------------------------------------------------
    # 31. Validate negative prediction targets
    # ---------------------------------------------------------

    negative_target = modeling_targets.copy(deep=True)
    negative_target.loc[
        0, "Days_Until_Next_Purchase"
    ] = -1

    with pytest.raises(
        ValueError,
        match="Prediction targets cannot be negative",
    ):
        create_lstm_sequences(
            {"modeling_targets": negative_target}
        )

    # ---------------------------------------------------------
    # 32. Validate empty feature selection
    # ---------------------------------------------------------

    with pytest.raises(
        ValueError,
        match="No feature columns were selected",
    ):
        create_lstm_sequences(
            input_data,
            sequence_length=3,
            feature_columns=[],
        )

    # ---------------------------------------------------------
    # 33. Validate missing feature columns
    # ---------------------------------------------------------

    with pytest.raises(
        ValueError,
        match="Missing feature columns",
    ):
        create_lstm_sequences(
            input_data,
            sequence_length=3,
            feature_columns=[
                "Purchase_Value",
                "Nonexistent_Feature",
            ],
        )

    # ---------------------------------------------------------
    # 34. Validate forbidden identifier/date/target features
    # ---------------------------------------------------------

    forbidden_features = [
        "Customer ID",
        "Invoice_ID",
        "InvoiceDate",
        "Next_Purchase_Date",
        "Next_Invoice_ID",
        "Days_Until_Next_Purchase",
    ]

    for forbidden_feature in forbidden_features:

        with pytest.raises(
            ValueError,
            match="Identifiers, dates, and targets cannot be features",
        ):
            create_lstm_sequences(
                input_data,
                sequence_length=3,
                feature_columns=[forbidden_feature],
            )

    # ---------------------------------------------------------
    # 35. Validate missing feature values
    # ---------------------------------------------------------

    missing_feature = modeling_targets.copy(deep=True)
    missing_feature.loc[0, "Purchase_Value"] = np.nan

    with pytest.raises(
        ValueError,
        match="Feature columns contain missing or infinite values",
    ):
        create_lstm_sequences(
            {"modeling_targets": missing_feature},
            sequence_length=3,
            feature_columns=feature_columns,
        )

    # ---------------------------------------------------------
    # 36. Validate infinite feature values
    # ---------------------------------------------------------

    infinite_feature = modeling_targets.copy(deep=True)

    # Purchase_Value is already float, so infinity can be assigned.
    infinite_feature.loc[0, "Purchase_Value"] = np.inf

    with pytest.raises(
        ValueError,
        match="Feature columns contain missing or infinite values",
    ):
        create_lstm_sequences(
            {"modeling_targets": infinite_feature},
            sequence_length=3,
            feature_columns=feature_columns,
        )

    # ---------------------------------------------------------
    # 37. Validate automatic feature selection
    # ---------------------------------------------------------

    automatic_result = create_lstm_sequences(
        input_data,
        sequence_length=3,
    )

    expected_automatic_features = [
        "Purchase_Value",
        "Product_Line_Count",
        "Unique_Product_Count",
        "Total_Quantity",
        "Category_A_Line_Count",
        "Category_A_Value",
    ]

    assert (
        automatic_result["feature_columns"]
        == expected_automatic_features
    )

    assert automatic_result["X"].shape == (5, 3, 6)

    # ---------------------------------------------------------
    # 38. Validate shorter sequence length
    # ---------------------------------------------------------

    shorter_result = create_lstm_sequences(
        input_data,
        sequence_length=2,
        feature_columns=feature_columns,
    )

    assert shorter_result["X"].shape == (5, 2, 6)
    assert shorter_result["X_mask"].shape == (5, 2)
    assert shorter_result["y"].shape == (5,)

    expected_shorter_mask = np.array(
        [
            [False, True],
            [True, True],
            [True, True],
            [False, True],
            [True, True],
        ],
        dtype=bool,
    )

    np.testing.assert_array_equal(
        shorter_result["X_mask"],
        expected_shorter_mask,
    )

    assert shorter_result["sequence_audit"]["sequence_length"] == 2
    assert shorter_result["sequence_audit"]["sequences_with_padding"] == 2
    assert shorter_result["sequence_audit"]["sequences_without_padding"] == 3



#########################################################################################################
#----------------------------- Auto encoder -------------------------
#########################################################################################################

def test_select_autoencoder_features():

    import pandas as pd
    import pytest

    # ---------------------------------------------------------
    # 1. Prepare sample customer features
    # ---------------------------------------------------------

    customer_features = pd.DataFrame(
        {
            "Customer ID": [101, 102, 103, 104],
            "Recency": [5, 10, 15, 20],
            "Frequency": [12, 8, 5, 2],
            "Monetary": [1500.50, 900.00, 450.75, 120.00],
            "Engagement Score": [85.5, 70.0, 55.5, 30.0],
            "Customer Segment": [
                "High Value",
                "Regular",
                "Occasional",
                "At Risk",
            ],
            "Country": ["India", "UK", "USA", "India"],
            "Is Active": [True, False, True, False],
        }
    )

    excluded_columns = [
        "Customer ID",
        "Recency",
    ]

    original_data = customer_features.copy(deep=True)

    # ---------------------------------------------------------
    # 2. Execute production function
    # ---------------------------------------------------------

    customer_ids, X = select_autoencoder_features(
        customer_features,
        excluded_columns,
    )

    # ---------------------------------------------------------
    # 3. Validate returned object types
    # ---------------------------------------------------------

    assert isinstance(customer_ids, pd.Series)
    assert isinstance(X, pd.DataFrame)

    # ---------------------------------------------------------
    # 4. Validate customer IDs
    # ---------------------------------------------------------

    expected_customer_ids = pd.Series(
        [101, 102, 103, 104],
        name="Customer ID",
    )

    pd.testing.assert_series_equal(
        customer_ids.reset_index(drop=True),
        expected_customer_ids,
    )

    assert len(customer_ids) == len(customer_features)

    # ---------------------------------------------------------
    # 5. Validate selected numeric feature columns
    # ---------------------------------------------------------

    expected_columns = [
        "Frequency",
        "Monetary",
        "Engagement Score",
    ]

    assert X.columns.tolist() == expected_columns
    assert X.shape == (4, 3)
    assert X.index.equals(customer_features.index)

    # ---------------------------------------------------------
    # 6. Validate selected feature values
    # ---------------------------------------------------------

    expected_X = pd.DataFrame(
        {
            "Frequency": [12, 8, 5, 2],
            "Monetary": [1500.50, 900.00, 450.75, 120.00],
            "Engagement Score": [85.5, 70.0, 55.5, 30.0],
        },
        index=customer_features.index,
    )

    pd.testing.assert_frame_equal(
        X,
        expected_X,
    )

    # ---------------------------------------------------------
    # 7. Validate numeric data types
    # ---------------------------------------------------------

    assert all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in X.dtypes
    )

    # ---------------------------------------------------------
    # 8. Validate explicitly excluded columns
    # ---------------------------------------------------------

    assert "Customer ID" not in X.columns
    assert "Recency" not in X.columns

    # ---------------------------------------------------------
    # 9. Validate non-numeric columns are excluded
    # ---------------------------------------------------------

    assert "Customer Segment" not in X.columns
    assert "Country" not in X.columns

    # Boolean columns are not selected by include="number".
    assert "Is Active" not in X.columns

    # ---------------------------------------------------------
    # 10. Validate customer IDs are preserved separately
    # ---------------------------------------------------------

    assert customer_ids.name == "Customer ID"
    assert customer_ids.tolist() == [101, 102, 103, 104]

    assert "Customer ID" not in X.columns

    # ---------------------------------------------------------
    # 11. Validate original DataFrame remains unchanged
    # ---------------------------------------------------------

    pd.testing.assert_frame_equal(
        customer_features,
        original_data,
    )

    # ---------------------------------------------------------
    # 12. Validate returned customer IDs are independent copies
    # ---------------------------------------------------------

    customer_ids.iloc[0] = 999

    assert customer_features.loc[0, "Customer ID"] == 101

    # ---------------------------------------------------------
    # 13. Validate returned features are independent copies
    # ---------------------------------------------------------

    X.iloc[0, 0] = 999

    assert customer_features.loc[0, "Frequency"] == 12

    # ---------------------------------------------------------
    # 14. Validate nonexistent excluded columns
    # ---------------------------------------------------------

    result_ids, result_X = select_autoencoder_features(
        original_data,
        [
            "Customer ID",
            "Nonexistent Column",
        ],
    )

    assert result_X.columns.tolist() == [
        "Recency",
        "Frequency",
        "Monetary",
        "Engagement Score",
    ]

    assert result_X.shape == (4, 4)

    assert result_ids.tolist() == [101, 102, 103, 104]

    # ---------------------------------------------------------
    # 15. Validate additional caller-specified exclusions
    # ---------------------------------------------------------

    _, additional_exclusion_X = select_autoencoder_features(
        original_data,
        [
            "Customer ID",
            "Recency",
            "Frequency",
            "Monetary",
        ],
    )

    assert additional_exclusion_X.columns.tolist() == [
        "Engagement Score",
    ]

    assert additional_exclusion_X.shape == (4, 1)

    # ---------------------------------------------------------
    # 16. Validate missing Customer ID column
    # ---------------------------------------------------------

    missing_customer_id = original_data.drop(
        columns=["Customer ID"]
    )

    with pytest.raises(
        ValueError,
        match="Customer ID column is missing",
    ):
        select_autoencoder_features(
            missing_customer_id,
            excluded_columns,
        )

    # ---------------------------------------------------------
    # 17. Validate no numeric features available
    # ---------------------------------------------------------

    non_numeric_data = pd.DataFrame(
        {
            "Customer ID": [101, 102, 103],
            "Customer Segment": [
                "High Value",
                "Regular",
                "At Risk",
            ],
            "Country": ["India", "UK", "USA"],
        }
    )

    with pytest.raises(
        ValueError,
        match="No numeric features available for the autoencoder",
    ):
        select_autoencoder_features(
            non_numeric_data,
            ["Customer ID"],
        )

    # ---------------------------------------------------------
    # 18. Validate all numeric features are excluded
    # ---------------------------------------------------------

    all_numeric_excluded = pd.DataFrame(
        {
            "Customer ID": [101, 102, 103],
            "Frequency": [5, 10, 15],
            "Monetary": [100.0, 200.0, 300.0],
            "Customer Segment": [
                "Regular",
                "High Value",
                "At Risk",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="No numeric features available for the autoencoder",
    ):
        select_autoencoder_features(
            all_numeric_excluded,
            [
                "Customer ID",
                "Frequency",
                "Monetary",
            ],
        )

    # ---------------------------------------------------------
    # 19. Validate empty DataFrame with Customer ID column
    # ---------------------------------------------------------

    empty_customer_data = pd.DataFrame(
        columns=[
            "Customer ID",
            "Frequency",
            "Monetary",
        ]
    )

    with pytest.raises(
        ValueError,
        match="No numeric features available for the autoencoder",
    ):
        select_autoencoder_features(
            empty_customer_data,
            ["Customer ID"],
        )

    # ---------------------------------------------------------
    # 20. Validate empty excluded_columns list
    # ---------------------------------------------------------

    _, no_exclusions_X = select_autoencoder_features(
        original_data,
        [],
    )

    # Customer ID is retained because it is numeric
    # and was not explicitly excluded.

    assert no_exclusions_X.columns.tolist() == [
        "Customer ID",
        "Recency",
        "Frequency",
        "Monetary",
        "Engagement Score",
    ]

    assert no_exclusions_X.shape == (4, 5)

    assert "Customer ID" in no_exclusions_X.columns

    # ---------------------------------------------------------
    # 21. Validate integer and floating-point feature types
    # ---------------------------------------------------------

    assert pd.api.types.is_integer_dtype(
        no_exclusions_X["Customer ID"]
    )

    assert pd.api.types.is_integer_dtype(
        no_exclusions_X["Recency"]
    )

    assert pd.api.types.is_integer_dtype(
        no_exclusions_X["Frequency"]
    )

    assert pd.api.types.is_float_dtype(
        no_exclusions_X["Monetary"]
    )

    assert pd.api.types.is_float_dtype(
        no_exclusions_X["Engagement Score"]
    )

    # ---------------------------------------------------------
    # 22. Validate numeric feature count
    # ---------------------------------------------------------

    assert len(no_exclusions_X.select_dtypes(include="number").columns) == 5

    # ---------------------------------------------------------
    # 23. Validate input remains unchanged after all executions
    # ---------------------------------------------------------

    pd.testing.assert_frame_equal(
        original_data,
        customer_features,
    )
