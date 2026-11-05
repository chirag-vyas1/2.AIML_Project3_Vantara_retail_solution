import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

################################################################################
#--- Display function for dicumentation ----
################################################################
def display_feature_validation(
    validation_result: dict,
    feature_name: str,
) -> None:
    """
    Display feature validation results in a clean,
    notebook-friendly format.

    Parameters
    ----------
    validation_result : dict
        Dictionary returned by a feature validation function.

    feature_name : str
        Name of the feature being validated.
    """

    validation = validation_result.get(
        "validation",
        {},
    )

    summary = validation_result.get(
        "summary",
        {},
    )

    failed_checks = validation_result.get(
        "failed_checks",
        [],
    )

    print("=" * 75)
    print(f"{feature_name.upper()} — VALIDATION REPORT")
    print("=" * 75)

    # ------------------------------------------------------------
    # Validation checks
    # ------------------------------------------------------------

    print("\nVALIDATION CHECKS")
    print("-" * 75)

    for check_name, result in validation.items():

        if check_name == "passed":
            continue

        status = "PASS" if bool(result) else "FAIL"

        display_name = check_name.replace(
            "_",
            " ",
        ).title()

        print(
            f"{status:<8} {display_name}"
        )

    # ------------------------------------------------------------
    # Overall result
    # ------------------------------------------------------------

    print("\nOVERALL RESULT")
    print("-" * 75)

    passed = validation.get(
        "passed",
        False,
    )

    if passed:
        print("✓ VALIDATION PASSED")
    else:
        print("✗ VALIDATION FAILED")

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    print("\nSUMMARY")
    print("-" * 75)

    for key, value in summary.items():

        display_name = key.replace(
            "_",
            " ",
        ).title()

        print(
            f"{display_name:<40}: {value}"
        )

    # ------------------------------------------------------------
    # Failed checks
    # ------------------------------------------------------------

    print("\nFAILED CHECKS")
    print("-" * 75)

    if failed_checks:

        for check in failed_checks:
            print(f"✗ {check}")

    else:

        print("None")

    print("=" * 75)



########################################################################
#  1. Resency Frequency Monetary feature
########################################################################

#------------ 1A. Analyze RFM basis ----------------
def analyze_rfm_basis(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> dict:
    """
    Establish the analytical basis for RFM feature engineering.

    This exploratory analysis documents:
        - qualifying historical transactions
        - number of customers
        - distinct orders
        - customer purchase frequency
        - total monetary value
        - average order spend
        - latest purchase date
        - recency

    This function is intended for notebook analysis and
    documentation before the finalized RFM feature function
    is applied.
    """

    working_df = df.copy()

    working_df["InvoiceDate"] = pd.to_datetime(
        working_df["InvoiceDate"],
        errors="raise",
    )

    # ---------------------------------------------------------
    # 1. Select qualifying historical purchase transactions
    # ---------------------------------------------------------

    rfm_transactions = working_df[
        (working_df["InvoiceDate"] <= cutoff_date)
        & (working_df["Transaction Type"] == "Purchase")
        & (working_df["Customer ID Missing Flag"] == 0)
        & (working_df["Include In Sales Analysis"] == True)
    ].copy()

    # ---------------------------------------------------------
    # 2. Customer-level aggregation
    # ---------------------------------------------------------

    customer_analysis = (
        rfm_transactions
        .groupby("Customer ID")
        .agg(
            Last_Purchase_Date=("InvoiceDate", "max"),
            Frequency=("Invoice", "nunique"),
            Total_Monetary_Value=(
                "Analytical Transaction Value",
                "sum",
            ),
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # 3. Calculate Recency
    # ---------------------------------------------------------

    customer_analysis["Recency"] = (
        cutoff_date
        - customer_analysis["Last_Purchase_Date"]
    ).dt.total_seconds() / (24 * 60 * 60)

    # ---------------------------------------------------------
    # 4. Calculate Average Order Spend
    # ---------------------------------------------------------

    customer_analysis["Average_Order_Spend"] = (
        customer_analysis["Total_Monetary_Value"]
        / customer_analysis["Frequency"]
    )

    # ---------------------------------------------------------
    # 5. Round analytical values
    # ---------------------------------------------------------

    customer_analysis["Recency"] = (
        customer_analysis["Recency"].round(2)
    )

    customer_analysis["Total_Monetary_Value"] = (
        customer_analysis["Total_Monetary_Value"].round(2)
    )

    customer_analysis["Average_Order_Spend"] = (
        customer_analysis["Average_Order_Spend"].round(2)
    )

    # ---------------------------------------------------------
    # 6. Summary information
    # ---------------------------------------------------------

    analysis_summary = {
        "cutoff_date": cutoff_date,
        "historical_transaction_count": len(rfm_transactions),
        "customer_count": customer_analysis["Customer ID"].nunique(),
        "distinct_invoice_count": rfm_transactions["Invoice"].nunique(),
        "total_monetary_value": round(
            rfm_transactions["Analytical Transaction Value"].sum(),
            2,
        ),
        "average_order_spend": round(
            rfm_transactions["Analytical Transaction Value"].sum()
            / rfm_transactions["Invoice"].nunique(),
            2,
        ),
        "minimum_recency": customer_analysis["Recency"].min(),
        "maximum_recency": customer_analysis["Recency"].max(),
        "median_recency": customer_analysis["Recency"].median(),
        "minimum_frequency": customer_analysis["Frequency"].min(),
        "maximum_frequency": customer_analysis["Frequency"].max(),
    }

    return {
        "summary": analysis_summary,
        "customer_analysis": customer_analysis,
        "rfm_transactions": rfm_transactions,
    }

# ----------- 1B. Create RMF feature ------------
def create_rfm_features(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Create customer-level RFM features.

    Features
    --------
    Recency:
        Number of days since the customer's latest qualifying purchase.

    Frequency:
        Number of distinct orders/invoices.

    Total Monetary Value:
        Total qualifying purchase spend.

    Average Order Spend:
        Total Monetary Value divided by the number of distinct orders.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction-level dataframe.

    cutoff_date : pd.Timestamp or None, default=None
        Point-in-time cutoff used to prevent future-data leakage.

        If None, the cutoff is automatically calculated as
        90 days before the maximum InvoiceDate in the dataset.

        If supplied, the provided cutoff date is used.

    Returns
    -------
    pd.DataFrame
        Customer-level RFM dataframe with one row per customer.
    """

    # ---------------------------------------------------------
    # 1. Validate required columns
    # ---------------------------------------------------------

    required_columns = {
        "Customer ID",
        "Invoice",
        "InvoiceDate",
        "Transaction Type",
        "Customer ID Missing Flag",
        "Include In Sales Analysis",
        "Analytical Transaction Value",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # ---------------------------------------------------------
    # 2. Work on a copy so the input dataframe is not modified
    # ---------------------------------------------------------

    working_df = df.copy()

    # ---------------------------------------------------------
    # 3. Ensure InvoiceDate is datetime
    # ---------------------------------------------------------

    working_df["InvoiceDate"] = pd.to_datetime(
        working_df["InvoiceDate"],
        errors="raise",
    )

    # ---------------------------------------------------------
    # 4. Determine point-in-time cutoff
    # ---------------------------------------------------------

    max_date = working_df["InvoiceDate"].max()

    if pd.isna(max_date):
        raise ValueError(
            "InvoiceDate contains no valid dates."
        )

    if cutoff_date is None:

        cutoff_date = (
            max_date - pd.Timedelta(days=90)
        )

    else:

        cutoff_date = pd.Timestamp(cutoff_date)

    # ---------------------------------------------------------
    # 5. Select qualifying historical purchases
    # ---------------------------------------------------------

    rfm_transactions = working_df[
        (working_df["InvoiceDate"] <= cutoff_date)
        & (working_df["Transaction Type"] == "Purchase")
        & (working_df["Customer ID Missing Flag"] == 0)
        & (working_df["Include In Sales Analysis"] == True)
    ].copy()

    if rfm_transactions.empty:
        raise ValueError(
            "No qualifying purchase transactions found "
            "before or on the supplied cutoff date."
        )

    # ---------------------------------------------------------
    # 6. Recency
    # ---------------------------------------------------------

    last_purchase = (
        rfm_transactions
        .groupby("Customer ID")["InvoiceDate"]
        .max()
    )

    recency = (
        cutoff_date - last_purchase
    ).dt.total_seconds() / (24 * 60 * 60)

    # ---------------------------------------------------------
    # 7. Frequency
    # ---------------------------------------------------------

    frequency = (
        rfm_transactions
        .groupby("Customer ID")["Invoice"]
        .nunique()
        .rename("Frequency")
    )

    # ---------------------------------------------------------
    # 8. Total Monetary Value
    # ---------------------------------------------------------

    total_monetary_value = (
        rfm_transactions
        .groupby("Customer ID")["Analytical Transaction Value"]
        .sum()
        .rename("Total_Monetary_Value")
    )

    # ---------------------------------------------------------
    # 9. Average Order Spend
    # ---------------------------------------------------------

    average_order_spend = (
        total_monetary_value / frequency
    ).rename("Average_Order_Spend")

    # ---------------------------------------------------------
    # 10. Combine customer-level RFM features
    # ---------------------------------------------------------

    rfm = pd.concat(
        [
            recency.rename("Recency"),
            frequency,
            total_monetary_value,
            average_order_spend,
        ],
        axis=1,
    ).reset_index()

    # ---------------------------------------------------------
    # 11. Round final feature values
    # ---------------------------------------------------------

    rfm["Recency"] = rfm["Recency"].round(2)

    rfm["Total_Monetary_Value"] = (
        rfm["Total_Monetary_Value"].round(2)
    )

    rfm["Average_Order_Spend"] = (
        rfm["Average_Order_Spend"].round(2)
    )

    # ---------------------------------------------------------
    # 12. Final validation
    # ---------------------------------------------------------

    if rfm["Customer ID"].duplicated().any():
        raise ValueError(
            "RFM output contains duplicate Customer IDs."
        )

    if rfm["Recency"].isna().any():
        raise ValueError(
            "RFM output contains missing Recency values."
        )

    if (rfm["Recency"] < 0).any():
        raise ValueError(
            "RFM output contains negative Recency values."
        )

    if rfm["Frequency"].isna().any():
        raise ValueError(
            "RFM output contains missing Frequency values."
        )

    if (rfm["Frequency"] <= 0).any():
        raise ValueError(
            "RFM output contains zero or negative Frequency values."
        )

    if rfm["Total_Monetary_Value"].isna().any():
        raise ValueError(
            "RFM output contains missing Total Monetary Value."
        )

    if (rfm["Total_Monetary_Value"] <= 0).any():
        raise ValueError(
            "RFM output contains zero or negative Total Monetary Value."
        )

    if rfm["Average_Order_Spend"].isna().any():
        raise ValueError(
            "RFM output contains missing Average Order Spend."
        )

    if (rfm["Average_Order_Spend"] <= 0).any():
        raise ValueError(
            "RFM output contains zero or negative Average Order Spend."
        )

    return rfm

# ----------- 1C. Vadlidate RMF feature  ----------------
def validate_rfm_features(
    rfm_features: pd.DataFrame,
) -> dict:
    """
    Validate customer-level RFM features.

    Validation Rules
    ----------------
    1. Required RFM columns must exist.
    2. Each Customer ID must be unique.
    3. Customer ID must not be missing.
    4. Recency must not be missing or negative.
    5. Frequency must not be missing, zero, or negative.
    6. Total Monetary Value must not be missing or non-positive.
    7. Average Order Spend must not be missing or non-positive.
    8. Average Order Spend must equal:
           Total Monetary Value / Frequency
       before final rounding.
    9. The output must contain at least one customer.

    Parameters
    ----------
    rfm_features : pd.DataFrame
        Customer-level RFM feature dataframe.

    Returns
    -------
    dict
        Validation results containing individual checks and
        an overall pass/fail status.
    """

    required_columns = [
        "Customer ID",
        "Recency",
        "Frequency",
        "Total_Monetary_Value",
        "Average_Order_Spend",
    ]

    validation_results = {}

    # ---------------------------------------------------------
    # 1. Required columns
    # ---------------------------------------------------------

    missing_columns = [
        column
        for column in required_columns
        if column not in rfm_features.columns
    ]

    validation_results["required_columns_present"] = (
        len(missing_columns) == 0
    )

    validation_results["missing_columns"] = missing_columns

    if missing_columns:
        validation_results["passed"] = False
        return validation_results

    # ---------------------------------------------------------
    # 2. Dataset contains customers
    # ---------------------------------------------------------

    validation_results["customer_count"] = len(rfm_features)

    validation_results["has_customers"] = (
        len(rfm_features) > 0
    )

    # ---------------------------------------------------------
    # 3. Customer ID validation
    # ---------------------------------------------------------

    validation_results["missing_customer_ids"] = int(
        rfm_features["Customer ID"].isna().sum()
    )

    validation_results["duplicate_customer_ids"] = int(
        rfm_features["Customer ID"].duplicated().sum()
    )

    validation_results["unique_customer_ids"] = (
        rfm_features["Customer ID"].nunique()
    )

    validation_results["customer_ids_valid"] = (
        validation_results["missing_customer_ids"] == 0
        and validation_results["duplicate_customer_ids"] == 0
    )

    # ---------------------------------------------------------
    # 4. Recency validation
    # ---------------------------------------------------------

    validation_results["missing_recency"] = int(
        rfm_features["Recency"].isna().sum()
    )

    validation_results["negative_recency"] = int(
        (rfm_features["Recency"] < 0).sum()
    )

    validation_results["recency_valid"] = (
        validation_results["missing_recency"] == 0
        and validation_results["negative_recency"] == 0
    )

    # ---------------------------------------------------------
    # 5. Frequency validation
    # ---------------------------------------------------------

    validation_results["missing_frequency"] = int(
        rfm_features["Frequency"].isna().sum()
    )

    validation_results["non_positive_frequency"] = int(
        (rfm_features["Frequency"] <= 0).sum()
    )

    validation_results["frequency_valid"] = (
        validation_results["missing_frequency"] == 0
        and validation_results["non_positive_frequency"] == 0
    )

    # ---------------------------------------------------------
    # 6. Total Monetary Value validation
    # ---------------------------------------------------------

    validation_results["missing_total_monetary_value"] = int(
        rfm_features["Total_Monetary_Value"].isna().sum()
    )

    validation_results["non_positive_total_monetary_value"] = int(
        (rfm_features["Total_Monetary_Value"] <= 0).sum()
    )

    validation_results["total_monetary_value_valid"] = (
        validation_results["missing_total_monetary_value"] == 0
        and validation_results["non_positive_total_monetary_value"] == 0
    )

    # ---------------------------------------------------------
    # 7. Average Order Spend validation
    # ---------------------------------------------------------

    validation_results["missing_average_order_spend"] = int(
        rfm_features["Average_Order_Spend"].isna().sum()
    )

    validation_results["non_positive_average_order_spend"] = int(
        (rfm_features["Average_Order_Spend"] <= 0).sum()
    )

    validation_results["average_order_spend_valid"] = (
        validation_results["missing_average_order_spend"] == 0
        and validation_results["non_positive_average_order_spend"] == 0
    )

    # ---------------------------------------------------------
    # 8. Monetary consistency validation
    # ---------------------------------------------------------

    expected_average_order_spend = (
        rfm_features["Total_Monetary_Value"]
        / rfm_features["Frequency"]
    )

    monetary_difference = (
        expected_average_order_spend
        - rfm_features["Average_Order_Spend"]
    ).abs()

    validation_results["maximum_monetary_difference"] = float(
        monetary_difference.max()
    )

    # Average Order Spend is rounded to 2 decimal places.
    # Therefore, a maximum difference of approximately 0.005
    # is mathematically expected.

    rounding_tolerance = 0.005 + 1e-9

    validation_results["rounding_tolerance"] = (
    rounding_tolerance
    )

    validation_results["monetary_consistency_valid"] = (
    validation_results["maximum_monetary_difference"]
    <= rounding_tolerance
     )

    # ---------------------------------------------------------
    # 9. Overall validation
    # ---------------------------------------------------------

    validation_checks = [
        validation_results["has_customers"],
        validation_results["required_columns_present"],
        validation_results["customer_ids_valid"],
        validation_results["recency_valid"],
        validation_results["frequency_valid"],
        validation_results["total_monetary_value_valid"],
        validation_results["average_order_spend_valid"],
        validation_results["monetary_consistency_valid"],
    ]

    validation_results["passed"] = all(
        validation_checks
    )

    return validation_results



####################################################################
#------------ 2. CLV Feature Engineering -------------------
####################################################################

#---------- 2.1.Analyze CLV feature -------------------
def analyze_clv_basis(
    df_cleaned: pd.DataFrame,
    rfm_features: pd.DataFrame,
    cutoff_date: pd.Timestamp | None = None,
) -> dict:
    """
    Establish and validate the analytical basis for
    historical Customer Lifetime Value (CLV).

    Historical CLV
    --------------
    Historical CLV is defined as the cumulative qualifying
    customer spend available on or before the point-in-time cutoff.

    Cutoff behavior
    ---------------
    If cutoff_date is None:
        Automatically calculate the cutoff as:

            maximum InvoiceDate - 90 days

    If cutoff_date is supplied:
        Use the supplied cutoff date.

    Parameters
    ----------
    df_cleaned : pd.DataFrame
        Cleaned transaction-level dataframe.

    rfm_features : pd.DataFrame
        Validated customer-level RFM dataframe.

    cutoff_date : pd.Timestamp or None, default=None
        Optional point-in-time cutoff date.

    Returns
    -------
    dict
        Contains:
        - summary
        - distribution
        - customer_analysis
        - clv_transactions
        - validation
    """

    # =========================================================
    # 1. Validate required RFM columns
    # =========================================================

    required_rfm_columns = {
        "Customer ID",
        "Recency",
        "Frequency",
        "Total_Monetary_Value",
        "Average_Order_Spend",
    }

    missing_rfm_columns = (
        required_rfm_columns - set(rfm_features.columns)
    )

    if missing_rfm_columns:
        raise ValueError(
            f"Missing required RFM columns: "
            f"{sorted(missing_rfm_columns)}"
        )

    # =========================================================
    # 2. Validate required transaction columns
    # =========================================================

    required_transaction_columns = {
        "Customer ID",
        "Invoice",
        "InvoiceDate",
        "Transaction Type",
        "Customer ID Missing Flag",
        "Include In Sales Analysis",
        "Analytical Transaction Value",
    }

    missing_transaction_columns = (
        required_transaction_columns - set(df_cleaned.columns)
    )

    if missing_transaction_columns:
        raise ValueError(
            "Missing required transaction columns: "
            f"{sorted(missing_transaction_columns)}"
        )

    # =========================================================
    # 3. Create working copies
    # =========================================================

    working_df = df_cleaned.copy()
    rfm = rfm_features.copy()

    # =========================================================
    # 4. Ensure InvoiceDate is datetime
    # =========================================================

    working_df["InvoiceDate"] = pd.to_datetime(
        working_df["InvoiceDate"],
        errors="raise",
    )

    # =========================================================
    # 5. Determine cutoff date
    # =========================================================

    maximum_invoice_date = working_df["InvoiceDate"].max()

    if pd.isna(maximum_invoice_date):
        raise ValueError(
            "InvoiceDate contains no valid dates."
        )

    if cutoff_date is None:

        cutoff_date = (
            maximum_invoice_date
            - pd.Timedelta(days=90)
        )

        cutoff_source = "Automatically calculated"

    else:

        cutoff_date = pd.Timestamp(cutoff_date)

        cutoff_source = "User supplied"

    # =========================================================
    # 6. Validate cutoff against available data
    # =========================================================

    if cutoff_date > maximum_invoice_date:
        raise ValueError(
            "Cutoff date cannot be later than "
            "the maximum InvoiceDate in the dataset."
        )

    # =========================================================
    # 7. Select qualifying historical transactions
    # =========================================================

    clv_transactions = working_df[
        (working_df["InvoiceDate"] <= cutoff_date)
        & (working_df["Transaction Type"] == "Purchase")
        & (working_df["Customer ID Missing Flag"] == 0)
        & (working_df["Include In Sales Analysis"] == True)
    ].copy()

    if clv_transactions.empty:
        raise ValueError(
            "No qualifying historical transactions found "
            "on or before the supplied/calculated cutoff date."
        )

    # =========================================================
    # 8. Compare customer population with RFM
    # =========================================================

    transaction_customers = set(
        clv_transactions["Customer ID"].unique()
    )

    rfm_customers = set(
        rfm["Customer ID"].unique()
    )

    missing_from_rfm = (
        transaction_customers - rfm_customers
    )

    extra_rfm_customers = (
        rfm_customers - transaction_customers
    )

    # =========================================================
    # 9. Customer-level historical information
    # =========================================================

    transaction_history = (
        clv_transactions
        .groupby("Customer ID")
        .agg(
            First_Purchase_Date=("InvoiceDate", "min"),
            Last_Purchase_Date=("InvoiceDate", "max"),
            Historical_Invoice_Count=("Invoice", "nunique"),
        )
        .reset_index()
    )

    # =========================================================
    # 10. Customer lifetime duration
    # =========================================================

    transaction_history["Customer_Lifetime_Days"] = (
        transaction_history["Last_Purchase_Date"]
        - transaction_history["First_Purchase_Date"]
    ).dt.total_seconds() / (24 * 60 * 60)

    # =========================================================
    # 11. Merge RFM with historical information
    # =========================================================

    clv_analysis = rfm.merge(
        transaction_history,
        on="Customer ID",
        how="left",
        validate="one_to_one",
    )

    # =========================================================
    # 12. Historical CLV
    # =========================================================

    clv_analysis["Historical_CLV"] = (
        clv_analysis["Total_Monetary_Value"]
    )

    # =========================================================
    # 13. Historical average order value
    # =========================================================

    clv_analysis["Historical_Average_Order_Value"] = (
        clv_analysis["Historical_CLV"]
        / clv_analysis["Frequency"]
    )

    # =========================================================
    # 14. Round analytical values
    # =========================================================

    clv_analysis["Customer_Lifetime_Days"] = (
        clv_analysis["Customer_Lifetime_Days"].round(2)
    )

    clv_analysis["Historical_CLV"] = (
        clv_analysis["Historical_CLV"].round(2)
    )

    clv_analysis["Historical_Average_Order_Value"] = (
        clv_analysis[
            "Historical_Average_Order_Value"
        ].round(2)
    )

    # =========================================================
    # 15. CLV distribution
    # =========================================================

    clv_distribution = (
        clv_analysis[
            [
                "Historical_CLV",
                "Customer_Lifetime_Days",
                "Historical_Average_Order_Value",
            ]
        ].describe()
    )

    # =========================================================
    # 16. Lifetime analysis
    # =========================================================

    zero_lifetime_customers = int(
        (
            clv_analysis["Customer_Lifetime_Days"]
            == 0
        ).sum()
    )

    multiple_purchase_date_customers = int(
        (
            clv_analysis["Customer_Lifetime_Days"]
            > 0
        ).sum()
    )

    # =========================================================
    # 17. RFM-to-CLV monetary consistency
    # =========================================================

    monetary_difference = (
        clv_analysis["Total_Monetary_Value"]
        - clv_analysis["Historical_CLV"]
    ).abs()

    maximum_rfm_clv_difference = float(
        monetary_difference.max()
    )

    # =========================================================
    # 18. RFM-to-CLV frequency consistency
    # =========================================================

    frequency_difference = (
        clv_analysis["Frequency"]
        - clv_analysis["Historical_Invoice_Count"]
    ).abs()

    maximum_frequency_difference = int(
        frequency_difference.max()
    )

    # =========================================================
    # 19. Point-in-time leakage check
    # =========================================================

    transactions_after_cutoff = int(
        (
            clv_transactions["InvoiceDate"]
            > cutoff_date
        ).sum()
    )

    # =========================================================
    # 20. Summary
    # =========================================================

    summary = {
        "maximum_invoice_date": maximum_invoice_date,
        "cutoff_date": cutoff_date,
        "cutoff_source": cutoff_source,
        "prediction_window_days": 90,

        "historical_transaction_count": len(
            clv_transactions
        ),

        "customer_count": clv_analysis[
            "Customer ID"
        ].nunique(),

        "distinct_invoice_count": clv_transactions[
            "Invoice"
        ].nunique(),

        "total_historical_revenue": round(
            clv_transactions[
                "Analytical Transaction Value"
            ].sum(),
            2,
        ),

        "minimum_historical_clv": clv_analysis[
            "Historical_CLV"
        ].min(),

        "median_historical_clv": clv_analysis[
            "Historical_CLV"
        ].median(),

        "maximum_historical_clv": clv_analysis[
            "Historical_CLV"
        ].max(),

        "minimum_customer_lifetime_days": clv_analysis[
            "Customer_Lifetime_Days"
        ].min(),

        "median_customer_lifetime_days": clv_analysis[
            "Customer_Lifetime_Days"
        ].median(),

        "maximum_customer_lifetime_days": clv_analysis[
            "Customer_Lifetime_Days"
        ].max(),

        "zero_lifetime_customers": (
            zero_lifetime_customers
        ),

        "multiple_purchase_date_customers": (
            multiple_purchase_date_customers
        ),

        "maximum_rfm_clv_difference": (
            maximum_rfm_clv_difference
        ),

        "maximum_frequency_difference": (
            maximum_frequency_difference
        ),

        "transactions_after_cutoff": (
            transactions_after_cutoff
        ),
    }

    # =========================================================
    # 21. Validation
    # =========================================================

    validation = {
        "customer_population_valid": (
            len(missing_from_rfm) == 0
        ),

        "rfm_clv_monetary_consistency": (
            maximum_rfm_clv_difference == 0
        ),

        "rfm_clv_frequency_consistency": (
            maximum_frequency_difference == 0
        ),

        "point_in_time_cutoff_valid": (
            transactions_after_cutoff == 0
        ),

        "historical_clv_values_valid": (
            clv_analysis["Historical_CLV"]
            .notna()
            .all()
            and
            (
                clv_analysis["Historical_CLV"] > 0
            ).all()
        ),
    }

    validation["passed"] = all(
        validation.values()
    )

    # =========================================================
    # 22. Return results
    # =========================================================

    return {
        "summary": summary,
        "distribution": clv_distribution,
        "customer_analysis": clv_analysis,
        "clv_transactions": clv_transactions,
        "validation": validation,
    }

# --------- 2.2 Create CLV feature ----------------
def create_clv_features(
    df_cleaned: pd.DataFrame,
    rfm_features: pd.DataFrame,
    cutoff_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Create customer-level historical CLV features.

    Features
    --------
    Historical_CLV:
        Cumulative qualifying customer spend available on or
        before the point-in-time cutoff.

    Customer_Lifetime_Days:
        Number of days between the customer's first and
        latest qualifying purchase available before the cutoff.

    Historical_Average_Order_Value:
        Historical CLV divided by the number of distinct
        historical orders.

    Cutoff behavior
    ---------------
    If cutoff_date is None:
        Automatically calculate:

            maximum InvoiceDate - 90 days

    If cutoff_date is supplied:
        Use the supplied cutoff date.

    Parameters
    ----------
    df_cleaned : pd.DataFrame
        Cleaned transaction-level dataframe.

    rfm_features : pd.DataFrame
        Validated customer-level RFM dataframe.

    cutoff_date : pd.Timestamp or None, default=None
        Optional point-in-time cutoff date.

    Returns
    -------
    pd.DataFrame
        Customer-level CLV feature dataframe.
    """

    # =========================================================
    # 1. Validate required RFM columns
    # =========================================================

    required_rfm_columns = {
        "Customer ID",
        "Frequency",
        "Total_Monetary_Value",
    }

    missing_rfm_columns = (
        required_rfm_columns - set(rfm_features.columns)
    )

    if missing_rfm_columns:
        raise ValueError(
            f"Missing required RFM columns: "
            f"{sorted(missing_rfm_columns)}"
        )

    # =========================================================
    # 2. Validate required transaction columns
    # =========================================================

    required_transaction_columns = {
        "Customer ID",
        "Invoice",
        "InvoiceDate",
        "Transaction Type",
        "Customer ID Missing Flag",
        "Include In Sales Analysis",
        "Analytical Transaction Value",
    }

    missing_transaction_columns = (
        required_transaction_columns - set(df_cleaned.columns)
    )

    if missing_transaction_columns:
        raise ValueError(
            "Missing required transaction columns: "
            f"{sorted(missing_transaction_columns)}"
        )

    # =========================================================
    # 3. Work on copies
    # =========================================================

    working_df = df_cleaned.copy()
    rfm = rfm_features.copy()

    # =========================================================
    # 4. Ensure InvoiceDate is datetime
    # =========================================================

    working_df["InvoiceDate"] = pd.to_datetime(
        working_df["InvoiceDate"],
        errors="raise",
    )

    # =========================================================
    # 5. Determine cutoff date
    # =========================================================

    maximum_invoice_date = working_df["InvoiceDate"].max()

    if pd.isna(maximum_invoice_date):
        raise ValueError(
            "InvoiceDate contains no valid dates."
        )

    if cutoff_date is None:

        cutoff_date = (
            maximum_invoice_date
            - pd.Timedelta(days=90)
        )

    else:

        cutoff_date = pd.Timestamp(cutoff_date)

    # =========================================================
    # 6. Validate cutoff date
    # =========================================================

    if cutoff_date > maximum_invoice_date:
        raise ValueError(
            "Cutoff date cannot be later than "
            "the maximum InvoiceDate."
        )

    # =========================================================
    # 7. Select qualifying historical transactions
    # =========================================================

    clv_transactions = working_df[
        (working_df["InvoiceDate"] <= cutoff_date)
        & (working_df["Transaction Type"] == "Purchase")
        & (working_df["Customer ID Missing Flag"] == 0)
        & (working_df["Include In Sales Analysis"] == True)
    ].copy()

    if clv_transactions.empty:
        raise ValueError(
            "No qualifying historical transactions found "
            "on or before the cutoff date."
        )

    # =========================================================
    # 8. Calculate first and last purchase dates
    # =========================================================

    customer_history = (
        clv_transactions
        .groupby("Customer ID")
        .agg(
            First_Purchase_Date=("InvoiceDate", "min"),
            Last_Purchase_Date=("InvoiceDate", "max"),
            Historical_Invoice_Count=("Invoice", "nunique"),
        )
        .reset_index()
    )

    # =========================================================
    # 9. Calculate customer lifetime
    # =========================================================

    customer_history["Customer_Lifetime_Days"] = (
        customer_history["Last_Purchase_Date"]
        - customer_history["First_Purchase_Date"]
    ).dt.total_seconds() / (24 * 60 * 60)

    # =========================================================
    # 10. Select the RFM customer-level foundation
    # =========================================================

    clv_features = rfm[
        [
            "Customer ID",
            "Frequency",
            "Total_Monetary_Value",
        ]
    ].copy()

    # =========================================================
    # 11. Historical CLV
    # =========================================================

    clv_features["Historical_CLV"] = (
        clv_features["Total_Monetary_Value"]
    )

    # =========================================================
    # 12. Merge customer history
    # =========================================================

    clv_features = clv_features.merge(
        customer_history[
            [
                "Customer ID",
                "First_Purchase_Date",
                "Last_Purchase_Date",
                "Historical_Invoice_Count",
                "Customer_Lifetime_Days",
            ]
        ],
        on="Customer ID",
        how="left",
        validate="one_to_one",
    )

    # =========================================================
    # 13. Historical average order value
    # =========================================================

    clv_features["Historical_Average_Order_Value"] = (
        clv_features["Historical_CLV"]
        / clv_features["Frequency"]
    )

    # =========================================================
    # 14. Final column selection
    # =========================================================

    clv_features = clv_features[
        [
            "Customer ID",
            "Historical_CLV",
            "Customer_Lifetime_Days",
            "Historical_Average_Order_Value",
            "First_Purchase_Date",
            "Last_Purchase_Date",
            "Historical_Invoice_Count",
        ]
    ].copy()

    # =========================================================
    # 15. Round numerical features
    # =========================================================

    clv_features["Historical_CLV"] = (
        clv_features["Historical_CLV"].round(2)
    )

    clv_features["Customer_Lifetime_Days"] = (
        clv_features["Customer_Lifetime_Days"].round(2)
    )

    clv_features["Historical_Average_Order_Value"] = (
        clv_features[
            "Historical_Average_Order_Value"
        ].round(2)
    )

    # =========================================================
    # 16. Final validation
    # =========================================================

    if clv_features["Customer ID"].duplicated().any():
        raise ValueError(
            "CLV output contains duplicate Customer IDs."
        )

    if clv_features["Historical_CLV"].isna().any():
        raise ValueError(
            "CLV output contains missing Historical CLV values."
        )

    if (
        clv_features["Historical_CLV"] <= 0
    ).any():
        raise ValueError(
            "CLV output contains non-positive Historical CLV."
        )

    if clv_features["Customer_Lifetime_Days"].isna().any():
        raise ValueError(
            "CLV output contains missing Customer Lifetime values."
        )

    if (
        clv_features["Customer_Lifetime_Days"] < 0
    ).any():
        raise ValueError(
            "CLV output contains negative Customer Lifetime values."
        )

    if clv_features[
        "Historical_Average_Order_Value"
    ].isna().any():
        raise ValueError(
            "CLV output contains missing Historical Average Order Value."
        )

    return clv_features

#---------- 2.3 Validate CLV feature ----------------
def validate_clv_features(
    clv_features: pd.DataFrame,
) -> dict:
    """
    Validate customer-level historical CLV features.

    Validation Rules
    ----------------
    1. Required CLV columns must exist.
    2. Customer ID must be unique and non-missing.
    3. Historical CLV must be present and positive.
    4. Customer Lifetime Days must be present and non-negative.
    5. Historical Average Order Value must be present and positive.
    6. First and Last Purchase Dates must be present.
    7. Last Purchase Date must not precede First Purchase Date.
    8. Historical Invoice Count must be positive.
    9. Historical Average Order Value must be mathematically
       consistent with Historical CLV / Historical Invoice Count.
    10. Customer Lifetime Days must be consistent with
        Last Purchase Date - First Purchase Date.

    Parameters
    ----------
    clv_features : pd.DataFrame
        Customer-level CLV feature dataframe.

    Returns
    -------
    dict
        Validation results and overall pass/fail status.
    """

    required_columns = [
        "Customer ID",
        "Historical_CLV",
        "Customer_Lifetime_Days",
        "Historical_Average_Order_Value",
        "First_Purchase_Date",
        "Last_Purchase_Date",
        "Historical_Invoice_Count",
    ]

    validation_results = {}

    # =========================================================
    # 1. Required columns
    # =========================================================

    missing_columns = [
        column
        for column in required_columns
        if column not in clv_features.columns
    ]

    validation_results["required_columns_present"] = (
        len(missing_columns) == 0
    )

    validation_results["missing_columns"] = missing_columns

    if missing_columns:
        validation_results["passed"] = False
        return validation_results

    # =========================================================
    # 2. Customer population
    # =========================================================

    validation_results["customer_count"] = len(
        clv_features
    )

    validation_results["has_customers"] = (
        len(clv_features) > 0
    )

    # =========================================================
    # 3. Customer ID validation
    # =========================================================

    validation_results["missing_customer_ids"] = int(
        clv_features["Customer ID"].isna().sum()
    )

    validation_results["duplicate_customer_ids"] = int(
        clv_features["Customer ID"].duplicated().sum()
    )

    validation_results["unique_customer_ids"] = (
        clv_features["Customer ID"].nunique()
    )

    validation_results["customer_ids_valid"] = (
        validation_results["missing_customer_ids"] == 0
        and validation_results["duplicate_customer_ids"] == 0
    )

    # =========================================================
    # 4. Historical CLV validation
    # =========================================================

    validation_results["missing_historical_clv"] = int(
        clv_features["Historical_CLV"].isna().sum()
    )

    validation_results["non_positive_historical_clv"] = int(
        (clv_features["Historical_CLV"] <= 0).sum()
    )

    validation_results["historical_clv_valid"] = (
        validation_results["missing_historical_clv"] == 0
        and validation_results["non_positive_historical_clv"] == 0
    )

    # =========================================================
    # 5. Customer lifetime validation
    # =========================================================

    validation_results["missing_customer_lifetime"] = int(
        clv_features["Customer_Lifetime_Days"].isna().sum()
    )

    validation_results["negative_customer_lifetime"] = int(
        (clv_features["Customer_Lifetime_Days"] < 0).sum()
    )

    validation_results["customer_lifetime_valid"] = (
        validation_results["missing_customer_lifetime"] == 0
        and validation_results["negative_customer_lifetime"] == 0
    )

    # =========================================================
    # 6. Purchase date validation
    # =========================================================

    validation_results["missing_first_purchase_date"] = int(
        clv_features["First_Purchase_Date"].isna().sum()
    )

    validation_results["missing_last_purchase_date"] = int(
        clv_features["Last_Purchase_Date"].isna().sum()
    )

    invalid_purchase_date_order = (
        clv_features["Last_Purchase_Date"]
        < clv_features["First_Purchase_Date"]
    )

    validation_results["invalid_purchase_date_order"] = int(
        invalid_purchase_date_order.sum()
    )

    validation_results["purchase_dates_valid"] = (
        validation_results["missing_first_purchase_date"] == 0
        and validation_results["missing_last_purchase_date"] == 0
        and validation_results["invalid_purchase_date_order"] == 0
    )

    # =========================================================
    # 7. Historical invoice count validation
    # =========================================================

    validation_results["missing_historical_invoice_count"] = int(
        clv_features[
            "Historical_Invoice_Count"
        ].isna().sum()
    )

    validation_results["non_positive_historical_invoice_count"] = int(
        (
            clv_features[
                "Historical_Invoice_Count"
            ] <= 0
        ).sum()
    )

    validation_results["historical_invoice_count_valid"] = (
        validation_results[
            "missing_historical_invoice_count"
        ] == 0
        and validation_results[
            "non_positive_historical_invoice_count"
        ] == 0
    )

    # =========================================================
    # 8. Historical Average Order Value validation
    # =========================================================

    validation_results["missing_average_order_value"] = int(
        clv_features[
            "Historical_Average_Order_Value"
        ].isna().sum()
    )

    validation_results["non_positive_average_order_value"] = int(
        (
            clv_features[
                "Historical_Average_Order_Value"
            ] <= 0
        ).sum()
    )

    validation_results["average_order_value_valid"] = (
        validation_results[
            "missing_average_order_value"
        ] == 0
        and validation_results[
            "non_positive_average_order_value"
        ] == 0
    )

    # =========================================================
    # 9. CLV mathematical consistency
    # =========================================================

    expected_average_order_value = (
        clv_features["Historical_CLV"]
        / clv_features["Historical_Invoice_Count"]
    )

    average_order_difference = (
        expected_average_order_value
        - clv_features[
            "Historical_Average_Order_Value"
        ]
    ).abs()

    maximum_average_order_difference = float(
        average_order_difference.max()
    )

    validation_results[
        "maximum_average_order_difference"
    ] = maximum_average_order_difference

    # Values are rounded to two decimal places in the
    # feature-generation function, so a difference of
    # approximately 0.005 is expected.

    rounding_tolerance = 0.005 + 1e-9

    validation_results["rounding_tolerance"] = (
        rounding_tolerance
    )

    validation_results[
        "average_order_value_consistency_valid"
    ] = (
        maximum_average_order_difference
        <= rounding_tolerance
    )

    # =========================================================
    # 10. Customer lifetime mathematical consistency
    # =========================================================

    expected_lifetime_days = (
        clv_features["Last_Purchase_Date"]
        - clv_features["First_Purchase_Date"]
    ).dt.total_seconds() / (24 * 60 * 60)

    lifetime_difference = (
        expected_lifetime_days
        - clv_features["Customer_Lifetime_Days"]
    ).abs()

    maximum_lifetime_difference = float(
        lifetime_difference.max()
    )

    validation_results[
        "maximum_lifetime_difference"
    ] = maximum_lifetime_difference

    validation_results[
        "customer_lifetime_consistency_valid"
    ] = (
        maximum_lifetime_difference <= 0.01
    )

    # =========================================================
    # 11. Overall validation
    # =========================================================

    validation_checks = [
        validation_results["has_customers"],
        validation_results["required_columns_present"],
        validation_results["customer_ids_valid"],
        validation_results["historical_clv_valid"],
        validation_results["customer_lifetime_valid"],
        validation_results["purchase_dates_valid"],
        validation_results["historical_invoice_count_valid"],
        validation_results["average_order_value_valid"],
        validation_results[
            "average_order_value_consistency_valid"
        ],
        validation_results[
            "customer_lifetime_consistency_valid"
        ],
    ]

    validation_results["passed"] = all(
        validation_checks
    )

    return validation_results



################################################################
#---------- 3. Average_basket_size feature engineering----------------
################################################################

#---- 3.1 A. analyze Average_basket_size feature -----
def analyze_avg_basket_size_basis(
    df_cleaned: pd.DataFrame,
    cutoff_date: pd.Timestamp | None = None,
) -> dict:
    """
    Analyze the transaction-level basis for Average Basket Size.

    Average Basket Size
    -------------------
    Average number of purchased items per distinct customer order.

    Basket Size for an order:
        Sum of qualifying purchased Quantity within the invoice.

    Customer Average Basket Size:
        Total purchased Quantity across qualifying orders
        divided by the number of distinct qualifying orders.

    Cutoff behavior
    ---------------
    If cutoff_date is None:
        Automatically calculate:
            maximum InvoiceDate - 90 days

    If cutoff_date is supplied:
        Use the supplied cutoff date.
    """

    required_columns = {
        "Customer ID",
        "Invoice",
        "InvoiceDate",
        "Quantity",
        "Transaction Type",
        "Customer ID Missing Flag",
        "Include In Sales Analysis",
    }

    missing_columns = required_columns - set(df_cleaned.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    working_df = df_cleaned.copy()

    working_df["InvoiceDate"] = pd.to_datetime(
        working_df["InvoiceDate"],
        errors="raise",
    )

    # ---------------------------------------------------------
    # Determine point-in-time cutoff
    # ---------------------------------------------------------

    maximum_invoice_date = working_df["InvoiceDate"].max()

    if pd.isna(maximum_invoice_date):
        raise ValueError(
            "InvoiceDate contains no valid dates."
        )

    if cutoff_date is None:
        cutoff_date = (
            maximum_invoice_date
            - pd.Timedelta(days=90)
        )
        cutoff_source = "Automatically calculated"
    else:
        cutoff_date = pd.Timestamp(cutoff_date)
        cutoff_source = "User supplied"

    if cutoff_date > maximum_invoice_date:
        raise ValueError(
            "cutoff_date cannot be later than the maximum InvoiceDate."
        )

    # ---------------------------------------------------------
    # Select qualifying historical purchases
    # ---------------------------------------------------------

    basket_transactions = working_df[
        (working_df["InvoiceDate"] <= cutoff_date)
        & (working_df["Transaction Type"] == "Purchase")
        & (working_df["Customer ID Missing Flag"] == 0)
        & (working_df["Include In Sales Analysis"] == True)
    ].copy()

    if basket_transactions.empty:
        raise ValueError(
            "No qualifying purchase transactions found "
            "before or on the supplied cutoff date."
        )

    # ---------------------------------------------------------
    # Calculate basket size at invoice level
    # ---------------------------------------------------------

    invoice_baskets = (
        basket_transactions
        .groupby(
            ["Customer ID", "Invoice"],
            as_index=False
        )
        .agg(
            Basket_Size=("Quantity", "sum"),
            Invoice_Date=("InvoiceDate", "max"),
        )
    )

    # ---------------------------------------------------------
    # Customer-level basket analysis
    # ---------------------------------------------------------

    customer_basket_analysis = (
        invoice_baskets
        .groupby("Customer ID")
        .agg(
            Total_Qualifying_Quantity=(
                "Basket_Size",
                "sum"
            ),
            Number_of_Orders=(
                "Invoice",
                "nunique"
            ),
            Minimum_Basket_Size=(
                "Basket_Size",
                "min"
            ),
            Median_Basket_Size=(
                "Basket_Size",
                "median"
            ),
            Maximum_Basket_Size=(
                "Basket_Size",
                "max"
            ),
        )
        .reset_index()
    )

    customer_basket_analysis[
        "Average_Basket_Size"
    ] = (
        customer_basket_analysis[
            "Total_Qualifying_Quantity"
        ]
        / customer_basket_analysis["Number_of_Orders"]
    )

    # ---------------------------------------------------------
    # Overall summary
    # ---------------------------------------------------------

    summary = {
        "maximum_invoice_date": maximum_invoice_date,
        "cutoff_date": cutoff_date,
        "cutoff_source": cutoff_source,
        "prediction_window_days": 90,

        "historical_transaction_count": len(
            basket_transactions
        ),

        "customer_count": (
            customer_basket_analysis[
                "Customer ID"
            ].nunique()
        ),

        "distinct_invoice_count": (
            invoice_baskets["Invoice"].nunique()
        ),

        "total_qualifying_quantity": float(
            basket_transactions["Quantity"].sum()
        ),

        "minimum_average_basket_size": float(
            customer_basket_analysis[
                "Average_Basket_Size"
            ].min()
        ),

        "median_average_basket_size": float(
            customer_basket_analysis[
                "Average_Basket_Size"
            ].median()
        ),

        "maximum_average_basket_size": float(
            customer_basket_analysis[
                "Average_Basket_Size"
            ].max()
        ),

        "minimum_orders_per_customer": int(
            customer_basket_analysis[
                "Number_of_Orders"
            ].min()
        ),

        "maximum_orders_per_customer": int(
            customer_basket_analysis[
                "Number_of_Orders"
            ].max()
        ),
    }

    # ---------------------------------------------------------
    # Validation checks for analysis
    # ---------------------------------------------------------

    validation = {
        "customer_count_valid": (
            customer_basket_analysis["Customer ID"]
            .notna()
            .all()
        ),

        "no_duplicate_customer_ids": (
            not customer_basket_analysis[
                "Customer ID"
            ].duplicated().any()
        ),

        "basket_sizes_positive": (
            (
                invoice_baskets["Basket_Size"] > 0
            ).all()
        ),

        "average_basket_sizes_positive": (
            (
                customer_basket_analysis[
                    "Average_Basket_Size"
                ] > 0
            ).all()
        ),

        "orders_positive": (
            (
                customer_basket_analysis[
                    "Number_of_Orders"
                ] > 0
            ).all()
        ),

        "cutoff_valid": (
            basket_transactions["InvoiceDate"]
            .max()
            <= cutoff_date
        ),
    }

    validation["passed"] = all(
        validation.values()
    )

    return {
        "summary": summary,
        "customer_analysis": customer_basket_analysis,
        "invoice_baskets": invoice_baskets,
        "transactions": basket_transactions,
        "validation": validation,
    }

#---- 3.2 Create Average_basket_size feature ----------------
def create_avg_basket_size_features(
    df_cleaned: pd.DataFrame,
    cutoff_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Create customer-level Average Basket Size features.

    Average Basket Size
    -------------------
    Average number of purchased items per distinct qualifying order.

    Calculation
    -----------
    1. Calculate total purchased Quantity for each customer and invoice.
    2. Count distinct qualifying invoices for each customer.
    3. Divide total purchased Quantity by the number of orders.

    Cutoff behavior
    ---------------
    If cutoff_date is None:
        Automatically calculate:
            maximum InvoiceDate - 90 days

    If cutoff_date is supplied:
        Use the supplied cutoff date.

    Returns
    -------
    pd.DataFrame
        Customer-level dataframe containing:
            Customer ID
            Average_Basket_Size
    """

    required_columns = {
        "Customer ID",
        "Invoice",
        "InvoiceDate",
        "Quantity",
        "Transaction Type",
        "Customer ID Missing Flag",
        "Include In Sales Analysis",
    }

    missing_columns = required_columns - set(df_cleaned.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    working_df = df_cleaned.copy()

    working_df["InvoiceDate"] = pd.to_datetime(
        working_df["InvoiceDate"],
        errors="raise",
    )

    # ---------------------------------------------------------
    # Determine point-in-time cutoff
    # ---------------------------------------------------------

    maximum_invoice_date = working_df["InvoiceDate"].max()

    if pd.isna(maximum_invoice_date):
        raise ValueError(
            "InvoiceDate contains no valid dates."
        )

    if cutoff_date is None:
        cutoff_date = (
            maximum_invoice_date
            - pd.Timedelta(days=90)
        )
    else:
        cutoff_date = pd.Timestamp(cutoff_date)

    if cutoff_date > maximum_invoice_date:
        raise ValueError(
            "cutoff_date cannot be later than the maximum InvoiceDate."
        )

    # ---------------------------------------------------------
    # Select qualifying historical purchases
    # ---------------------------------------------------------

    basket_transactions = working_df[
        (working_df["InvoiceDate"] <= cutoff_date)
        & (working_df["Transaction Type"] == "Purchase")
        & (working_df["Customer ID Missing Flag"] == 0)
        & (working_df["Include In Sales Analysis"] == True)
    ].copy()

    if basket_transactions.empty:
        raise ValueError(
            "No qualifying purchase transactions found "
            "before or on the supplied cutoff date."
        )

    # ---------------------------------------------------------
    # Calculate basket size for each customer and invoice
    # ---------------------------------------------------------

    invoice_baskets = (
        basket_transactions
        .groupby(
            ["Customer ID", "Invoice"],
            as_index=False,
        )
        .agg(
            Basket_Size=("Quantity", "sum"),
        )
    )

    # ---------------------------------------------------------
    # Calculate customer-level Average Basket Size
    # ---------------------------------------------------------

    customer_basket = (
        invoice_baskets
        .groupby("Customer ID")
        .agg(
            Total_Qualifying_Quantity=(
                "Basket_Size",
                "sum",
            ),
            Number_of_Orders=(
                "Invoice",
                "nunique",
            ),
        )
        .reset_index()
    )

    customer_basket["Average_Basket_Size"] = (
        customer_basket["Total_Qualifying_Quantity"]
        / customer_basket["Number_of_Orders"]
    )

    # ---------------------------------------------------------
    # Keep only the production feature
    # ---------------------------------------------------------

    avg_basket_features = customer_basket[
        [
            "Customer ID",
            "Average_Basket_Size",
        ]
    ].copy()

    avg_basket_features["Average_Basket_Size"] = (
        avg_basket_features["Average_Basket_Size"]
        .round(2)
    )

    # ---------------------------------------------------------
    # Basic structural validation
    # ---------------------------------------------------------

    if avg_basket_features["Customer ID"].isna().any():
        raise ValueError(
            "Average Basket Size features contain missing Customer IDs."
        )

    if avg_basket_features["Customer ID"].duplicated().any():
        raise ValueError(
            "Average Basket Size features contain duplicate Customer IDs."
        )

    if avg_basket_features["Average_Basket_Size"].isna().any():
        raise ValueError(
            "Average Basket Size contains missing values."
        )

    if (
        avg_basket_features["Average_Basket_Size"] <= 0
    ).any():
        raise ValueError(
            "Average Basket Size contains zero or negative values."
        )

    return avg_basket_features

#---- 3.3 Validate Average_basket_size feature ----------------
def validate_avg_basket_size_features(
    avg_basket_features: pd.DataFrame,
) -> dict:
    """
    Validate customer-level Average Basket Size features.

    Validation Rules
    ----------------
    1. Required columns must exist.
    2. Customer ID must be unique and non-missing.
    3. Average Basket Size must be present.
    4. Average Basket Size must be positive.
    """

    required_columns = [
        "Customer ID",
        "Average_Basket_Size",
    ]

    validation_results = {}

    # ---------------------------------------------------------
    # Required columns
    # ---------------------------------------------------------

    missing_columns = [
        column
        for column in required_columns
        if column not in avg_basket_features.columns
    ]

    validation_results["required_columns_present"] = (
        len(missing_columns) == 0
    )

    validation_results["missing_columns"] = missing_columns

    if missing_columns:
        validation_results["passed"] = False
        return validation_results

    # ---------------------------------------------------------
    # Customer ID validation
    # ---------------------------------------------------------

    validation_results["customer_count"] = len(
        avg_basket_features
    )

    validation_results["has_customers"] = (
        len(avg_basket_features) > 0
    )

    validation_results["missing_customer_ids"] = int(
        avg_basket_features["Customer ID"]
        .isna()
        .sum()
    )

    validation_results["duplicate_customer_ids"] = int(
        avg_basket_features["Customer ID"]
        .duplicated()
        .sum()
    )

    validation_results["unique_customer_ids"] = (
        avg_basket_features["Customer ID"]
        .nunique()
    )

    validation_results["customer_ids_valid"] = (
        validation_results["missing_customer_ids"] == 0
        and validation_results["duplicate_customer_ids"] == 0
    )

    # ---------------------------------------------------------
    # Average Basket Size validation
    # ---------------------------------------------------------

    validation_results["missing_average_basket_size"] = int(
        avg_basket_features["Average_Basket_Size"]
        .isna()
        .sum()
    )

    validation_results["non_positive_average_basket_size"] = int(
        (
            avg_basket_features["Average_Basket_Size"]
            <= 0
        ).sum()
    )

    validation_results["average_basket_size_valid"] = (
        validation_results["missing_average_basket_size"] == 0
        and validation_results[
            "non_positive_average_basket_size"
        ] == 0
    )

    # ---------------------------------------------------------
    # Overall validation
    # ---------------------------------------------------------

    validation_checks = [
        validation_results["has_customers"],
        validation_results["required_columns_present"],
        validation_results["customer_ids_valid"],
        validation_results["average_basket_size_valid"],
    ]

    validation_results["passed"] = all(
        validation_checks
    )

    return validation_results



################################################################
#---------- 4. analyze_purchase_frequency_trend_basis ----------------
################################################################


#---- 4.1 Analyze purchase frequency trend basis ----------------
def analyze_purchase_frequency_trend_basis(
    df_cleaned: pd.DataFrame,
    cutoff_date: pd.Timestamp | None = None,
) -> dict:
    """
    Analyze the basis for Purchase Frequency Trend.

    Purchase Frequency Trend
    ------------------------
    Linear slope of monthly distinct purchase-order frequency
    over time.

    Monthly purchase frequency is defined as the number of
    distinct purchase invoices for a customer in each month.

    IMPORTANT
    ---------
    Months with zero purchases are included between the
    customer's first purchase month and the point-in-time
    cutoff.

    Interpretation
    --------------
    Positive slope:
        Purchase frequency is increasing.

    Negative slope:
        Purchase frequency is decreasing.

    Near-zero slope:
        Purchase frequency is relatively stable.

    Customers with fewer than two observed purchase months:
        Trend is assigned 0.0 because a meaningful temporal
        slope cannot be estimated from a single observed
        purchase month.

    Cutoff behavior
    ---------------
    If cutoff_date is None:
        Automatically calculate:
            maximum InvoiceDate - 90 days

    If cutoff_date is supplied:
        Use the supplied cutoff date.
    """

    required_columns = {
        "Customer ID",
        "Invoice",
        "InvoiceDate",
        "Transaction Type",
        "Customer ID Missing Flag",
        "Include In Sales Analysis",
    }

    missing_columns = required_columns - set(df_cleaned.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    working_df = df_cleaned.copy()

    working_df["InvoiceDate"] = pd.to_datetime(
        working_df["InvoiceDate"],
        errors="raise",
    )

    # ---------------------------------------------------------
    # Determine point-in-time cutoff
    # ---------------------------------------------------------

    maximum_invoice_date = working_df["InvoiceDate"].max()

    if pd.isna(maximum_invoice_date):
        raise ValueError(
            "InvoiceDate contains no valid dates."
        )

    if cutoff_date is None:
        cutoff_date = (
            maximum_invoice_date
            - pd.Timedelta(days=90)
        )
        cutoff_source = "Automatically calculated"
    else:
        cutoff_date = pd.Timestamp(cutoff_date)
        cutoff_source = "User supplied"

    if cutoff_date > maximum_invoice_date:
        raise ValueError(
            "cutoff_date cannot be later than "
            "the maximum InvoiceDate."
        )

    # ---------------------------------------------------------
    # Select qualifying historical purchases
    # ---------------------------------------------------------

    frequency_transactions = working_df[
        (working_df["InvoiceDate"] <= cutoff_date)
        & (working_df["Transaction Type"] == "Purchase")
        & (working_df["Customer ID Missing Flag"] == 0)
        & (working_df["Include In Sales Analysis"] == True)
    ].copy()

    if frequency_transactions.empty:
        raise ValueError(
            "No qualifying purchase transactions found "
            "before or on the supplied cutoff date."
        )

    # ---------------------------------------------------------
    # Create monthly purchase frequency
    # ---------------------------------------------------------

    frequency_transactions["Purchase_Month"] = (
        frequency_transactions["InvoiceDate"]
        .dt.to_period("M")
    )

    monthly_purchase_frequency = (
        frequency_transactions
        .groupby(
            ["Customer ID", "Purchase_Month"]
        )["Invoice"]
        .nunique()
        .reset_index(
            name="Monthly_Purchase_Frequency"
        )
    )

    # ---------------------------------------------------------
    # Create complete monthly timeline
    #
    # Every month from the customer's first purchase month
    # through the cutoff month is represented.
    # Months without purchases receive frequency = 0.
    # ---------------------------------------------------------

    cutoff_month = cutoff_date.to_period("M")

    customer_first_month = (
        monthly_purchase_frequency
        .groupby("Customer ID")["Purchase_Month"]
        .min()
    )

    complete_monthly_frequency = []

    for customer_id, first_month in customer_first_month.items():

        customer_month_range = pd.period_range(
            start=first_month,
            end=cutoff_month,
            freq="M",
        )

        customer_month_df = pd.DataFrame(
            {
                "Customer ID": customer_id,
                "Purchase_Month": customer_month_range,
            }
        )

        complete_monthly_frequency.append(
            customer_month_df
        )

    complete_monthly_frequency = pd.concat(
        complete_monthly_frequency,
        ignore_index=True,
    )

    monthly_frequency = (
        complete_monthly_frequency
        .merge(
            monthly_purchase_frequency,
            on=[
                "Customer ID",
                "Purchase_Month",
            ],
            how="left",
            validate="one_to_one",
        )
    )

    monthly_frequency[
        "Monthly_Purchase_Frequency"
    ] = (
        monthly_frequency[
            "Monthly_Purchase_Frequency"
        ]
        .fillna(0)
        .astype(int)
    )

    # ---------------------------------------------------------
    # Create ordered numeric month index
    # ---------------------------------------------------------

    monthly_frequency["Month_Index"] = (
        monthly_frequency[
            "Purchase_Month"
        ].astype("int64")
    )

    # ---------------------------------------------------------
    # Calculate purchase-frequency slope
    # ---------------------------------------------------------

    def calculate_frequency_slope(group):
        """
        Calculate linear slope of monthly purchase frequency.

        Zero-purchase months are included.

        Customers with fewer than two observed purchase
        months receive a slope of zero.
        """

        observed_purchase_months = (
            group["Monthly_Purchase_Frequency"] > 0
        ).sum()

        if observed_purchase_months < 2:
            return 0.0

        x = group[
            "Month_Index"
        ].to_numpy(dtype=float)

        y = group[
            "Monthly_Purchase_Frequency"
        ].to_numpy(dtype=float)

        denominator = (
            (x - x.mean()) ** 2
        ).sum()

        if denominator == 0:
            return 0.0

        slope = (
            (
                (x - x.mean())
                * (y - y.mean())
            ).sum()
            / denominator
        )

        return float(slope)

    customer_frequency_trend = (
        monthly_frequency
        .groupby("Customer ID")
        .apply(
            calculate_frequency_slope,
            include_groups=False,
        )
        .rename(
            "Purchase_Frequency_Trend"
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # Supporting customer-level analysis
    # ---------------------------------------------------------

    customer_month_statistics = (
        monthly_frequency
        .groupby("Customer ID")
        .agg(
            Observed_Purchase_Months=(
                "Monthly_Purchase_Frequency",
                lambda values: int(
                    (values > 0).sum()
                ),
            ),
            Analysis_Months=(
                "Purchase_Month",
                "count",
            ),
            Total_Purchase_Orders=(
                "Monthly_Purchase_Frequency",
                "sum",
            ),
        )
        .reset_index()
    )

    customer_frequency_trend = (
        customer_frequency_trend
        .merge(
            customer_month_statistics,
            on="Customer ID",
            how="left",
            validate="one_to_one",
        )
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    summary = {
        "maximum_invoice_date": maximum_invoice_date,
        "cutoff_date": cutoff_date,
        "cutoff_source": cutoff_source,
        "prediction_window_days": 90,

        "historical_transaction_count": len(
            frequency_transactions
        ),

        "customer_count": (
            customer_frequency_trend[
                "Customer ID"
            ].nunique()
        ),

        "observed_calendar_month_count": (
            monthly_frequency[
                "Purchase_Month"
            ].nunique()
        ),

        "minimum_observed_purchase_months": int(
            customer_frequency_trend[
                "Observed_Purchase_Months"
            ].min()
        ),

        "median_observed_purchase_months": float(
            customer_frequency_trend[
                "Observed_Purchase_Months"
            ].median()
        ),

        "maximum_observed_purchase_months": int(
            customer_frequency_trend[
                "Observed_Purchase_Months"
            ].max()
        ),

        "minimum_analysis_months": int(
            customer_frequency_trend[
                "Analysis_Months"
            ].min()
        ),

        "maximum_analysis_months": int(
            customer_frequency_trend[
                "Analysis_Months"
            ].max()
        ),

        "minimum_frequency_trend": float(
            customer_frequency_trend[
                "Purchase_Frequency_Trend"
            ].min()
        ),

        "median_frequency_trend": float(
            customer_frequency_trend[
                "Purchase_Frequency_Trend"
            ].median()
        ),

        "maximum_frequency_trend": float(
            customer_frequency_trend[
                "Purchase_Frequency_Trend"
            ].max()
        ),

        "customers_with_single_purchase_month": int(
            (
                customer_frequency_trend[
                    "Observed_Purchase_Months"
                ] == 1
            ).sum()
        ),
    }

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    validation = {
        "customer_count_valid": (
            customer_frequency_trend[
                "Customer ID"
            ].notna().all()
        ),

        "no_duplicate_customer_ids": (
            not customer_frequency_trend[
                "Customer ID"
            ].duplicated().any()
        ),

        "frequency_values_valid": (
            (
                monthly_frequency[
                    "Monthly_Purchase_Frequency"
                ] >= 0
            ).all()
        ),

        "trend_values_valid": (
            customer_frequency_trend[
                "Purchase_Frequency_Trend"
            ].notna().all()
        ),

        "analysis_months_valid": (
            (
                customer_frequency_trend[
                    "Analysis_Months"
                ] > 0
            ).all()
        ),

        "cutoff_valid": (
            frequency_transactions[
                "InvoiceDate"
            ].max()
            <= cutoff_date
        ),
    }

    validation["passed"] = all(
        validation.values()
    )

    return {
        "summary": summary,
        "monthly_frequency": monthly_frequency,
        "customer_analysis": customer_frequency_trend,
        "transactions": frequency_transactions,
        "validation": validation,
    }

#---- 4.2 Create purchase frequency trend feature ----------------
def create_purchase_frequency_trend_features(
    df_cleaned: pd.DataFrame,
    cutoff_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Create customer-level Purchase Frequency Trend features.

    Purchase Frequency Trend
    ------------------------
    Linear slope of monthly distinct purchase-order frequency
    over time.

    Zero-purchase months between the customer's first purchase
    month and the point-in-time cutoff are included.

    Interpretation
    --------------
    Positive trend:
        Increasing purchase frequency.

    Negative trend:
        Decreasing purchase frequency.

    Near-zero trend:
        Relatively stable purchase frequency.

    Customers with fewer than two observed purchase months:
        Trend is assigned 0.0 because a meaningful temporal
        slope cannot be estimated from a single observed
        purchase month.

    Cutoff behavior
    ---------------
    If cutoff_date is None:
        Automatically calculate:

            maximum InvoiceDate - 90 days

    If cutoff_date is supplied:
        Use the supplied cutoff date.

    Parameters
    ----------
    df_cleaned : pd.DataFrame
        Cleaned transaction-level dataframe.

    cutoff_date : pd.Timestamp or None, default=None
        Point-in-time cutoff date.

    Returns
    -------
    pd.DataFrame
        Customer-level dataframe containing:

        Customer ID
        Purchase_Frequency_Trend
    """

    required_columns = {
        "Customer ID",
        "Invoice",
        "InvoiceDate",
        "Transaction Type",
        "Customer ID Missing Flag",
        "Include In Sales Analysis",
    }

    missing_columns = (
        required_columns - set(df_cleaned.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    working_df = df_cleaned.copy()

    # ---------------------------------------------------------
    # Standardize InvoiceDate
    # ---------------------------------------------------------

    working_df["InvoiceDate"] = pd.to_datetime(
        working_df["InvoiceDate"],
        errors="raise",
    )

    # ---------------------------------------------------------
    # Determine point-in-time cutoff
    # ---------------------------------------------------------

    maximum_invoice_date = (
        working_df["InvoiceDate"].max()
    )

    if pd.isna(maximum_invoice_date):
        raise ValueError(
            "InvoiceDate contains no valid dates."
        )

    if cutoff_date is None:

        cutoff_date = (
            maximum_invoice_date
            - pd.Timedelta(days=90)
        )

    else:

        cutoff_date = pd.Timestamp(
            cutoff_date
        )

    if cutoff_date > maximum_invoice_date:
        raise ValueError(
            "cutoff_date cannot be later than "
            "the maximum InvoiceDate."
        )

    # ---------------------------------------------------------
    # Select qualifying historical purchases
    # ---------------------------------------------------------

    frequency_transactions = working_df[
        (working_df["InvoiceDate"] <= cutoff_date)
        & (
            working_df["Transaction Type"]
            == "Purchase"
        )
        & (
            working_df["Customer ID Missing Flag"]
            == 0
        )
        & (
            working_df["Include In Sales Analysis"]
            == True
        )
    ].copy()

    if frequency_transactions.empty:
        raise ValueError(
            "No qualifying purchase transactions "
            "found before or on the supplied cutoff date."
        )

    # ---------------------------------------------------------
    # Create purchase month
    # ---------------------------------------------------------

    frequency_transactions[
        "Purchase_Month"
    ] = (
        frequency_transactions["InvoiceDate"]
        .dt.to_period("M")
    )

    # ---------------------------------------------------------
    # Monthly distinct purchase-order frequency
    # ---------------------------------------------------------

    monthly_purchase_frequency = (
        frequency_transactions
        .groupby(
            [
                "Customer ID",
                "Purchase_Month",
            ]
        )["Invoice"]
        .nunique()
        .reset_index(
            name="Monthly_Purchase_Frequency"
        )
    )

    # ---------------------------------------------------------
    # Create complete monthly timeline
    #
    # Every month from the customer's first purchase
    # month through the cutoff month is included.
    # Months without purchases receive frequency = 0.
    # ---------------------------------------------------------

    cutoff_month = (
        cutoff_date.to_period("M")
    )

    customer_first_month = (
        monthly_purchase_frequency
        .groupby("Customer ID")[
            "Purchase_Month"
        ]
        .min()
    )

    complete_monthly_frequency = []

    for customer_id, first_month in (
        customer_first_month.items()
    ):

        customer_month_range = pd.period_range(
            start=first_month,
            end=cutoff_month,
            freq="M",
        )

        customer_month_df = pd.DataFrame(
            {
                "Customer ID": customer_id,
                "Purchase_Month": (
                    customer_month_range
                ),
            }
        )

        complete_monthly_frequency.append(
            customer_month_df
        )

    complete_monthly_frequency = pd.concat(
        complete_monthly_frequency,
        ignore_index=True,
    )

    monthly_frequency = (
        complete_monthly_frequency
        .merge(
            monthly_purchase_frequency,
            on=[
                "Customer ID",
                "Purchase_Month",
            ],
            how="left",
            validate="one_to_one",
        )
    )

    # ---------------------------------------------------------
    # Fill months without purchases with zero
    # ---------------------------------------------------------

    monthly_frequency[
        "Monthly_Purchase_Frequency"
    ] = (
        monthly_frequency[
            "Monthly_Purchase_Frequency"
        ]
        .fillna(0)
        .astype(int)
    )

    # ---------------------------------------------------------
    # Create ordered numeric month index
    # ---------------------------------------------------------

    monthly_frequency["Month_Index"] = (
        monthly_frequency[
            "Purchase_Month"
        ].astype("int64")
    )

    # ---------------------------------------------------------
    # Calculate customer-level trend
    # ---------------------------------------------------------

    def calculate_frequency_slope(group):

        observed_purchase_months = (
            group[
                "Monthly_Purchase_Frequency"
            ] > 0
        ).sum()

        # A trend cannot be estimated from
        # only one observed purchase month.
        if observed_purchase_months < 2:
            return 0.0

        x = group[
            "Month_Index"
        ].to_numpy(dtype=float)

        y = group[
            "Monthly_Purchase_Frequency"
        ].to_numpy(dtype=float)

        denominator = (
            (x - x.mean()) ** 2
        ).sum()

        if denominator == 0:
            return 0.0

        slope = (
            (
                (x - x.mean())
                * (y - y.mean())
            ).sum()
            / denominator
        )

        return float(slope)

    purchase_frequency_trend = (
        monthly_frequency
        .groupby("Customer ID")
        .apply(
            calculate_frequency_slope,
            include_groups=False,
        )
        .rename(
            "Purchase_Frequency_Trend"
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # Round final feature
    # ---------------------------------------------------------

    purchase_frequency_trend[
        "Purchase_Frequency_Trend"
    ] = (
        purchase_frequency_trend[
            "Purchase_Frequency_Trend"
        ].round(6)
    )

    # ---------------------------------------------------------
    # Structural validation
    # ---------------------------------------------------------

    if (
        purchase_frequency_trend[
            "Customer ID"
        ].isna().any()
    ):
        raise ValueError(
            "Purchase Frequency Trend contains "
            "missing Customer IDs."
        )

    if (
        purchase_frequency_trend[
            "Customer ID"
        ].duplicated().any()
    ):
        raise ValueError(
            "Purchase Frequency Trend contains "
            "duplicate Customer IDs."
        )

    if (
        purchase_frequency_trend[
            "Purchase_Frequency_Trend"
        ].isna().any()
    ):
        raise ValueError(
            "Purchase Frequency Trend contains "
            "missing values."
        )

    return purchase_frequency_trend

#---- 4.3 Validate purchase frequency trend feature ----------------
def validate_purchase_frequency_trend_features(
    purchase_frequency_trend_features: pd.DataFrame,
) -> dict:
    """
    Validate customer-level Purchase Frequency Trend features.

    Validation Rules
    ----------------
    1. Required columns must exist.
    2. Customer ID must be unique and non-missing.
    3. Purchase Frequency Trend must be present.
    4. Purchase Frequency Trend must contain finite values.
    """

    required_columns = [
        "Customer ID",
        "Purchase_Frequency_Trend",
    ]

    validation_results = {}

    # ---------------------------------------------------------
    # Required columns
    # ---------------------------------------------------------

    missing_columns = [
        column
        for column in required_columns
        if column not in purchase_frequency_trend_features.columns
    ]

    validation_results["required_columns_present"] = (
        len(missing_columns) == 0
    )

    validation_results["missing_columns"] = missing_columns

    if missing_columns:
        validation_results["passed"] = False
        return validation_results

    # ---------------------------------------------------------
    # Customer ID validation
    # ---------------------------------------------------------

    validation_results["customer_count"] = len(
        purchase_frequency_trend_features
    )

    validation_results["has_customers"] = (
        len(purchase_frequency_trend_features) > 0
    )

    validation_results["missing_customer_ids"] = int(
        purchase_frequency_trend_features[
            "Customer ID"
        ]
        .isna()
        .sum()
    )

    validation_results["duplicate_customer_ids"] = int(
        purchase_frequency_trend_features[
            "Customer ID"
        ]
        .duplicated()
        .sum()
    )

    validation_results["unique_customer_ids"] = (
        purchase_frequency_trend_features[
            "Customer ID"
        ]
        .nunique()
    )

    validation_results["customer_ids_valid"] = (
        validation_results["missing_customer_ids"] == 0
        and validation_results["duplicate_customer_ids"] == 0
    )

    # ---------------------------------------------------------
    # Purchase Frequency Trend validation
    # ---------------------------------------------------------

    trend_values = (
        purchase_frequency_trend_features[
            "Purchase_Frequency_Trend"
        ]
    )

    validation_results["missing_trend_values"] = int(
        trend_values.isna().sum()
    )

    validation_results["infinite_trend_values"] = int(
        (~np.isfinite(trend_values)).sum()
    )

    validation_results["trend_values_valid"] = (
        validation_results["missing_trend_values"] == 0
        and validation_results["infinite_trend_values"] == 0
    )

    # ---------------------------------------------------------
    # Overall validation
    # ---------------------------------------------------------

    validation_checks = [
        validation_results["has_customers"],
        validation_results["required_columns_present"],
        validation_results["customer_ids_valid"],
        validation_results["trend_values_valid"],
    ]

    validation_results["passed"] = all(
        validation_checks
    )

    return validation_results



################################################################
#---------- 5. time between consecutive purchases ----------------
################################################################

#---- 5.1 Calculate time between consecutive purchases ----------------
def analyze_interpurchase_variance_basis(
    df_cleaned: pd.DataFrame,
    cutoff_date: pd.Timestamp | None = None,
) -> dict:
    """
    Analyze the basis for Interpurchase Variance.

    Interpurchase Variance
    ----------------------
    Variance of the number of days between consecutive
    qualifying customer purchase orders.

    Purchase events are defined at the invoice/order level.
    Multiple transaction lines belonging to the same invoice
    therefore represent one purchase event.

    Variance is calculated using population variance (ddof=0).

    This means:
        - No purchase interval: variance = 0
        - One purchase interval: variance = 0
        - Two or more intervals: actual variance is calculated

    Cutoff behavior
    ---------------
    If cutoff_date is None:
        Automatically calculate:

            maximum InvoiceDate - 90 days

    If cutoff_date is supplied:
        Use the supplied cutoff date.
    """

    required_columns = {
        "Customer ID",
        "Invoice",
        "InvoiceDate",
        "Transaction Type",
        "Customer ID Missing Flag",
        "Include In Sales Analysis",
    }

    missing_columns = (
        required_columns - set(df_cleaned.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    working_df = df_cleaned.copy()

    # ---------------------------------------------------------
    # Standardize InvoiceDate
    # ---------------------------------------------------------

    working_df["InvoiceDate"] = pd.to_datetime(
        working_df["InvoiceDate"],
        errors="raise",
    )

    # ---------------------------------------------------------
    # Determine cutoff
    # ---------------------------------------------------------

    maximum_invoice_date = (
        working_df["InvoiceDate"].max()
    )

    if pd.isna(maximum_invoice_date):
        raise ValueError(
            "InvoiceDate contains no valid dates."
        )

    if cutoff_date is None:

        cutoff_date = (
            maximum_invoice_date
            - pd.Timedelta(days=90)
        )

        cutoff_source = "Automatically calculated"

    else:

        cutoff_date = pd.Timestamp(cutoff_date)
        cutoff_source = "User supplied"

    if cutoff_date > maximum_invoice_date:
        raise ValueError(
            "cutoff_date cannot be later than "
            "the maximum InvoiceDate."
        )

    # ---------------------------------------------------------
    # Select qualifying historical purchases
    # ---------------------------------------------------------

    purchase_transactions = working_df[
        (working_df["InvoiceDate"] <= cutoff_date)
        & (
            working_df["Transaction Type"]
            == "Purchase"
        )
        & (
            working_df["Customer ID Missing Flag"]
            == 0
        )
        & (
            working_df["Include In Sales Analysis"]
            == True
        )
    ].copy()

    if purchase_transactions.empty:
        raise ValueError(
            "No qualifying purchase transactions "
            "found before or on the supplied cutoff date."
        )

    # ---------------------------------------------------------
    # Create invoice-level purchase events
    # ---------------------------------------------------------

    invoice_purchases = (
        purchase_transactions
        .groupby(
            [
                "Customer ID",
                "Invoice",
            ],
            as_index=False,
        )
        .agg(
            Purchase_Date=(
                "InvoiceDate",
                "min",
            )
        )
    )

    # ---------------------------------------------------------
    # Sort purchase events chronologically
    # ---------------------------------------------------------

    invoice_purchases = (
        invoice_purchases
        .sort_values(
            [
                "Customer ID",
                "Purchase_Date",
                "Invoice",
            ]
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Calculate interpurchase intervals
    # ---------------------------------------------------------

    invoice_purchases[
        "Previous_Purchase_Date"
    ] = (
        invoice_purchases
        .groupby("Customer ID")[
            "Purchase_Date"
        ]
        .shift(1)
    )

    invoice_purchases[
        "Interpurchase_Days"
    ] = (
        invoice_purchases["Purchase_Date"]
        - invoice_purchases[
            "Previous_Purchase_Date"
        ]
    ).dt.total_seconds() / (
        24 * 60 * 60
    )

    # ---------------------------------------------------------
    # Keep actual intervals
    # ---------------------------------------------------------

    interpurchase_intervals = (
        invoice_purchases[
            invoice_purchases[
                "Interpurchase_Days"
            ].notna()
        ]
        .copy()
    )

    # ---------------------------------------------------------
    # Customer order statistics
    # ---------------------------------------------------------

    customer_order_counts = (
        invoice_purchases
        .groupby("Customer ID")
        .agg(
            Number_of_Orders=(
                "Invoice",
                "nunique",
            ),
            First_Purchase_Date=(
                "Purchase_Date",
                "min",
            ),
            Last_Purchase_Date=(
                "Purchase_Date",
                "max",
            ),
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # Customer interval statistics
    #
    # Population variance is used: ddof=0.
    # ---------------------------------------------------------

    customer_interval_statistics = (
        interpurchase_intervals
        .groupby("Customer ID")
        .agg(
            Number_of_Intervals=(
                "Interpurchase_Days",
                "count",
            ),
            Mean_Interpurchase_Days=(
                "Interpurchase_Days",
                "mean",
            ),
            Median_Interpurchase_Days=(
                "Interpurchase_Days",
                "median",
            ),
            Minimum_Interpurchase_Days=(
                "Interpurchase_Days",
                "min",
            ),
            Maximum_Interpurchase_Days=(
                "Interpurchase_Days",
                "max",
            ),
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # Calculate population variance
    # ---------------------------------------------------------

    interval_variance = (
        interpurchase_intervals
        .groupby("Customer ID")[
            "Interpurchase_Days"
        ]
        .var(ddof=0)
        .rename("Interpurchase_Variance")
        .reset_index()
    )

    customer_interval_statistics = (
        customer_interval_statistics
        .merge(
            interval_variance,
            on="Customer ID",
            how="left",
            validate="one_to_one",
        )
    )

    # ---------------------------------------------------------
    # Merge customers with no intervals
    # ---------------------------------------------------------

    customer_analysis = (
        customer_order_counts
        .merge(
            customer_interval_statistics,
            on="Customer ID",
            how="left",
            validate="one_to_one",
        )
    )

    # Customers with no interval have no observed variability.
    customer_analysis[
        "Number_of_Intervals"
    ] = (
        customer_analysis[
            "Number_of_Intervals"
        ]
        .fillna(0)
        .astype(int)
    )

    customer_analysis[
        "Interpurchase_Variance"
    ] = (
        customer_analysis[
            "Interpurchase_Variance"
        ]
        .fillna(0.0)
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    summary = {
        "maximum_invoice_date": maximum_invoice_date,
        "cutoff_date": cutoff_date,
        "cutoff_source": cutoff_source,
        "prediction_window_days": 90,

        "historical_transaction_count": len(
            purchase_transactions
        ),

        "customer_count": (
            customer_analysis[
                "Customer ID"
            ].nunique()
        ),

        "distinct_invoice_count": (
            invoice_purchases[
                "Invoice"
            ].nunique()
        ),

        "total_interpurchase_intervals": len(
            interpurchase_intervals
        ),

        "minimum_orders_per_customer": int(
            customer_analysis[
                "Number_of_Orders"
            ].min()
        ),

        "median_orders_per_customer": float(
            customer_analysis[
                "Number_of_Orders"
            ].median()
        ),

        "maximum_orders_per_customer": int(
            customer_analysis[
                "Number_of_Orders"
            ].max()
        ),

        "customers_with_no_interpurchase_interval": int(
            (
                customer_analysis[
                    "Number_of_Intervals"
                ] == 0
            ).sum()
        ),

        "customers_with_one_interpurchase_interval": int(
            (
                customer_analysis[
                    "Number_of_Intervals"
                ] == 1
            ).sum()
        ),

        "customers_with_variance": int(
            (
                customer_analysis[
                    "Number_of_Intervals"
                ] >= 2
            ).sum()
        ),

        "minimum_interpurchase_days": float(
            interpurchase_intervals[
                "Interpurchase_Days"
            ].min()
        ),

        "median_interpurchase_days": float(
            interpurchase_intervals[
                "Interpurchase_Days"
            ].median()
        ),

        "maximum_interpurchase_days": float(
            interpurchase_intervals[
                "Interpurchase_Days"
            ].max()
        ),

        "minimum_interpurchase_variance": float(
            customer_analysis[
                "Interpurchase_Variance"
            ].min()
        ),

        "median_interpurchase_variance": float(
            customer_analysis[
                "Interpurchase_Variance"
            ].median()
        ),

        "maximum_interpurchase_variance": float(
            customer_analysis[
                "Interpurchase_Variance"
            ].max()
        ),
    }

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    validation = {
        "customer_count_valid": (
            customer_analysis[
                "Customer ID"
            ].notna().all()
        ),

        "no_duplicate_customer_ids": (
            not customer_analysis[
                "Customer ID"
            ].duplicated().any()
        ),

        "order_counts_valid": (
            (
                customer_analysis[
                    "Number_of_Orders"
                ] > 0
            ).all()
        ),

        "interpurchase_intervals_valid": (
            (
                interpurchase_intervals[
                    "Interpurchase_Days"
                ] >= 0
            ).all()
        ),

        "variance_values_non_negative": (
            (
                customer_analysis[
                    "Interpurchase_Variance"
                ] >= 0
            ).all()
        ),

        "cutoff_valid": (
            purchase_transactions[
                "InvoiceDate"
            ].max()
            <= cutoff_date
        ),
    }

    validation["passed"] = all(
        validation.values()
    )

    return {
        "summary": summary,
        "invoice_purchases": invoice_purchases,
        "interpurchase_intervals": (
            interpurchase_intervals
        ),
        "customer_analysis": customer_analysis,
        "validation": validation,
    }

#---- 5.2 Create time between consecutive purchases feature ----------------
def create_interpurchase_variance_features(
    df_cleaned: pd.DataFrame,
    cutoff_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Create customer-level Interpurchase Variance features.

    Interpurchase Variance
    ----------------------
    Population variance of the number of days between
    consecutive qualifying customer purchase orders.

    Purchase events are defined at the invoice/order level.
    Multiple transaction lines belonging to the same invoice
    therefore represent one purchase event.

    Variance behavior
    -----------------
    No purchase interval:
        Variance = 0.0

    One purchase interval:
        Variance = 0.0

    Two or more purchase intervals:
        Population variance of the interpurchase intervals.

    Cutoff behavior
    ---------------
    If cutoff_date is None:
        Automatically calculate:

            maximum InvoiceDate - 90 days

    If cutoff_date is supplied:
        Use the supplied cutoff date.

    Returns
    -------
    pd.DataFrame
        Customer-level dataframe containing:

        Customer ID
        Interpurchase_Variance
    """

    required_columns = {
        "Customer ID",
        "Invoice",
        "InvoiceDate",
        "Transaction Type",
        "Customer ID Missing Flag",
        "Include In Sales Analysis",
    }

    missing_columns = (
        required_columns - set(df_cleaned.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    working_df = df_cleaned.copy()

    # ---------------------------------------------------------
    # Standardize InvoiceDate
    # ---------------------------------------------------------

    working_df["InvoiceDate"] = pd.to_datetime(
        working_df["InvoiceDate"],
        errors="raise",
    )

    # ---------------------------------------------------------
    # Determine point-in-time cutoff
    # ---------------------------------------------------------

    maximum_invoice_date = (
        working_df["InvoiceDate"].max()
    )

    if pd.isna(maximum_invoice_date):
        raise ValueError(
            "InvoiceDate contains no valid dates."
        )

    if cutoff_date is None:

        cutoff_date = (
            maximum_invoice_date
            - pd.Timedelta(days=90)
        )

    else:

        cutoff_date = pd.Timestamp(
            cutoff_date
        )

    if cutoff_date > maximum_invoice_date:
        raise ValueError(
            "cutoff_date cannot be later than "
            "the maximum InvoiceDate."
        )

    # ---------------------------------------------------------
    # Select qualifying historical purchases
    # ---------------------------------------------------------

    purchase_transactions = working_df[
        (working_df["InvoiceDate"] <= cutoff_date)
        & (
            working_df["Transaction Type"]
            == "Purchase"
        )
        & (
            working_df["Customer ID Missing Flag"]
            == 0
        )
        & (
            working_df["Include In Sales Analysis"]
            == True
        )
    ].copy()

    if purchase_transactions.empty:
        raise ValueError(
            "No qualifying purchase transactions "
            "found before or on the supplied cutoff date."
        )

    # ---------------------------------------------------------
    # Create invoice-level purchase events
    #
    # Multiple transaction lines within the same invoice
    # represent one purchase event.
    # ---------------------------------------------------------

    invoice_purchases = (
        purchase_transactions
        .groupby(
            [
                "Customer ID",
                "Invoice",
            ],
            as_index=False,
        )
        .agg(
            Purchase_Date=(
                "InvoiceDate",
                "min",
            )
        )
    )

    # ---------------------------------------------------------
    # Sort purchase events chronologically
    # ---------------------------------------------------------

    invoice_purchases = (
        invoice_purchases
        .sort_values(
            [
                "Customer ID",
                "Purchase_Date",
                "Invoice",
            ]
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Calculate previous purchase date
    # ---------------------------------------------------------

    invoice_purchases[
        "Previous_Purchase_Date"
    ] = (
        invoice_purchases
        .groupby("Customer ID")[
            "Purchase_Date"
        ]
        .shift(1)
    )

    # ---------------------------------------------------------
    # Calculate interpurchase interval in days
    # ---------------------------------------------------------

    invoice_purchases[
        "Interpurchase_Days"
    ] = (
        invoice_purchases["Purchase_Date"]
        - invoice_purchases[
            "Previous_Purchase_Date"
        ]
    ).dt.total_seconds() / (
        24 * 60 * 60
    )

    # ---------------------------------------------------------
    # Keep customers' actual purchase intervals
    # ---------------------------------------------------------

    interpurchase_intervals = (
        invoice_purchases[
            invoice_purchases[
                "Interpurchase_Days"
            ].notna()
        ]
        .copy()
    )

    # ---------------------------------------------------------
    # Calculate population variance
    #
    # ddof=0 means:
    #   0 intervals → no interval → final variance 0
    #   1 interval  → variance 0
    #   2+ intervals → actual population variance
    # ---------------------------------------------------------

    customer_variance = (
        interpurchase_intervals
        .groupby("Customer ID")[
            "Interpurchase_Days"
        ]
        .var(ddof=0)
        .rename(
            "Interpurchase_Variance"
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # Create complete customer population
    #
    # This ensures customers with only one purchase order
    # are also retained in the final feature table.
    # ---------------------------------------------------------

    customer_population = (
        invoice_purchases[
            ["Customer ID"]
        ]
        .drop_duplicates()
    )

    interpurchase_variance_features = (
        customer_population
        .merge(
            customer_variance,
            on="Customer ID",
            how="left",
            validate="one_to_one",
        )
    )

    # Customers without an interpurchase interval receive 0.
    interpurchase_variance_features[
        "Interpurchase_Variance"
    ] = (
        interpurchase_variance_features[
            "Interpurchase_Variance"
        ]
        .fillna(0.0)
        .round(6)
    )

    # ---------------------------------------------------------
    # Structural validation
    # ---------------------------------------------------------

    if (
        interpurchase_variance_features[
            "Customer ID"
        ].isna().any()
    ):
        raise ValueError(
            "Interpurchase Variance features contain "
            "missing Customer IDs."
        )

    if (
        interpurchase_variance_features[
            "Customer ID"
        ].duplicated().any()
    ):
        raise ValueError(
            "Interpurchase Variance features contain "
            "duplicate Customer IDs."
        )

    if (
        interpurchase_variance_features[
            "Interpurchase_Variance"
        ].isna().any()
    ):
        raise ValueError(
            "Interpurchase Variance contains "
            "missing values."
        )

    if (
        interpurchase_variance_features[
            "Interpurchase_Variance"
        ] < 0
    ).any():
        raise ValueError(
            "Interpurchase Variance contains "
            "negative values."
        )

    return interpurchase_variance_features

#---- 5.3 Validate time between consecutive purchases feature ----------------
def validate_interpurchase_variance_features(
    interpurchase_variance_features: pd.DataFrame,
) -> dict:
    """
    Validate customer-level Interpurchase Variance features.

    Validation Rules
    ----------------
    1. Required columns must exist.
    2. Customer ID must be unique and non-missing.
    3. Interpurchase Variance must be present.
    4. Interpurchase Variance must be finite.
    5. Interpurchase Variance must be non-negative.
    """

    required_columns = [
        "Customer ID",
        "Interpurchase_Variance",
    ]

    validation_results = {}

    # ---------------------------------------------------------
    # Required columns
    # ---------------------------------------------------------

    missing_columns = [
        column
        for column in required_columns
        if column not in interpurchase_variance_features.columns
    ]

    validation_results["required_columns_present"] = (
        len(missing_columns) == 0
    )

    validation_results["missing_columns"] = missing_columns

    if missing_columns:
        validation_results["passed"] = False
        return validation_results

    # ---------------------------------------------------------
    # Customer ID validation
    # ---------------------------------------------------------

    validation_results["customer_count"] = len(
        interpurchase_variance_features
    )

    validation_results["has_customers"] = (
        len(interpurchase_variance_features) > 0
    )

    validation_results["missing_customer_ids"] = int(
        interpurchase_variance_features[
            "Customer ID"
        ]
        .isna()
        .sum()
    )

    validation_results["duplicate_customer_ids"] = int(
        interpurchase_variance_features[
            "Customer ID"
        ]
        .duplicated()
        .sum()
    )

    validation_results["unique_customer_ids"] = (
        interpurchase_variance_features[
            "Customer ID"
        ]
        .nunique()
    )

    validation_results["customer_ids_valid"] = (
        validation_results["missing_customer_ids"] == 0
        and validation_results["duplicate_customer_ids"] == 0
    )

    # ---------------------------------------------------------
    # Interpurchase Variance validation
    # ---------------------------------------------------------

    variance_values = (
        interpurchase_variance_features[
            "Interpurchase_Variance"
        ]
    )

    validation_results["missing_variance_values"] = int(
        variance_values.isna().sum()
    )

    validation_results["infinite_variance_values"] = int(
        (~np.isfinite(variance_values)).sum()
    )

    validation_results["negative_variance_values"] = int(
        (variance_values < 0).sum()
    )

    validation_results["variance_values_valid"] = (
        validation_results["missing_variance_values"] == 0
        and validation_results["infinite_variance_values"] == 0
        and validation_results["negative_variance_values"] == 0
    )

    # ---------------------------------------------------------
    # Overall validation
    # ---------------------------------------------------------

    validation_checks = [
        validation_results["has_customers"],
        validation_results["required_columns_present"],
        validation_results["customer_ids_valid"],
        validation_results["variance_values_valid"],
    ]

    validation_results["passed"] = all(
        validation_checks
    )

    return validation_results




###############################################################
#---------- 6. Seasonal purchase concentration ----------------
################################################################

#---- 6.1 Calculate seasonal purchase concentration feature ----------------
def analyze_seasonal_purchase_concentration_basis(
    df_cleaned: pd.DataFrame,
    cutoff_date: pd.Timestamp | None = None,
) -> dict:
    """
    Analyze the basis for Seasonal Purchase Concentration.

    Seasonal Purchase Concentration
    --------------------------------
    Measures how concentrated a customer's purchase orders
    are across calendar months.

    Purchase events are defined at the invoice/order level.
    Multiple transaction lines belonging to the same invoice
    therefore represent one purchase event.

    Analysis metrics
    -----------------
    Seasonal_Concentration_HHI:
        Sum of squared monthly purchase shares.

        Lower value:
            Purchases are distributed across more months.

        Higher value:
            Purchases are concentrated in fewer months.

    Peak_Month_Share:
        Largest proportion of the customer's purchase orders
        occurring in any single calendar month.

    Cutoff behavior
    ---------------
    If cutoff_date is None:
        Automatically calculate:

            maximum InvoiceDate - 90 days

    If cutoff_date is supplied:
        Use the supplied cutoff date.

    The production feature definition is not finalized by
    this function. The purpose is to inspect the seasonal
    behavior and determine the appropriate production rule.
    """

    required_columns = {
        "Customer ID",
        "Invoice",
        "InvoiceDate",
        "Transaction Type",
        "Customer ID Missing Flag",
        "Include In Sales Analysis",
    }

    missing_columns = (
        required_columns - set(df_cleaned.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    working_df = df_cleaned.copy()

    # ---------------------------------------------------------
    # Standardize InvoiceDate
    # ---------------------------------------------------------

    working_df["InvoiceDate"] = pd.to_datetime(
        working_df["InvoiceDate"],
        errors="raise",
    )

    # ---------------------------------------------------------
    # Determine point-in-time cutoff
    # ---------------------------------------------------------

    maximum_invoice_date = (
        working_df["InvoiceDate"].max()
    )

    if pd.isna(maximum_invoice_date):
        raise ValueError(
            "InvoiceDate contains no valid dates."
        )

    if cutoff_date is None:

        cutoff_date = (
            maximum_invoice_date
            - pd.Timedelta(days=90)
        )

        cutoff_source = "Automatically calculated"

    else:

        cutoff_date = pd.Timestamp(cutoff_date)

        cutoff_source = "User supplied"

    if cutoff_date > maximum_invoice_date:
        raise ValueError(
            "cutoff_date cannot be later than "
            "the maximum InvoiceDate."
        )

    # ---------------------------------------------------------
    # Select qualifying historical purchases
    # ---------------------------------------------------------

    purchase_transactions = working_df[
        (working_df["InvoiceDate"] <= cutoff_date)
        & (
            working_df["Transaction Type"]
            == "Purchase"
        )
        & (
            working_df["Customer ID Missing Flag"]
            == 0
        )
        & (
            working_df["Include In Sales Analysis"]
            == True
        )
    ].copy()

    if purchase_transactions.empty:
        raise ValueError(
            "No qualifying purchase transactions "
            "found before or on the supplied cutoff date."
        )

    # ---------------------------------------------------------
    # Create invoice-level purchase events
    # ---------------------------------------------------------

    invoice_purchases = (
        purchase_transactions
        .groupby(
            [
                "Customer ID",
                "Invoice",
            ],
            as_index=False,
        )
        .agg(
            Purchase_Date=(
                "InvoiceDate",
                "min",
            )
        )
    )

    # ---------------------------------------------------------
    # Extract calendar month
    # ---------------------------------------------------------

    invoice_purchases["Purchase_Month"] = (
        invoice_purchases["Purchase_Date"]
        .dt.month
    )

    invoice_purchases["Purchase_Month_Name"] = (
        invoice_purchases["Purchase_Date"]
        .dt.month_name()
    )

    # ---------------------------------------------------------
    # Monthly order counts
    # ---------------------------------------------------------

    monthly_orders = (
        invoice_purchases
        .groupby(
            [
                "Customer ID",
                "Purchase_Month",
                "Purchase_Month_Name",
            ]
        )["Invoice"]
        .nunique()
        .reset_index(
            name="Monthly_Order_Count"
        )
    )

    # ---------------------------------------------------------
    # Customer total orders
    # ---------------------------------------------------------

    customer_total_orders = (
        invoice_purchases
        .groupby("Customer ID")["Invoice"]
        .nunique()
        .rename("Total_Orders")
        .reset_index()
    )

    # ---------------------------------------------------------
    # Calculate monthly purchase share
    # ---------------------------------------------------------

    monthly_orders = (
        monthly_orders
        .merge(
            customer_total_orders,
            on="Customer ID",
            how="left",
            validate="many_to_one",
        )
    )

    monthly_orders[
        "Monthly_Purchase_Share"
    ] = (
        monthly_orders[
            "Monthly_Order_Count"
        ]
        / monthly_orders[
            "Total_Orders"
        ]
    )

    # ---------------------------------------------------------
    # Calculate customer-level concentration metrics
    # ---------------------------------------------------------

    customer_seasonality = (
        monthly_orders
        .groupby("Customer ID")
        .agg(
            Active_Calendar_Months=(
                "Purchase_Month",
                "nunique",
            ),
            Total_Orders=(
                "Total_Orders",
                "first",
            ),
            Peak_Month_Share=(
                "Monthly_Purchase_Share",
                "max",
            ),
            Seasonal_Concentration_HHI=(
                "Monthly_Purchase_Share",
                lambda shares: (
                    shares ** 2
                ).sum(),
            ),
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # Identify customer's peak purchase month
    # ---------------------------------------------------------

    peak_months = (
        monthly_orders
        .sort_values(
            [
                "Customer ID",
                "Monthly_Purchase_Share",
                "Purchase_Month",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .drop_duplicates(
            subset=["Customer ID"],
            keep="first",
        )
        [
            [
                "Customer ID",
                "Purchase_Month_Name",
            ]
        ]
        .rename(
            columns={
                "Purchase_Month_Name":
                    "Peak_Purchase_Month"
            }
        )
    )

    customer_seasonality = (
        customer_seasonality
        .merge(
            peak_months,
            on="Customer ID",
            how="left",
            validate="one_to_one",
        )
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    summary = {
        "maximum_invoice_date": maximum_invoice_date,
        "cutoff_date": cutoff_date,
        "cutoff_source": cutoff_source,
        "prediction_window_days": 90,

        "historical_transaction_count": len(
            purchase_transactions
        ),

        "customer_count": (
            customer_seasonality[
                "Customer ID"
            ].nunique()
        ),

        "distinct_invoice_count": (
            invoice_purchases[
                "Invoice"
            ].nunique()
        ),

        "minimum_orders_per_customer": int(
            customer_seasonality[
                "Total_Orders"
            ].min()
        ),

        "median_orders_per_customer": float(
            customer_seasonality[
                "Total_Orders"
            ].median()
        ),

        "maximum_orders_per_customer": int(
            customer_seasonality[
                "Total_Orders"
            ].max()
        ),

        "minimum_active_calendar_months": int(
            customer_seasonality[
                "Active_Calendar_Months"
            ].min()
        ),

        "median_active_calendar_months": float(
            customer_seasonality[
                "Active_Calendar_Months"
            ].median()
        ),

        "maximum_active_calendar_months": int(
            customer_seasonality[
                "Active_Calendar_Months"
            ].max()
        ),

        "customers_with_one_order": int(
            (
                customer_seasonality[
                    "Total_Orders"
                ] == 1
            ).sum()
        ),

        "customers_with_one_active_month": int(
            (
                customer_seasonality[
                    "Active_Calendar_Months"
                ] == 1
            ).sum()
        ),

        "minimum_peak_month_share": float(
            customer_seasonality[
                "Peak_Month_Share"
            ].min()
        ),

        "median_peak_month_share": float(
            customer_seasonality[
                "Peak_Month_Share"
            ].median()
        ),

        "maximum_peak_month_share": float(
            customer_seasonality[
                "Peak_Month_Share"
            ].max()
        ),

        "minimum_seasonal_hhi": float(
            customer_seasonality[
                "Seasonal_Concentration_HHI"
            ].min()
        ),

        "median_seasonal_hhi": float(
            customer_seasonality[
                "Seasonal_Concentration_HHI"
            ].median()
        ),

        "maximum_seasonal_hhi": float(
            customer_seasonality[
                "Seasonal_Concentration_HHI"
            ].max()
        ),
    }

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    validation = {
        "customer_count_valid": (
            customer_seasonality[
                "Customer ID"
            ].notna().all()
        ),

        "no_duplicate_customer_ids": (
            not customer_seasonality[
                "Customer ID"
            ].duplicated().any()
        ),

        "order_counts_valid": (
            (
                customer_seasonality[
                    "Total_Orders"
                ] > 0
            ).all()
        ),

        "monthly_shares_valid": (
            (
                monthly_orders[
                    "Monthly_Purchase_Share"
                ] >= 0
            ).all()
            and
            (
                monthly_orders[
                    "Monthly_Purchase_Share"
                ] <= 1
            ).all()
        ),

        "peak_month_share_valid": (
            (
                customer_seasonality[
                    "Peak_Month_Share"
                ] >= 0
            ).all()
            and
            (
                customer_seasonality[
                    "Peak_Month_Share"
                ] <= 1
            ).all()
        ),

        "seasonal_hhi_valid": (
            (
                customer_seasonality[
                    "Seasonal_Concentration_HHI"
                ] > 0
            ).all()
            and
            (
                customer_seasonality[
                    "Seasonal_Concentration_HHI"
                ] <= 1
            ).all()
        ),

        "cutoff_valid": (
            purchase_transactions[
                "InvoiceDate"
            ].max()
            <= cutoff_date
        ),
    }

    validation["passed"] = all(
        validation.values()
    )

    return {
        "summary": summary,
        "invoice_purchases": invoice_purchases,
        "monthly_orders": monthly_orders,
        "customer_analysis": customer_seasonality,
        "validation": validation,
    }

#---- 6.2 Create seasonal purchase concentration feature ----------------
def create_seasonal_purchase_concentration_features(
    df_cleaned: pd.DataFrame,
    cutoff_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Create customer-level Seasonal Purchase Concentration.

    Seasonal Purchase Concentration
    --------------------------------
    Calculated as the Herfindahl-Hirschman Index (HHI) of
    the customer's purchase-order distribution across
    calendar months.

    Formula
    -------
    HHI = sum(monthly_purchase_share ** 2)

    Interpretation
    --------------
    Lower HHI:
        Purchases are distributed across multiple months.

    Higher HHI:
        Purchases are concentrated in fewer months.

    HHI = 1:
        All observed purchase orders occurred in one
        calendar month.

    Important
    ---------
    A high HHI for a customer with very few orders indicates
    concentration in the observed history, but does not by
    itself prove that the customer is genuinely seasonal.

    Purchase events are defined at the invoice/order level.
    Multiple transaction lines belonging to the same invoice
    therefore represent one purchase event.

    Cutoff behavior
    ---------------
    If cutoff_date is None:
        Automatically calculate:

            maximum InvoiceDate - 90 days

    If cutoff_date is supplied:
        Use the supplied cutoff date.

    Returns
    -------
    pd.DataFrame
        Customer-level dataframe containing:

        Customer ID
        Seasonal_Purchase_Concentration
    """

    required_columns = {
        "Customer ID",
        "Invoice",
        "InvoiceDate",
        "Transaction Type",
        "Customer ID Missing Flag",
        "Include In Sales Analysis",
    }

    missing_columns = (
        required_columns - set(df_cleaned.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    working_df = df_cleaned.copy()

    # ---------------------------------------------------------
    # Standardize InvoiceDate
    # ---------------------------------------------------------

    working_df["InvoiceDate"] = pd.to_datetime(
        working_df["InvoiceDate"],
        errors="raise",
    )

    # ---------------------------------------------------------
    # Determine point-in-time cutoff
    # ---------------------------------------------------------

    maximum_invoice_date = (
        working_df["InvoiceDate"].max()
    )

    if pd.isna(maximum_invoice_date):
        raise ValueError(
            "InvoiceDate contains no valid dates."
        )

    if cutoff_date is None:

        cutoff_date = (
            maximum_invoice_date
            - pd.Timedelta(days=90)
        )

    else:

        cutoff_date = pd.Timestamp(
            cutoff_date
        )

    if cutoff_date > maximum_invoice_date:
        raise ValueError(
            "cutoff_date cannot be later than "
            "the maximum InvoiceDate."
        )

    # ---------------------------------------------------------
    # Select qualifying historical purchases
    # ---------------------------------------------------------

    purchase_transactions = working_df[
        (working_df["InvoiceDate"] <= cutoff_date)
        & (
            working_df["Transaction Type"]
            == "Purchase"
        )
        & (
            working_df["Customer ID Missing Flag"]
            == 0
        )
        & (
            working_df["Include In Sales Analysis"]
            == True
        )
    ].copy()

    if purchase_transactions.empty:
        raise ValueError(
            "No qualifying purchase transactions "
            "found before or on the supplied cutoff date."
        )

    # ---------------------------------------------------------
    # Create invoice-level purchase events
    # ---------------------------------------------------------

    invoice_purchases = (
        purchase_transactions
        .groupby(
            [
                "Customer ID",
                "Invoice",
            ],
            as_index=False,
        )
        .agg(
            Purchase_Date=(
                "InvoiceDate",
                "min",
            )
        )
    )

    # ---------------------------------------------------------
    # Extract calendar month
    # ---------------------------------------------------------

    invoice_purchases["Purchase_Month"] = (
        invoice_purchases["Purchase_Date"]
        .dt.month
    )

    # ---------------------------------------------------------
    # Count orders per customer per calendar month
    # ---------------------------------------------------------

    monthly_orders = (
        invoice_purchases
        .groupby(
            [
                "Customer ID",
                "Purchase_Month",
            ]
        )["Invoice"]
        .nunique()
        .reset_index(
            name="Monthly_Order_Count"
        )
    )

    # ---------------------------------------------------------
    # Calculate total orders per customer
    # ---------------------------------------------------------

    customer_total_orders = (
        invoice_purchases
        .groupby("Customer ID")["Invoice"]
        .nunique()
        .rename("Total_Orders")
        .reset_index()
    )

    # ---------------------------------------------------------
    # Calculate monthly purchase share
    # ---------------------------------------------------------

    monthly_orders = (
        monthly_orders
        .merge(
            customer_total_orders,
            on="Customer ID",
            how="left",
            validate="many_to_one",
        )
    )

    monthly_orders[
        "Monthly_Purchase_Share"
    ] = (
        monthly_orders[
            "Monthly_Order_Count"
        ]
        / monthly_orders[
            "Total_Orders"
        ]
    )

    # ---------------------------------------------------------
    # Calculate HHI
    # ---------------------------------------------------------

    seasonal_concentration = (
        monthly_orders
        .groupby("Customer ID")[
            "Monthly_Purchase_Share"
        ]
        .apply(
            lambda shares: (
                shares ** 2
            ).sum()
        )
        .rename(
            "Seasonal_Purchase_Concentration"
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # Round final feature
    # ---------------------------------------------------------

    seasonal_concentration[
        "Seasonal_Purchase_Concentration"
    ] = (
        seasonal_concentration[
            "Seasonal_Purchase_Concentration"
        ]
        .round(6)
    )

    # ---------------------------------------------------------
    # Structural validation
    # ---------------------------------------------------------

    if (
        seasonal_concentration[
            "Customer ID"
        ].isna().any()
    ):
        raise ValueError(
            "Seasonal Purchase Concentration "
            "contains missing Customer IDs."
        )

    if (
        seasonal_concentration[
            "Customer ID"
        ].duplicated().any()
    ):
        raise ValueError(
            "Seasonal Purchase Concentration "
            "contains duplicate Customer IDs."
        )

    if (
        seasonal_concentration[
            "Seasonal_Purchase_Concentration"
        ].isna().any()
    ):
        raise ValueError(
            "Seasonal Purchase Concentration "
            "contains missing values."
        )

    if (
        seasonal_concentration[
            "Seasonal_Purchase_Concentration"
        ] < 0
    ).any():
        raise ValueError(
            "Seasonal Purchase Concentration "
            "contains negative values."
        )

    if (
        seasonal_concentration[
            "Seasonal_Purchase_Concentration"
        ] > 1
    ).any():
        raise ValueError(
            "Seasonal Purchase Concentration "
            "contains values greater than 1."
        )

    return seasonal_concentration

#---- 6.3 Validate seasonal purchase concentration feature ----------------
def validate_seasonal_purchase_concentration_features(
    seasonal_purchase_concentration_features: pd.DataFrame,
) -> dict:
    """
    Validate customer-level Seasonal Purchase Concentration.

    Validation Rules
    ----------------
    1. Required columns must exist.
    2. Customer ID must be unique and non-missing.
    3. Seasonal Purchase Concentration must be present.
    4. Values must be finite.
    5. HHI must be between 0 and 1.
    """

    required_columns = [
        "Customer ID",
        "Seasonal_Purchase_Concentration",
    ]

    validation_results = {}

    # ---------------------------------------------------------
    # Required columns
    # ---------------------------------------------------------

    missing_columns = [
        column
        for column in required_columns
        if column not in seasonal_purchase_concentration_features.columns
    ]

    validation_results["required_columns_present"] = (
        len(missing_columns) == 0
    )

    validation_results["missing_columns"] = missing_columns

    if missing_columns:
        validation_results["passed"] = False
        return validation_results

    # ---------------------------------------------------------
    # Customer ID validation
    # ---------------------------------------------------------

    validation_results["customer_count"] = len(
        seasonal_purchase_concentration_features
    )

    validation_results["has_customers"] = (
        len(seasonal_purchase_concentration_features) > 0
    )

    validation_results["missing_customer_ids"] = int(
        seasonal_purchase_concentration_features[
            "Customer ID"
        ]
        .isna()
        .sum()
    )

    validation_results["duplicate_customer_ids"] = int(
        seasonal_purchase_concentration_features[
            "Customer ID"
        ]
        .duplicated()
        .sum()
    )

    validation_results["unique_customer_ids"] = (
        seasonal_purchase_concentration_features[
            "Customer ID"
        ]
        .nunique()
    )

    validation_results["customer_ids_valid"] = (
        validation_results["missing_customer_ids"] == 0
        and validation_results["duplicate_customer_ids"] == 0
    )

    # ---------------------------------------------------------
    # Seasonal concentration validation
    # ---------------------------------------------------------

    concentration_values = (
        seasonal_purchase_concentration_features[
            "Seasonal_Purchase_Concentration"
        ]
    )

    validation_results["missing_concentration_values"] = int(
        concentration_values.isna().sum()
    )

    validation_results["infinite_concentration_values"] = int(
        (~np.isfinite(concentration_values)).sum()
    )

    validation_results["below_zero_values"] = int(
        (concentration_values < 0).sum()
    )

    validation_results["above_one_values"] = int(
        (concentration_values > 1).sum()
    )

    validation_results["concentration_values_valid"] = (
        validation_results["missing_concentration_values"] == 0
        and validation_results["infinite_concentration_values"] == 0
        and validation_results["below_zero_values"] == 0
        and validation_results["above_one_values"] == 0
    )

    # ---------------------------------------------------------
    # Overall validation
    # ---------------------------------------------------------

    validation_checks = [
        validation_results["has_customers"],
        validation_results["required_columns_present"],
        validation_results["customer_ids_valid"],
        validation_results["concentration_values_valid"],
    ]

    validation_results["passed"] = all(
        validation_checks
    )

    return validation_results



###############################################################
#---------- 7. product category affinity vectors --------------
###############################################################

#---- 7.1 Analyze product category affinity vectors ----------------
def analyze_product_taxonomy(df_cleaned):
    """
    Analyse and assign product categories for the genuine product universe.

    Purpose
    -------
    Build a StockCode-level product taxonomy that will later be used to
    construct the customer-level Product Category Affinity Vector.

    Design principles
    -----------------
    1. `Description` is the actual product-description field.
    2. `Description Standardized` is a Boolean quality flag and is NOT used
       as the product description.
    3. Verified administrative/non-product StockCodes are excluded from
       the taxonomy but remain in the cleaned transaction dataset.
    4. Missing descriptions are never guessed.
    5. Strong product-type keywords have priority over weak descriptive words.
    6. Multiple category matches are retained for audit.
    7. Exactly one primary category is assigned to each genuine StockCode.
    8. Products that cannot be confidently classified remain
       `Other / Unclassified`.
    9. One row is produced per genuine StockCode.
    10. Generic descriptive words such as LIGHT, PINK, GIFT, FLOWER,
        ROSE and DESIGN are not allowed to dominate actual product types.

    Returns
    -------
    dict
        Complete taxonomy analysis package.
    """

    # =========================================================
    # 1. Validate required columns
    # =========================================================

    required_columns = [
        "StockCode",
        "Description",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df_cleaned.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    working_df = df_cleaned.copy()

    # =========================================================
    # 2. Standardise StockCode
    # =========================================================

    working_df["StockCode_Key"] = (
        working_df["StockCode"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    working_df = working_df[
        working_df["StockCode_Key"].notna()
        & (working_df["StockCode_Key"] != "")
    ].copy()

    # =========================================================
    # 3. Use the actual Description column
    #
    # IMPORTANT:
    # Description Standardized is a Boolean flag in the current
    # dataframe and therefore is NOT used for taxonomy.
    # =========================================================

    working_df["Taxonomy_Description"] = (
        working_df["Description"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    working_df["Taxonomy_Description"] = (
        working_df["Taxonomy_Description"]
        .replace("", pd.NA)
    )

    # =========================================================
    # 4. Verified administrative/non-product StockCodes
    # =========================================================

    NON_PRODUCT_STOCK_CODES = {
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

    # =========================================================
    # 5. Exclude administrative codes from product taxonomy
    # =========================================================

    product_transactions = working_df[
        ~working_df["StockCode_Key"].isin(
            NON_PRODUCT_STOCK_CODES
        )
    ].copy()

    # =========================================================
    # 6. Build one canonical description per StockCode
    # =========================================================

    def select_canonical_description(group):

        descriptions = (
            group["Taxonomy_Description"]
            .dropna()
            .astype("string")
            .str.strip()
        )

        descriptions = descriptions[
            descriptions != ""
        ]

        if descriptions.empty:
            return "NO DESCRIPTION"

        return (
            descriptions
            .value_counts()
            .index[0]
        )

    product_catalog = (
        product_transactions
        .groupby("StockCode_Key")
        .apply(
            select_canonical_description,
            include_groups=False,
        )
        .rename("Description")
        .reset_index()
    )

    # =========================================================
    # 7. Description availability
    # =========================================================

    product_catalog["Description_Present"] = (
        product_catalog["Description"]
        != "NO DESCRIPTION"
    )

    # =========================================================
    # 8. FINAL PRODUCT TAXONOMY RULES
    #
    # Strong = actual product type/function
    # Weak   = supporting/contextual description
    #
    # Strong matches always take precedence over weak matches.
    # =========================================================

    category_rules = {

        # -----------------------------------------------------
        # HOME FRAGRANCE
        # -----------------------------------------------------

        "Home Fragrance": {
            "strong": [
                "INCENSE",
                "INCENSE CONE",
                "INCENSE CONES",
                "POTPOURRI",
                "DIFFUSER",
                "OIL BURNER",
                "AROMA BURNER",
                "FRAGRANCE BURNER",
            ],
            "weak": [
                "FRAGRANCE",
                "AROMA",
                "SCENTED",
            ],
        },

        # -----------------------------------------------------
        # LIGHTING & CANDLES
        # -----------------------------------------------------

        "Lighting & Candles": {
            "strong": [
                "CHANDELIER",
                "CHANDELIERS",
                "CANDLE",
                "CANDLES",
                "TEA LIGHT",
                "TEA LIGHTS",
                "TEALIGHT",
                "TEALIGHTS",
                "T LIGHT",
                "T LIGHTS",
                "T-LIGHT",
                "T-LIGHTS",
                "T/ LIGHT",
                "T/LIGHT",
                "T/LIGHTS",
                "TEALIGHT HOLDER",
                "TEA LIGHT HOLDER",
                "T-LIGHT HOLDER",
                "LIGHT HOLDER",
                "LANTERN",
                "LAMPSHADE",
                "LAMP SHADE",
                "LAMP",
                "LIGHT FITTING",
            ],
            "weak": [
                "LIGHT",
                "LIGHTING",
            ],
        },

        # -----------------------------------------------------
        # STATIONERY & PAPER
        # -----------------------------------------------------

        "Stationery & Paper": {
         "strong": [
        # Writing instruments
         "PENCIL",
         "PENCILS",
         "PEN",
         "PENS",
        "BALL PEN",
         "BALLPEN",
         "FELT PEN",
         "MARKER",
         "MARKERS",
         "PHOTO ALBUM",
         "PHOTO ALBUMS",
         "PHOTO ALBUM COVER",
         "HOLE PUNCH",
         "HOLE PUNCHES",
         "PUNCH",
         "STENCIL",
         "STENCILS",
         "CALCULATOR",
         "CALCULATORS",
        "MEMO PAD",
        "MEMO PADS",

        # Notebooks, books and paper products
        "SKETCHBOOK",
        "SKETCH BOOK",
        "JOURNAL",
        "NOTEBOOK",
        "NOTEPAD",
        "DIARY",
        "ADDRESS BOOK",
        "ADDRESS BOOKS",
        "BOOKCOVER",
        "BOOK COVER",
        "POCKET BOOK",
        "POCKETBOOK",
        "PHOTO ALBUM",
        "PHOTO ALBUMS",
        "ALBUM",
        "ALBUMS",
        "STATIONERY",
        "PAGE PHOTO",
        "PAGE PHOTOS",
        "P'WEIGHT",
        "PAPERWEIGHT",
        "P'WEIGHTS",

        # Paper accessories
        "STICKER",
        "STICKERS",
        "ENVELOPE",
        "ENVELOPES",
        "RULER",
        "RULERS",
        "ERASER",
        "ERASERS",
        "POSTCARD",
        "POSTCARDS",
        "BOOKMARK",
        "BOOKMARKS",
        "PAPER CLIP",
        "PAPER CLIPS",
        "CLIPBOARD",
        "CLIPBOARDS",

        # Writing / office supplies
        "WRITING SET",
        "WRITING SETS",
        "WRITING",
        "STAMP",
        "STAMPS",
        "INK PAD",
        "INKPAD",
        "INK PADS",

        # Tape and wrapping-related stationery
        "TAPE",
        "TAPES",
        "WRAP",
        "WRAPS",
        "WRAPPING",
        "WRAPPING PAPER",
        "WRAPPING TAPE",
        "BOOKCOVER TAPE",
        "BOOK COVER TAPE",
        "VIPPASSPORT COVER",
        "VIP PASSPORT COVER",
        "PASSPORT COVERS",

        # Craft and cutting tools
        "SCISSOR",
        "SCISSORS",
        "FOLDING SCISSOR",
        "FOLDING SCISSORS",
        "CRAFT SCISSOR",
        "CRAFT SCISSORS",
        "PAPER SCISSOR",
        "PAPER SCISSORS",
        "CRAFT KNIFE",
        "CRAFT KNIVES",

        # Chalkboards / display-writing products
        "CHALKBOARD",
        "CHALK BOARD",
        "CHALKBOARDS",
        "EASEL",
        "EASELS",
        "WHITEBOARD",
        "WHITE BOARD",
        ],

         "weak": [
          "PAPER",
          "PAPERS",
          "CARD",
          "CARDS",
          "CRAFT",
          "CRAFTS",
          "DESK",
          "DESK ACCESSORY",
          "DESK ACCESSORIES",
        ],
    },

        # -----------------------------------------------------
        # JEWELRY & FASHION ACCESSORIES
        # -----------------------------------------------------

        "Jewelry & Fashion Accessories": {
            "strong": [
                "BADGE",
                "BADGES",
                "CLIP",
                "CLIPS",
                "NECKLACE",
                "BRACELET",
                "EARRING",
                "EARRINGS",
                "BROOCH",
                "JEWELRY",
                "JEWELLERY",
                "CHARM",
                "CHARMS",
                "HAIR CLIP",
                "HAIR BAND",
                "HAIR ACCESSOR",
                "SOCKS",
                "GLOVES",
                "SLIPPERS",
                "SILK FAN",
                "HAND FAN",
                "FAN",
                "PARASOL",
                "SUNSHADE",
                "SHAWL",
                "SCARF",
                "HAT",
                "HATS",
                "CAP",
                "CAPS",
                "BELT",
                "BELTS",
                "PURSE",
                "FASHION ACCESSORY",
                "FASHION ACCESSORIES",
                "KEY-CHAIN",
                "KEY CHAINS",
                "KEY-CHAINS",
                "KEYCHAIN",
                "KEYCHAINS",
                "HAIRSLIDE",
                "HAIR SLIDE",
                "HAIR SLIDES",
                "EAR MUFF",
                "EAR MUFFS",




            ],
            "weak": [
                "RING",
                "FASHION",
                "ACCESSORY",
                "ACCESSORIES",
            ],
        },

        # -----------------------------------------------------
        # BAGS & TRAVEL ACCESSORIES
        # -----------------------------------------------------

        "Bags & Travel Accessories": {
            "strong": [
                "SHOPPER",
                "SHOPPERS",
                "LUGGAGE TAG",
                "PASSPORT COVER",
                "PASSPORT HOLDER",
                "TRAVEL TAG",
                "TRAVEL ACCESSOR",
                "LUGGAGE",
                "PASSPORT",
                "UMBRELLA",
                "SHOPPING BAG",
                "TOTE BAG",
                "TRAVEL BAG",
                "WEEKEND BAG",
                "COSMETIC BAG",
                "MAKEUP BAG",
                "WASH BAG",
                "WASHBAG",
                "PURSE",
                "POUCH",
                "WALLET",
                "HANDBAG",

            ],
            "weak": [
                "BAG",
                "BAGS",
            ],
        },

        # -----------------------------------------------------
        # KITCHEN & DINING
        # -----------------------------------------------------

        "Kitchen & Dining": {
            "strong": [
                "SALAD SERVER",
                "SALAD SERVERS",
                "LATTICE CHARGER",
                "WINE GOBLET",
                "WINE GOBLETS",
                "PEG PLACE",
                "MUG",
                "MUGS",
                "CUP",
                "CUPS",
                "SAUCER",
                "PLATE",
                "PLATES",
                "BOWL",
                "BOWLS",
                "DISH",
                "DISHES",
                "CUTLERY",
                "SPOON",
                "SPOONS",
                "FORK",
                "FORKS",
                "KNIFE",
                "KNIVES",
                "KITCHEN",
                "TEAPOT",
                "TEA POT",
                "KETTLE",
                "JUG",
                "GLASS",
                "GLASSES",
                "BOTTLE",
                "CAKE STAND",
                "CAKESTAND",
                "TRAY",
                "TRAYS",
                "PAN",
                "PANS",
                "FRYING PAN",
                "FOOD COVER",
                "TEA TOWEL",
                "TEA TOWELS",
                "MEASURING SPOON",
                "MEASURING SPOONS",
                "MEASURING CUP",
                "MEASURING CUPS",
                "OVEN GLOVE",
                "OVEN GLOVES",
                "APRON",
                "APRONS",
                "SALT SHAKER",
                "PEPPER SHAKER",
                "SALT AND PEPPER",
                "COASTER",
                "COASTERS",
                "CHOPSTICK",
                "CHOPSTICKS",
                "SERVING",
                "SERVING SPOON",
                "SERVING TRAY",
            ],
            "weak": [
                "CAKE",
                "TEA",
                "FOOD",
                "DINING",
                "PANTRY",
            ],
        },

        # -----------------------------------------------------
        # STORAGE & ORGANIZATION
        # -----------------------------------------------------

        "Storage & Organization": {
            "strong": [
                "STORAGE",
                "STORAGE BOX",
                "STORAGE CASE",
                "BOX",
                "BOXES",
                "BASKET",
                "BASKETS",
                "HOLDER",
                "HOLDERS",
                "CABINET",
                "DRAWER",
                "DRAWERS",
                "ORGANIZER",
                "ORGANIZERS",
                "ORGANISER",
                "ORGANISERS",
                "TRINKET BOX",
                "MINI CASE",
                "TRINKET CASE",
                "STORAGE CASE",
                "PENCIL CASE",
                "JEWELLERY BOX",
                "JEWELRY BOX",
                "COSMETIC CASE",
                "STORAGE BASKET",
                "STORAGE BASKETS",
                "MAGNETIC MEMO PAD",
                "MAGNETIC TO DO LIST",
                "FIRST AID TIN",
                "WOODEN SHELF",
                "WOODEN SHELVES",
                "DRESSER",
                "DRESSERS",
            ],
            "weak": [
                "CASE",
                "CASES",
                "ORGANIZATION",
                "ORGANISATION",
            ],
        },

        # -----------------------------------------------------
        # GARDEN & OUTDOOR
        # -----------------------------------------------------

        "Garden & Outdoor": {
            "strong": [
                "GARDEN",
                "GARDENING",
                "GARDEN TOOL",
                "GARDEN TOOLS",
                "PLANTER",
                "PLANT POT",
                "PLANT POT",
                "FLOWER POT",
                "WATERING CAN",
                "BIRDHOUSE",
                "BIRD HOUSE",
                "BIRD FEEDER",
                "GARDEN TOOL",
                "GARDENING TOOL",
                "PATIO",
                "OUTDOOR",
                "PICNIC",
            ],
            "weak": [
                "PLANT",
                "PLANTS",
                "FLOWER",
                "FLOWERS",
            ],
        },

        # -----------------------------------------------------
        # BATH & PERSONAL CARE
        # -----------------------------------------------------

        "Bath & Personal Care": {
            "strong": [
                "FLANNEL",
                "BATH",
                "BATHROOM",
                "SOAP",
                "SPONGE",
                "TOWEL",
                "TOWELS",
                "COSMETIC",
                "COSMETICS",
                "BEAUTY",
                "PERSONAL CARE",
                "TOILET",
                "TOILETRY",
                "TOILETRIES",
                "BATH MAT",
                "BATH MAT",
                "WASH CLOTH",
                "FACE CLOTH",
            ],
            "weak": [
                "CARE",
            ],
        },

        # -----------------------------------------------------
        # TOYS & GAMES
        # -----------------------------------------------------

        "Toys & Games": {
            "strong": [
                "SPACE CADET",
                "SPACE FROG",
                "SPACE OWL",
                "NINJA RABBIT",
                "TOY",
                "TOYS",
                "GAME",
                "GAMES",
                "PUZZLE",
                "PUZZLES",
                "PADDED MOBILES",
                "PADDED MOBILE",
                "DOLL",
                "DOLLS",
                "PLAYING CARDS",
                "CARD GAME",
                "INFLATABLE",
                "CHILDREN",
                "CHILDRENS",
                "BUILDING BLOCK",
                "BUILDING BLOCKS",
                "TEDDY",
                "TEDDIES",
                "SOFT TOY",
                "SOFT TOYS",
                "YO-YO",
                "YOYO",
                "RATTLE",
                "RATTLES",
                "SOFT MOBILE",
                "SOFT MOBILES",
                "BABY MOBILE",
                "BABY MOBILES",
                "TOY MOBILE",
                "TOY MOBILES",
            ],
            "weak": [
                "RUBBER",
                "PLAY",
            ],
        },

        # -----------------------------------------------------
        # PARTY & CELEBRATION
        # -----------------------------------------------------

        "Party & Celebration": {
            "strong": [
                "GARLAND",
                "PARTY",
                "BALLOON",
                "BALLOONS",
                "BUNTING",
                "CONFETTI",
                "PARTY BAG",
                "PARTY BAGS",
                "PARTY SUPPL",
                "NAPKIN",
                "NAPKINS",
                "CAKE DECORATION",
                "CAKE DECORATIONS",
                "PARTY DECORATION",
                "PARTY DECORATIONS",
                "PARTY HAT",
                "PARTY HATS",
                "PARTY CANDLE",
                "PARTY CANDLES",
            ],
            "weak": [
                "CELEBRATION",
            ],
        },

        # -----------------------------------------------------
        # GIFTS & SEASONAL
        # -----------------------------------------------------

        "Gifts & Seasonal": {
            "strong": [
                "FEATHER TREE",
                "FEATHER TREES",
                "BUTTERFLY STRING",
                "JEWEL",
                "JEWELS",
                "IPOD",
                "CHRISTMAS",
                "XMAS",
                "EASTER",
                "HALLOWEEN",
                "VALENTINE",
                "VALENTINES",
                "CHRISTMAS TREE",
                "TREE DECORATION",
                "TREE DECORATIONS",
                "ORNAMENT",
                "ORNAMENTS",
                "CHRISTMAS DECORATION",
                "CHRISTMAS DECORATIONS",
            ],
            "weak": [
                "GIFT",
                "GIFTS",
                "GIFT WRAP",
                "SEASONAL",
            ],
        },

        # -----------------------------------------------------
        # FOOD & CONSUMABLES
        # -----------------------------------------------------

        "Food & Consumables": {
            "strong": [
                "CHOCOLATE",
                "CHOCOLATES",
                "SWEET",
                "SWEETS",
                "BISCUIT",
                "BISCUITS",
                "COFFEE",
                "CANDY",
                "FOOD",
                "JAM",
                "MARMALADE",
                "HONEY",
                "SAUCE",
                "SAUCES",
                "PRESERVE",
                "PRESERVES",
            ],
            "weak": [
                "TEA",
            ],
        },

        # -----------------------------------------------------
        # HOME DECOR
        # -----------------------------------------------------

        "Home Decor": {
            "strong": [
                "DOORSTOP",
                "DOOR HANGER",
                "TABLECLOTH",
                "FILLER PAD",
                "FILLER PADS",
                "QUILTED",
                "TOADSTOOL",
                "TOADSTOOLS",
                "HANGGING",
                "HAND PAINTED"
                "hookS",
                "HOOK",
                "HOOKS",
                "CERAMIC BALL",
                "CERAMIC BALLS",
                "CERAMIC FIGURE",
                "BAR STOOL",
                "BARSTOOL",
                "STOOL",
                "STOOLS",
                "CUSHION",
                "CUSHIONS",
                "CUSHION COVER",
                "CUSHION COVERS",
                "GNOME",
                "GNOMES",
                "STANDING GNOME",
                "METAL GNOME",
                "PAINTED METAL",
                "PAPERWEIGHT",
                "PAPER WEIGHT",
                "PAPERWEIGHTS",
                "DECORATIVE PAPERWEIGHT",
                "DECORATIVE FIGURE",
                "DECORATIVE FIGURES",
                "DECORATIVE OBJECT",
                "DECORATIVE OBJECTS",
                "PHOTO FRAME",
                "PHOTO FRAMES",
                "PHOTOFRAME",
                "PHOTOFRAMES",
                "FRAME",
                "FRAMES",
                "PHOTO ALBUM FRAME",
                "MIRROR",
                "MIRRORS",
                "WALL ART",
                "WALL DECOR",
                "WALL DECORATION",
                "DOOR KNOB",
                "KNOB",
                "KNOBS",
                "FIGURINE",
                "FIGURINES",
                "CLOCK",
                "CLOCKS",
                "WALL CLOCK",
                "DOORMAT",
                "METAL SIGN",
                "SIGN",
                "SIGNS",
                "BELL",
                "BELLS",
                "CHERUB",
                "CHERUBS",
                "DECORATIVE OBJECT",
                "DECORATIVE OBJECTS",
                "ORNAMENTAL",
                "FLORAL",
                "FLORALS",
            ],
            "weak": [
                "DECORATION",
                "DECORATIONS",
                "DECORATIVE",
                "HOME",
                "VINTAGE",
            ],
        },
    }

    # =========================================================
    # 9. Category priority
    # =========================================================

    category_priority = [
        "Home Fragrance",
        "Lighting & Candles",
        "Stationery & Paper",
        "Jewelry & Fashion Accessories",
        "Bags & Travel Accessories",
        "Kitchen & Dining",
        "Storage & Organization",
        "Garden & Outdoor",
        "Bath & Personal Care",
        "Toys & Games",
        "Party & Celebration",
        "Gifts & Seasonal",
        "Food & Consumables",
        "Home Decor",
        "Other / Unclassified",
    ]

    # =========================================================
    # 10. Word-aware keyword matching
    # =========================================================

    def keyword_matches(description, keywords):

        if (
            pd.isna(description)
            or description == "NO DESCRIPTION"
        ):
            return False

        pattern = (
            r"(?<![A-Z0-9])(?:"
            + "|".join(
                re.escape(keyword)
                for keyword in sorted(
                    keywords,
                    key=len,
                    reverse=True,
                )
            )
            + r")(?![A-Z0-9])"
        )

        return bool(
            re.search(
                pattern,
                description,
                flags=re.IGNORECASE,
            )
        )

    # =========================================================
    # 11. Create strong/weak category matches
    # =========================================================

    for category, rules in category_rules.items():

        category_key = (
            category
            .replace(" & ", "_")
            .replace(" ", "_")
        )

        strong_column = (
            f"Strong_{category_key}"
        )

        weak_column = (
            f"Weak_{category_key}"
        )

        match_column = (
            f"Match_{category_key}"
        )

        product_catalog[strong_column] = (
            product_catalog["Description"]
            .apply(
                lambda value:
                keyword_matches(
                    value,
                    rules["strong"],
                )
            )
        )

        product_catalog[weak_column] = (
            product_catalog["Description"]
            .apply(
                lambda value:
                keyword_matches(
                    value,
                    rules["weak"],
                )
            )
        )

        product_catalog[match_column] = (
            product_catalog[strong_column]
            | product_catalog[weak_column]
        )

    # =========================================================
    # 12. Category match counts
    # =========================================================

    match_columns = [
        column
        for column in product_catalog.columns
        if column.startswith("Match_")
    ]

    strong_columns = [
        column
        for column in product_catalog.columns
        if column.startswith("Strong_")
    ]

    product_catalog["Matched_Category_Count"] = (
        product_catalog[match_columns]
        .sum(axis=1)
        .astype(int)
    )

    product_catalog["Strong_Category_Match_Count"] = (
        product_catalog[strong_columns]
        .sum(axis=1)
        .astype(int)
    )

    # =========================================================
    # 13. Primary category assignment
    #
    # Strong product-type matches beat weak/context matches.
    # =========================================================

    def assign_primary_category(row):

        description = row["Description"]

        if (
            pd.isna(description)
            or description == ""
            or description == "NO DESCRIPTION"
        ):
            return "Other / Unclassified"

        # -----------------------------------------------------
        # Strong product-type matches
        # -----------------------------------------------------

        for category in category_priority:

            if category == "Other / Unclassified":
                continue

            category_key = (
                category
                .replace(" & ", "_")
                .replace(" ", "_")
            )

            strong_column = (
                f"Strong_{category_key}"
            )

            if bool(row[strong_column]):
                return category

        # -----------------------------------------------------
        # Weak/context matches
        # -----------------------------------------------------

        for category in category_priority:

            if category == "Other / Unclassified":
                continue

            category_key = (
                category
                .replace(" & ", "_")
                .replace(" ", "_")
            )

            weak_column = (
                f"Weak_{category_key}"
            )

            if bool(row[weak_column]):
                return category

        return "Other / Unclassified"

    product_catalog["Product_Category"] = (
        product_catalog.apply(
            assign_primary_category,
            axis=1,
        )
    )

    # =========================================================
    # 14. Assignment status
    # =========================================================

    product_catalog[
        "Category_Assignment_Status"
    ] = "No_Rule_Match"

    product_catalog.loc[
        product_catalog["Description"]
        == "NO DESCRIPTION",
        "Category_Assignment_Status"
    ] = "Missing_Description"

    product_catalog.loc[
        (
            product_catalog["Description"]
            != "NO DESCRIPTION"
        )
        &
        (
            product_catalog[
                "Matched_Category_Count"
            ]
            > 0
        ),
        "Category_Assignment_Status"
    ] = "Rule_Assigned"

    # =========================================================
    # 15. Multi-match products
    # =========================================================

    multi_match_products = product_catalog[
        product_catalog["Matched_Category_Count"] > 1
    ].copy()

    multi_match_columns = [
        "StockCode_Key",
        "Description",
        "Matched_Category_Count",
        "Strong_Category_Match_Count",
        "Product_Category",
    ] + match_columns

    multi_match_products = (
        multi_match_products[
            multi_match_columns
        ]
        .sort_values(
            by=[
                "Matched_Category_Count",
                "Strong_Category_Match_Count",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # =========================================================
    # 16. Unclassified products
    # =========================================================

    unclassified_products = product_catalog[
        product_catalog["Product_Category"]
        == "Other / Unclassified"
    ].copy()

    unclassified_products = (
        unclassified_products[
            [
                "StockCode_Key",
                "Description",
                "Category_Assignment_Status",
            ]
        ]
        .sort_values(
            by=[
                "Category_Assignment_Status",
                "StockCode_Key",
            ]
        )
        .reset_index(drop=True)
    )

    # =========================================================
    # 17. Category summary
    # =========================================================

    category_summary = (
        product_catalog[
            [
                "Product_Category",
                "StockCode_Key",
            ]
        ]
        .groupby(
            "Product_Category",
            as_index=False,
        )
        .agg(
            Product_Count=(
                "StockCode_Key",
                "nunique",
            )
        )
    )

    category_summary = (
        pd.DataFrame(
            {
                "Product_Category":
                    category_priority
            }
        )
        .merge(
            category_summary,
            on="Product_Category",
            how="left",
        )
        .fillna(
            {
                "Product_Count": 0
            }
        )
    )

    category_summary["Product_Count"] = (
        category_summary["Product_Count"]
        .astype(int)
    )

    total_products = len(
        product_catalog
    )

    category_summary[
        "Product_Percentage"
    ] = (
        category_summary["Product_Count"]
        / total_products
        * 100
    ).round(2)

    # =========================================================
    # 18. Assignment summary
    # =========================================================

    assignment_summary = (
        product_catalog[
            "Category_Assignment_Status"
        ]
        .value_counts()
        .rename_axis(
            "Category_Assignment_Status"
        )
        .reset_index(
            name="Product_Count"
        )
    )

    assignment_summary[
        "Product_Percentage"
    ] = (
        assignment_summary["Product_Count"]
        / total_products
        * 100
    ).round(2)

    # =========================================================
    # 19. Validation
    # =========================================================

    validation = {}

    validation["product_count_is_5120"] = (
        total_products == 5120
    )

    validation["one_row_per_stockcode"] = (
        product_catalog[
            "StockCode_Key"
        ].is_unique
    )

    validation["no_missing_product_category"] = (
        product_catalog[
            "Product_Category"
        ].notna().all()
    )

    validation["administrative_codes_excluded"] = (
        not product_catalog[
            "StockCode_Key"
        ]
        .isin(
            NON_PRODUCT_STOCK_CODES
        )
        .any()
    )

    missing_description_count = int(
        (
            product_catalog[
                "Description"
            ]
            == "NO DESCRIPTION"
        ).sum()
    )

    validation[
        "missing_description_count_valid"
    ] = (
        0
        <= missing_description_count
        <= total_products
    )

    assignment_total = int(
        assignment_summary[
            "Product_Count"
        ].sum()
    )

    validation[
        "assignment_counts_reconcile"
    ] = (
        assignment_total
        == total_products
    )

    category_total = int(
        category_summary[
            "Product_Count"
        ].sum()
    )

    validation[
        "category_counts_reconcile"
    ] = (
        category_total
        == total_products
    )

    validation["passed"] = all(
        validation.values()
    )

    # =========================================================
    # 20. Overall summary
    # =========================================================

    summary = {
        "total_genuine_products":
            total_products,

        "rule_assigned_products": int(
            (
                product_catalog[
                    "Category_Assignment_Status"
                ]
                == "Rule_Assigned"
            ).sum()
        ),

        "missing_description_products": int(
            (
                product_catalog[
                    "Category_Assignment_Status"
                ]
                == "Missing_Description"
            ).sum()
        ),

        "no_rule_match_products": int(
            (
                product_catalog[
                    "Category_Assignment_Status"
                ]
                == "No_Rule_Match"
            ).sum()
        ),

        "multi_match_products": int(
            len(multi_match_products)
        ),

        "category_count":
            len(category_priority),

        "validation_passed":
            validation["passed"],
    }

    # =========================================================
    # 21. Return complete analysis package
    # =========================================================

    return {
        "product_catalog":
            product_catalog,

        "category_summary":
            category_summary,

        "assignment_summary":
            assignment_summary,

        "multi_match_products":
            multi_match_products,

        "unclassified_products":
            unclassified_products,

        "validation":
            validation,

        "category_rules":
            category_rules,

        "category_priority":
            category_priority,

        "non_product_stock_codes":
            NON_PRODUCT_STOCK_CODES,

        "summary":
            summary,
    }

#---- 7.2 Create product category affinity vectors ----------------
def create_product_category_affinity_features(
    df_cleaned: pd.DataFrame,
    cutoff_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Create customer-level Product Category Affinity Vector.

    Product taxonomy is generated automatically using the existing
    analyze_product_taxonomy() function.

    Affinity is calculated as the proportion of a customer's
    qualifying purchase value belonging to each product category.

    Parameters
    ----------
    df_cleaned : pd.DataFrame
        Cleaned transaction-level dataframe.

    cutoff_date : pd.Timestamp | None
        Point-in-time cutoff.

        If None:
            cutoff_date = max(InvoiceDate) - 90 days

        If provided:
            the supplied cutoff date is used.

    Returns
    -------
    pd.DataFrame
        One row per customer containing product category
        affinity features.
    """

    # =========================================================
    # 1. Validate required transaction columns
    # =========================================================

    required_columns = [
        "Customer ID",
        "StockCode",
        "InvoiceDate",
        "Transaction Type",
        "Include In Sales Analysis",
        "Analytical Transaction Value",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df_cleaned.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # =========================================================
    # 2. Create product taxonomy automatically
    # =========================================================

    product_taxonomy_analysis = analyze_product_taxonomy(
        df_cleaned
    )

    product_catalog = (
        product_taxonomy_analysis[
            "product_catalog"
        ]
        .copy()
    )

    category_priority = (
        product_taxonomy_analysis[
            "category_priority"
        ]
        .copy()
    )

    # =========================================================
    # 3. Validate taxonomy output
    # =========================================================

    required_taxonomy_columns = [
        "StockCode_Key",
        "Product_Category",
    ]

    missing_taxonomy_columns = [
        column
        for column in required_taxonomy_columns
        if column not in product_catalog.columns
    ]

    if missing_taxonomy_columns:
        raise ValueError(
            "Product taxonomy is missing required columns: "
            f"{missing_taxonomy_columns}"
        )

    if not category_priority:
        raise ValueError(
            "Product taxonomy contains no categories."
        )

    # =========================================================
    # 4. Create working dataframe
    # =========================================================

    working_df = df_cleaned.copy()

    working_df["StockCode_Key"] = (
        working_df["StockCode"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    working_df["InvoiceDate"] = pd.to_datetime(
        working_df["InvoiceDate"],
        errors="coerce",
    )

    # =========================================================
    # 5. Determine point-in-time cutoff
    # =========================================================

    max_invoice_date = (
        working_df["InvoiceDate"].max()
    )

    if pd.isna(max_invoice_date):
        raise ValueError(
            "InvoiceDate does not contain a valid date."
        )

    if cutoff_date is None:

        cutoff_date = (
            max_invoice_date
            - pd.Timedelta(days=90)
        )

    else:

        cutoff_date = pd.Timestamp(
            cutoff_date
        )

    if cutoff_date > max_invoice_date:
        raise ValueError(
            "cutoff_date cannot be later than "
            "max InvoiceDate."
        )

    # =========================================================
    # 6. Select qualifying customer purchases
    # =========================================================

    customer_transactions = working_df[
        (working_df["InvoiceDate"] <= cutoff_date)
        &
        (working_df["Customer ID"] != 0)
        &
        (working_df["Customer ID"].notna())
        &
        (working_df["Transaction Type"] == "Purchase")
        &
        (working_df["Include In Sales Analysis"] == True)
    ].copy()

    if customer_transactions.empty:
        raise ValueError(
            "No qualifying customer transactions found "
            "before the cutoff date."
        )

    # =========================================================
    # 7. Standardize Customer ID
    # =========================================================

    customer_transactions["Customer ID"] = (
        customer_transactions["Customer ID"]
        .astype("int64")
    )

    # =========================================================
    # 8. Prepare StockCode → Product Category mapping
    # =========================================================

    taxonomy_mapping = (
        product_catalog[
            [
                "StockCode_Key",
                "Product_Category",
            ]
        ]
        .drop_duplicates(
            subset=["StockCode_Key"]
        )
    )

    # =========================================================
    # 9. Join transactions with product taxonomy
    # =========================================================

    customer_transactions = (
        customer_transactions
        .merge(
            taxonomy_mapping,
            on="StockCode_Key",
            how="left",
            validate="many_to_one",
        )
    )

    # =========================================================
    # 10. Assign taxonomy gaps to Other / Unclassified
    # =========================================================

    customer_transactions[
        "Product_Category"
    ] = (
        customer_transactions[
            "Product_Category"
        ]
        .fillna("Other / Unclassified")
    )

    # =========================================================
    # 11. Aggregate purchase value by customer/category
    # =========================================================

    customer_category_value = (
        customer_transactions
        .groupby(
            [
                "Customer ID",
                "Product_Category",
            ],
            as_index=False,
        )
        .agg(
            Category_Purchase_Value=(
                "Analytical Transaction Value",
                "sum",
            )
        )
    )

    # =========================================================
    # 12. Create complete customer × category grid
    # =========================================================

    customers = (
        customer_transactions[
            ["Customer ID"]
        ]
        .drop_duplicates()
    )

    categories = pd.DataFrame(
        {
            "Product_Category": category_priority
        }
    )

    customers["_key"] = 1
    categories["_key"] = 1

    customer_category_grid = (
        customers
        .merge(
            categories,
            on="_key",
            how="outer",
        )
        .drop(
            columns="_key"
        )
    )

    # =========================================================
    # 13. Merge category purchase values
    # =========================================================

    customer_category_grid = (
        customer_category_grid
        .merge(
            customer_category_value,
            on=[
                "Customer ID",
                "Product_Category",
            ],
            how="left",
        )
    )

    customer_category_grid[
        "Category_Purchase_Value"
    ] = (
        customer_category_grid[
            "Category_Purchase_Value"
        ]
        .fillna(0.0)
    )

    # =========================================================
    # 14. Calculate total customer purchase value
    # =========================================================

    customer_category_grid[
        "Customer_Total_Purchase_Value"
    ] = (
        customer_category_grid
        .groupby(
            "Customer ID"
        )[
            "Category_Purchase_Value"
        ]
        .transform("sum")
    )

    # =========================================================
    # 15. Calculate category affinity
    # =========================================================

    customer_category_grid[
        "Product_Category_Affinity"
    ] = (
        customer_category_grid[
            "Category_Purchase_Value"
        ]
        /
        customer_category_grid[
            "Customer_Total_Purchase_Value"
        ]
    )

    # =========================================================
    # 16. Convert categories into feature columns
    # =========================================================

    affinity_features = (
        customer_category_grid
        .pivot(
            index="Customer ID",
            columns="Product_Category",
            values="Product_Category_Affinity",
        )
        .fillna(0.0)
        .reset_index()
    )

    # =========================================================
    # 17. Ensure every taxonomy category exists
    # =========================================================

    for category in category_priority:

        if category not in affinity_features.columns:

            affinity_features[category] = 0.0

    # =========================================================
    # 18. Rename category columns
    # =========================================================

    category_column_map = {
        category: (
            category
            .replace(" & ", "_")
            .replace(" ", "_")
            .replace("/", "_")
        )
        + "_Affinity"
        for category in category_priority
    }

    affinity_features = (
        affinity_features
        .rename(
            columns=category_column_map
        )
    )

    # =========================================================
    # 19. Arrange columns consistently
    # =========================================================

    affinity_columns = [
        category_column_map[category]
        for category in category_priority
    ]

    affinity_features = affinity_features[
        [
            "Customer ID"
        ]
        + affinity_columns
    ]

    # =========================================================
    # 20. Validation
    # =========================================================

    validation = {}

    validation[
        "one_row_per_customer"
    ] = (
        affinity_features[
            "Customer ID"
        ].is_unique
    )

    validation[
        "customer_count"
    ] = (
        len(affinity_features)
        ==
        customer_transactions[
            "Customer ID"
        ].nunique()
    )

    validation[
        "no_missing_customer_id"
    ] = (
        affinity_features[
            "Customer ID"
        ].notna().all()
    )

    validation[
        "no_negative_affinity"
    ] = (
        affinity_features[
            affinity_columns
        ]
        .ge(0)
        .all()
        .all()
    )

    validation[
        "no_affinity_above_one"
    ] = (
        affinity_features[
            affinity_columns
        ]
        .le(1)
        .all()
        .all()
    )

    affinity_sum = (
        affinity_features[
            affinity_columns
        ]
        .sum(axis=1)
    )

    validation[
        "affinity_sums_to_one"
    ] = (
        np.isclose(
            affinity_sum,
            1.0,
            atol=1e-9,
        ).all()
    )

    validation[
        "expected_category_count"
    ] = (
        len(affinity_columns)
        ==
        len(category_priority)
    )

    validation["passed"] = all(
        validation.values()
    )

    # =========================================================
    # 21. Store metadata
    # =========================================================

    affinity_features.attrs[
        "validation"
    ] = validation

    affinity_features.attrs[
        "cutoff_date"
    ] = cutoff_date

    affinity_features.attrs[
        "max_invoice_date"
    ] = max_invoice_date

    affinity_features.attrs[
        "qualifying_transaction_count"
    ] = len(
        customer_transactions
    )

    affinity_features.attrs[
        "customer_count"
    ] = affinity_features[
        "Customer ID"
    ].nunique()

    affinity_features.attrs[
        "category_count"
    ] = len(
        category_priority
    )

    # =========================================================
    # 22. Fail if validation does not pass
    # =========================================================

    if not validation["passed"]:

        failed_checks = [
            key
            for key, value in validation.items()
            if value is False
        ]

        raise ValueError(
            "Product Category Affinity validation failed: "
            f"{failed_checks}"
        )

    # =========================================================
    # 23. Return final feature dataframe
    # =========================================================

    return affinity_features

#---- 7.3 validate product category affinity vectors ----------------
def validate_product_category_affinity_features(
    affinity_features: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    cutoff_date: pd.Timestamp | None = None,
) -> dict:
    """
    Validate customer-level Product Category Affinity features.

    Parameters
    ----------
    affinity_features : pd.DataFrame
        Customer-level Product Category Affinity feature dataframe
        returned by create_product_category_affinity_features().

    df_cleaned : pd.DataFrame
        Cleaned transaction-level dataframe used to independently
        validate the expected customer population and cutoff.

    cutoff_date : pd.Timestamp | None
        Point-in-time cutoff.

        If None:
            cutoff_date = max(InvoiceDate) - 90 days

        If provided:
            the supplied cutoff date is used.

    Returns
    -------
    dict
        Validation results and summary information.
    """

    # =========================================================
    # 1. Validate required transaction columns
    # =========================================================

    required_transaction_columns = [
        "Customer ID",
        "InvoiceDate",
        "Transaction Type",
        "Include In Sales Analysis",
        "Analytical Transaction Value",
    ]

    missing_transaction_columns = [
        column
        for column in required_transaction_columns
        if column not in df_cleaned.columns
    ]

    if missing_transaction_columns:
        raise ValueError(
            "Missing required transaction columns: "
            f"{missing_transaction_columns}"
        )

    # =========================================================
    # 2. Validate required affinity columns
    # =========================================================

    if "Customer ID" not in affinity_features.columns:
        raise ValueError(
            "Affinity dataframe is missing 'Customer ID'."
        )

    affinity_columns = [
        column
        for column in affinity_features.columns
        if column.endswith("_Affinity")
    ]

    if not affinity_columns:
        raise ValueError(
            "No Product Category Affinity columns found."
        )

    # =========================================================
    # 3. Prepare transaction dataframe
    # =========================================================

    working_df = df_cleaned.copy()

    working_df["InvoiceDate"] = pd.to_datetime(
        working_df["InvoiceDate"],
        errors="coerce",
    )

    # =========================================================
    # 4. Determine cutoff automatically
    # =========================================================

    max_invoice_date = (
        working_df["InvoiceDate"].max()
    )

    if pd.isna(max_invoice_date):
        raise ValueError(
            "InvoiceDate does not contain a valid date."
        )

    if cutoff_date is None:

        cutoff_date = (
            max_invoice_date
            - pd.Timedelta(days=90)
        )

    else:

        cutoff_date = pd.Timestamp(
            cutoff_date
        )

    if cutoff_date > max_invoice_date:
        raise ValueError(
            "cutoff_date cannot be later than "
            "max InvoiceDate."
        )

    # =========================================================
    # 5. Reconstruct expected qualifying customer population
    # =========================================================

    qualifying_transactions = working_df[
        (working_df["InvoiceDate"] <= cutoff_date)
        &
        (working_df["Customer ID"] != 0)
        &
        (working_df["Customer ID"].notna())
        &
        (working_df["Transaction Type"] == "Purchase")
        &
        (working_df["Include In Sales Analysis"] == True)
    ].copy()

    expected_customers = (
        qualifying_transactions[
            "Customer ID"
        ]
        .astype("int64")
        .drop_duplicates()
    )

    # =========================================================
    # 6. Validation checks
    # =========================================================

    validation = {}

    # ---------------------------------------------------------
    # Customer ID checks
    # ---------------------------------------------------------

    validation[
        "customer_id_column_present"
    ] = (
        "Customer ID"
        in affinity_features.columns
    )

    validation[
        "no_missing_customer_id"
    ] = (
        affinity_features[
            "Customer ID"
        ].notna().all()
    )

    validation[
        "customer_id_is_unique"
    ] = (
        affinity_features[
            "Customer ID"
        ].is_unique
    )

    # ---------------------------------------------------------
    # Expected customer population
    # ---------------------------------------------------------

    actual_customers = (
        affinity_features[
            "Customer ID"
        ]
        .astype("int64")
        .drop_duplicates()
    )

    validation[
        "customer_count_matches"
    ] = (
        len(actual_customers)
        ==
        len(expected_customers)
    )

    validation[
        "customer_population_matches"
    ] = (
        set(actual_customers)
        ==
        set(expected_customers)
    )

    # ---------------------------------------------------------
    # Category feature checks
    # ---------------------------------------------------------

    validation[
        "affinity_columns_present"
    ] = (
        len(affinity_columns) > 0
    )

    validation[
        "affinity_values_are_numeric"
    ] = (
        affinity_features[
            affinity_columns
        ]
        .apply(
            lambda column:
                pd.api.types.is_numeric_dtype(
                    column
                )
        )
        .all()
    )

    validation[
        "no_missing_affinity_values"
    ] = (
        affinity_features[
            affinity_columns
        ]
        .notna()
        .all()
        .all()
    )

    validation[
        "no_negative_affinity"
    ] = (
        affinity_features[
            affinity_columns
        ]
        .ge(0)
        .all()
        .all()
    )

    validation[
        "no_affinity_above_one"
    ] = (
        affinity_features[
            affinity_columns
        ]
        .le(1)
        .all()
        .all()
    )

    # ---------------------------------------------------------
    # Affinity vector sum
    # ---------------------------------------------------------

    affinity_sum = (
        affinity_features[
            affinity_columns
        ]
        .sum(axis=1)
    )

    validation[
        "affinity_sums_to_one"
    ] = (
        np.isclose(
            affinity_sum,
            1.0,
            atol=1e-9,
        ).all()
    )

    # ---------------------------------------------------------
    # Category count
    # ---------------------------------------------------------

    validation[
        "expected_category_count"
    ] = (
        len(affinity_columns) == 15
    )

    # =========================================================
    # 7. Calculate summary statistics
    # =========================================================

    summary = {
        "customer_count": int(
            affinity_features[
                "Customer ID"
            ].nunique()
        ),

        "expected_customer_count": int(
            expected_customers.nunique()
        ),

        "category_affinity_count": int(
            len(affinity_columns)
        ),

        "max_invoice_date": max_invoice_date,

        "cutoff_date": cutoff_date,

        "qualifying_transaction_count": int(
            len(qualifying_transactions)
        ),

        "minimum_affinity_sum": float(
            affinity_sum.min()
        ),

        "maximum_affinity_sum": float(
            affinity_sum.max()
        ),

        "median_affinity_sum": float(
            affinity_sum.median()
        ),
    }

    # =========================================================
    # 8. Overall validation result
    # =========================================================

    validation["passed"] = all(
        validation.values()
    )

    # =========================================================
    # 9. Failed checks
    # =========================================================

    failed_checks = [
        key
        for key, value in validation.items()
        if value is False
    ]

    # =========================================================
    # 10. Return complete validation package
    # =========================================================

    return {
        "validation": validation,
        "summary": summary,
        "failed_checks": failed_checks,
    }

#---- 7.4 unclassified dataset --------------------------------------
def export_unclassified_products(product_taxonomy_analysis, output_path):
    """
    Export products that currently have no taxonomy rule match.
    Includes description frequency to help identify reusable rules.
    """

    unclassified_df = (
        product_taxonomy_analysis["unclassified_products"]
        .query("Category_Assignment_Status == 'No_Rule_Match'")
        .copy()
    )

    # Calculate how many StockCodes share the same description
    description_frequency = (
        unclassified_df
        .groupby("Description")["StockCode_Key"]
        .transform("count")
    )

    unclassified_df["Description_Frequency"] = description_frequency

    columns_to_export = [
        "StockCode_Key",
        "Description",
        "Description_Frequency",
        "Category_Assignment_Status",
    ]

    columns_to_export = [
        column
        for column in columns_to_export
        if column in unclassified_df.columns
    ]

    unclassified_df = (
        unclassified_df[columns_to_export]
        .sort_values(
            by=["Description_Frequency", "Description"],
            ascending=[False, True],
            na_position="last"
        )
        .reset_index(drop=True)
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    unclassified_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("Unclassified products exported successfully.")
    print(f"File: {output_path}")
    print(f"Rows: {len(unclassified_df):,}")
    print(f"Columns: {list(unclassified_df.columns)}")

    return unclassified_df

###############################################################
#---------- 8. return rate --------------
###############################################################

#----------- 8.1 create return rate --------
def create_return_rate_features(
    df_cleaned: pd.DataFrame,
    cutoff_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Create customer-level Return Rate features.

    Return Rate is defined as:

        Distinct Return/Cancellation Orders
        -----------------------------------
        Distinct Qualifying Purchase Orders

    The feature is calculated using transaction history available
    on or before the point-in-time cutoff.

    Qualifying purchase orders:
        - Customer ID is valid
        - Transaction Type == "Purchase"
        - Include In Sales Analysis == True
        - Quantity > 0
        - InvoiceDate <= cutoff

    Return/cancellation orders:
        - Customer ID is valid
        - Transaction Type is "Return" or "Cancellation"
        - Quantity < 0
        - InvoiceDate <= cutoff

    If cutoff_date is None, the cutoff is automatically calculated as:

        max(InvoiceDate) - 90 days

    Returns
    -------
    pd.DataFrame
        One row per customer with:
        - Customer ID
        - Return_Rate
    """

    # ============================================================
    # 1. Validate required columns
    # ============================================================

    required_columns = {
        "Customer ID",
        "InvoiceDate",
        "Quantity",
        "Transaction Type",
        "Include In Sales Analysis",
        "Invoice",
    }

    missing_columns = [
        column
        for column in required_columns
        if column not in df_cleaned.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # ============================================================
    # 2. Create working copy
    # ============================================================

    working_df = df_cleaned.copy()

    # ============================================================
    # 3. Standardize InvoiceDate
    # ============================================================

    working_df["InvoiceDate"] = pd.to_datetime(
        working_df["InvoiceDate"],
        errors="coerce",
    )

    if working_df["InvoiceDate"].isna().any():
        raise ValueError(
            "InvoiceDate contains missing or invalid values."
        )

    # ============================================================
    # 4. Determine point-in-time cutoff
    # ============================================================

    max_invoice_date = working_df["InvoiceDate"].max()

    if cutoff_date is None:
        cutoff_date = (
            max_invoice_date
            - pd.Timedelta(days=90)
        )
    else:
        cutoff_date = pd.Timestamp(cutoff_date)

    # ============================================================
    # 5. Restrict transactions to the historical period
    # ============================================================

    historical_df = working_df[
        working_df["InvoiceDate"] <= cutoff_date
    ].copy()

    if historical_df.empty:
        raise ValueError(
            "No transactions are available on or before "
            "the specified cutoff date."
        )

    # ============================================================
    # 6. Keep customer-identifiable transactions
    # ============================================================

    historical_df = historical_df[
        historical_df["Customer ID"].notna()
        & (historical_df["Customer ID"] != 0)
    ].copy()

    if historical_df.empty:
        raise ValueError(
            "No customer-identifiable transactions are "
            "available before or on the cutoff date."
        )

    # ============================================================
    # 7. Standardize Customer ID
    # ============================================================

    historical_df["Customer ID"] = (
        historical_df["Customer ID"]
        .astype("int64")
    )

    # ============================================================
    # 8. Identify qualifying purchase orders
    # ============================================================

    purchase_transactions = historical_df[
        (historical_df["Transaction Type"] == "Purchase")
        & (
            historical_df["Include In Sales Analysis"]
            == True
        )
        & (historical_df["Quantity"] > 0)
    ].copy()

    if purchase_transactions.empty:
        raise ValueError(
            "No qualifying purchase transactions were found "
            "before or on the cutoff date."
        )

    # ============================================================
    # 9. Count distinct qualifying purchase orders
    # ============================================================

    purchase_orders = (
        purchase_transactions
        .groupby("Customer ID")["Invoice"]
        .nunique()
        .rename("Purchase_Order_Count")
    )

    # ============================================================
    # 10. Identify return/cancellation transactions
    #
    # Negative quantities are retained because the PRD
    # explicitly identifies return behavior as a predictive
    # signal.
    # ============================================================

    return_transactions = historical_df[
        historical_df["Transaction Type"].isin(
            [
                "Return",
                "Cancellation",
            ]
        )
        & (historical_df["Quantity"] < 0)
    ].copy()

    # ============================================================
    # 11. Count distinct return/cancellation orders
    # ============================================================

    return_orders = (
        return_transactions
        .groupby("Customer ID")["Invoice"]
        .nunique()
        .rename(
            "Return_Cancellation_Order_Count"
        )
    )

    # ============================================================
    # 12. Combine customer-level order counts
    #
    # Start from qualifying purchase customers so the final
    # population remains exactly aligned with the other
    # customer-level features.
    # ============================================================

    return_rate_features = (
        purchase_orders
        .to_frame()
        .join(
            return_orders,
            how="left",
        )
        .fillna(
            {
                "Return_Cancellation_Order_Count": 0
            }
        )
        .reset_index()
    )

    # ============================================================
    # 13. Calculate Return Rate
    # ============================================================

    return_rate_features["Return_Rate"] = (
        return_rate_features[
            "Return_Cancellation_Order_Count"
        ]
        / return_rate_features[
            "Purchase_Order_Count"
        ]
    )

    # ============================================================
    # 14. Validate intermediate calculations
    # ============================================================

    if (
        return_rate_features[
            "Purchase_Order_Count"
        ] <= 0
    ).any():

        raise ValueError(
            "Purchase order count must be positive "
            "for every customer."
        )

    if (
        return_rate_features[
            "Return_Cancellation_Order_Count"
        ] < 0
    ).any():

        raise ValueError(
            "Return/cancellation order count cannot be negative."
        )

    if return_rate_features[
        "Return_Rate"
    ].isna().any():

        raise ValueError(
            "Return Rate contains missing values."
        )

    if return_rate_features[
        "Return_Rate"
    ].lt(0).any():

        raise ValueError(
            "Return Rate contains negative values."
        )

    # ============================================================
    # 15. Standardize Customer ID
    # ============================================================

    return_rate_features["Customer ID"] = (
        return_rate_features["Customer ID"]
        .astype("int64")
    )

    # ============================================================
    # 16. Keep final feature columns only
    # ============================================================

    return_rate_features = return_rate_features[
        [
            "Customer ID",
            "Return_Rate",
        ]
    ].copy()

    # ============================================================
    # 17. Validate customer-level structure
    # ============================================================

    if return_rate_features[
        "Customer ID"
    ].duplicated().any():

        raise ValueError(
            "Duplicate Customer IDs found in Return Rate features."
        )

    if return_rate_features[
        "Customer ID"
    ].isna().any():

        raise ValueError(
            "Missing Customer IDs found in Return Rate features."
        )

    # ============================================================
    # 18. Store metadata
    # ============================================================

    return_rate_features.attrs[
        "cutoff_date"
    ] = cutoff_date

    return_rate_features.attrs[
        "max_invoice_date"
    ] = max_invoice_date

    return_rate_features.attrs[
        "customer_count"
    ] = len(return_rate_features)

    return_rate_features.attrs[
        "purchase_order_count"
    ] = int(
        purchase_transactions[
            "Invoice"
        ].nunique()
    )

    return_rate_features.attrs[
        "return_cancellation_order_count"
    ] = int(
        return_transactions[
            "Invoice"
        ].nunique()
    )

    return_rate_features.attrs[
        "customers_with_returns"
    ] = int(
        (
            return_rate_features["Return_Rate"] > 0
        ).sum()
    )

    return_rate_features.attrs[
        "customers_without_returns"
    ] = int(
        (
            return_rate_features["Return_Rate"] == 0
        ).sum()
    )

    return return_rate_features

#----------- 8.2 validate return rate --------
def validate_return_rate_features(
    return_rate_features: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    cutoff_date: pd.Timestamp | None = None,
) -> dict:
    """
    Validate customer-level Return Rate features.

    Return Rate is defined as:

        Distinct Return/Cancellation Orders
        -----------------------------------
        Distinct Qualifying Purchase Orders

    The validation independently reconstructs the expected
    customer population and Return Rate values from df_cleaned.

    If cutoff_date is None, the cutoff is automatically calculated as:

        max(InvoiceDate) - 90 days

    Returns
    -------
    dict
        Dictionary containing:
        - validation checks
        - summary statistics
        - failed checks
    """

    # ============================================================
    # 1. Validate required source columns
    # ============================================================

    required_source_columns = {
        "Customer ID",
        "InvoiceDate",
        "Quantity",
        "Transaction Type",
        "Include In Sales Analysis",
        "Invoice",
    }

    missing_source_columns = [
        column
        for column in required_source_columns
        if column not in df_cleaned.columns
    ]

    if missing_source_columns:
        raise ValueError(
            "Missing required columns in df_cleaned: "
            f"{missing_source_columns}"
        )

    # ============================================================
    # 2. Validate feature table columns
    # ============================================================

    required_feature_columns = {
        "Customer ID",
        "Return_Rate",
    }

    missing_feature_columns = [
        column
        for column in required_feature_columns
        if column not in return_rate_features.columns
    ]

    if missing_feature_columns:
        raise ValueError(
            "Missing required columns in return_rate_features: "
            f"{missing_feature_columns}"
        )

    # ============================================================
    # 3. Create working copy
    # ============================================================

    working_df = df_cleaned.copy()

    feature_df = return_rate_features.copy()

    # ============================================================
    # 4. Standardize InvoiceDate
    # ============================================================

    working_df["InvoiceDate"] = pd.to_datetime(
        working_df["InvoiceDate"],
        errors="coerce",
    )

    if working_df["InvoiceDate"].isna().any():
        raise ValueError(
            "InvoiceDate contains missing or invalid values "
            "in df_cleaned."
        )

    # ============================================================
    # 5. Determine point-in-time cutoff
    # ============================================================

    max_invoice_date = working_df["InvoiceDate"].max()

    if cutoff_date is None:
        cutoff_date = (
            max_invoice_date
            - pd.Timedelta(days=90)
        )
    else:
        cutoff_date = pd.Timestamp(cutoff_date)

    # ============================================================
    # 6. Basic feature-table validation
    # ============================================================

    validation = {}

    validation[
        "customer_id_column_present"
    ] = "Customer ID" in feature_df.columns

    validation[
        "return_rate_column_present"
    ] = "Return_Rate" in feature_df.columns

    validation[
        "customer_id_is_unique"
    ] = (
        feature_df["Customer ID"].is_unique
    )

    validation[
        "no_missing_customer_id"
    ] = (
        feature_df["Customer ID"].notna().all()
    )

    validation[
        "no_missing_return_rate"
    ] = (
        feature_df["Return_Rate"].notna().all()
    )

    validation[
        "return_rate_is_numeric"
    ] = (
        pd.api.types.is_numeric_dtype(
            feature_df["Return_Rate"]
        )
    )

    validation[
        "no_negative_return_rate"
    ] = (
        feature_df["Return_Rate"] >= 0
    ).all()

    # ============================================================
    # 7. Restrict source data to point-in-time window
    # ============================================================

    historical_df = working_df[
        working_df["InvoiceDate"] <= cutoff_date
    ].copy()

    # ============================================================
    # 8. Keep customer-identifiable transactions
    # ============================================================

    historical_df = historical_df[
        historical_df["Customer ID"].notna()
        & (historical_df["Customer ID"] != 0)
    ].copy()

    historical_df["Customer ID"] = (
        historical_df["Customer ID"]
        .astype("int64")
    )

    # ============================================================
    # 9. Reconstruct qualifying purchase orders
    # ============================================================

    purchase_transactions = historical_df[
        (historical_df["Transaction Type"] == "Purchase")
        & (
            historical_df["Include In Sales Analysis"]
            == True
        )
        & (historical_df["Quantity"] > 0)
    ].copy()

    expected_purchase_orders = (
        purchase_transactions
        .groupby("Customer ID")["Invoice"]
        .nunique()
        .rename("Purchase_Order_Count")
    )

    # ============================================================
    # 10. Reconstruct return/cancellation orders
    # ============================================================

    return_transactions = historical_df[
        historical_df["Transaction Type"].isin(
            [
                "Return",
                "Cancellation",
            ]
        )
        & (historical_df["Quantity"] < 0)
    ].copy()

    expected_return_orders = (
        return_transactions
        .groupby("Customer ID")["Invoice"]
        .nunique()
        .rename(
            "Return_Cancellation_Order_Count"
        )
    )

    # ============================================================
    # 11. Reconstruct expected Return Rate
    # ============================================================

    expected_features = (
        expected_purchase_orders
        .to_frame()
        .join(
            expected_return_orders,
            how="left",
        )
        .fillna(
            {
                "Return_Cancellation_Order_Count": 0
            }
        )
    )

    expected_features["Expected_Return_Rate"] = (
        expected_features[
            "Return_Cancellation_Order_Count"
        ]
        / expected_features[
            "Purchase_Order_Count"
        ]
    )

    expected_features = (
        expected_features
        .reset_index()
    )

    # ============================================================
    # 12. Standardize feature Customer ID
    # ============================================================

    feature_df["Customer ID"] = (
        feature_df["Customer ID"]
        .astype("int64")
    )

    # ============================================================
    # 13. Validate customer population
    # ============================================================

    expected_customer_ids = set(
        expected_features["Customer ID"]
    )

    actual_customer_ids = set(
        feature_df["Customer ID"]
    )

    validation[
        "customer_count_matches"
    ] = (
        len(feature_df)
        == len(expected_customer_ids)
    )

    validation[
        "customer_population_matches"
    ] = (
        actual_customer_ids
        == expected_customer_ids
    )

    # ============================================================
    # 14. Compare actual Return Rate against expected values
    # ============================================================

    comparison_df = (
        expected_features[
            [
                "Customer ID",
                "Expected_Return_Rate",
            ]
        ]
        .merge(
            feature_df[
                [
                    "Customer ID",
                    "Return_Rate",
                ]
            ],
            on="Customer ID",
            how="left",
            validate="one_to_one",
        )
    )

    validation[
        "all_expected_return_rates_present"
    ] = (
        comparison_df["Return_Rate"]
        .notna()
        .all()
    )

    validation[
        "return_rates_match"
    ] = np.allclose(
        comparison_df["Return_Rate"],
        comparison_df["Expected_Return_Rate"],
        atol=1e-12,
        rtol=1e-9,
    )

    # ============================================================
    # 15. Validate Return Rate mathematical properties
    # ============================================================

    validation[
        "no_return_rate_above_expected_domain"
    ] = (
        feature_df["Return_Rate"] >= 0
    ).all()

    # Values greater than 1 are intentionally allowed.
    #
    # Example:
    # 1 purchase order + 4 return/cancellation orders
    # = Return Rate of 4.0
    #
    # Therefore, we do NOT enforce Return_Rate <= 1.

    # ============================================================
    # 16. Validate point-in-time cutoff
    # ============================================================

    validation[
        "cutoff_before_max_invoice_date"
    ] = (
        cutoff_date <= max_invoice_date
    )

    validation[
        "no_source_transaction_after_cutoff_used"
    ] = (
        historical_df["InvoiceDate"]
        <= cutoff_date
    ).all()

    # ============================================================
    # 17. Summary statistics
    # ============================================================

    return_rate_series = feature_df[
        "Return_Rate"
    ]

    summary = {
        "customer_count": int(
            len(feature_df)
        ),

        "expected_customer_count": int(
            len(expected_features)
        ),

        "purchase_order_count": int(
            purchase_transactions[
                "Invoice"
            ].nunique()
        ),

        "return_cancellation_order_count": int(
            return_transactions[
                "Invoice"
            ].nunique()
        ),

        "customers_with_returns": int(
            (
                return_rate_series > 0
            ).sum()
        ),

        "customers_without_returns": int(
            (
                return_rate_series == 0
            ).sum()
        ),

        "mean_return_rate": float(
            return_rate_series.mean()
        ),

        "median_return_rate": float(
            return_rate_series.median()
        ),

        "minimum_return_rate": float(
            return_rate_series.min()
        ),

        "maximum_return_rate": float(
            return_rate_series.max()
        ),

        "max_invoice_date": max_invoice_date,

        "cutoff_date": cutoff_date,
    }

    # ============================================================
    # 18. Identify failed checks
    # ============================================================

    failed_checks = [
        check
        for check, result in validation.items()
        if not bool(result)
    ]

    # ============================================================
    # 19. Overall validation result
    # ============================================================

    validation["passed"] = (
        len(failed_checks) == 0
    )

    # ============================================================
    # 20. Return validation package
    # ============================================================

    return {
        "validation": validation,
        "summary": summary,
        "failed_checks": failed_checks,
    }



###############################################################
#---------- 9. Discount/markdown sensitivity --------------
###############################################################

#----------- 9.1 analyze discount sensitivity --------
def analyze_discount_markdown_sensitivity(
    df_cleaned: pd.DataFrame,
    cutoff_date: pd.Timestamp | None = None,
) -> dict:
    """
    Analyze customer-level price sensitivity using observed purchase
    prices relative to the historical median price of each product.

    This is an exploratory analysis for developing the final
    Discount / Markdown Sensitivity feature.

    The dataset does not contain an explicit discount or promotion field.
    Therefore, below-median price behavior is treated as a
    price-based promotional sensitivity proxy, not as confirmed discounting.

    The analysis:

    - Uses the same point-in-time customer population as RFM/CLV.
    - Excludes verified non-product StockCodes from price analysis.
    - Preserves the complete historical customer population.
    - Assigns zero exposure to customers without eligible product-price
      observations.
    - Calculates transaction-level price deviation from product median.
    - Caps positive deviations at 30% for robust customer-level analysis.
    - Produces customer-level frequency and value exposure measures.

    Parameters
    ----------
    df_cleaned : pd.DataFrame
        Complete cleaned transaction dataframe.

    cutoff_date : pd.Timestamp | None
        Point-in-time cutoff. If None, automatically uses
        max InvoiceDate - 90 days.

    Returns
    -------
    dict
        Complete discount/markdown sensitivity analysis package.
    """

    # ============================================================
    # 1. Validate required columns
    # ============================================================

    required_columns = {
        "Invoice",
        "StockCode",
        "InvoiceDate",
        "Price",
        "Quantity",
        "Customer ID",
        "Transaction Type",
        "Include In Sales Analysis",
        "Analytical Transaction Value",
    }

    missing_columns = required_columns - set(df_cleaned.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # ============================================================
    # 2. Create working copy
    # ============================================================

    working_df = df_cleaned.copy()

    working_df["InvoiceDate"] = pd.to_datetime(
        working_df["InvoiceDate"],
        errors="coerce",
    )

    working_df["Price"] = pd.to_numeric(
        working_df["Price"],
        errors="coerce",
    )

    # ============================================================
    # 3. Determine point-in-time cutoff
    # ============================================================

    max_invoice_date = working_df["InvoiceDate"].max()

    if pd.isna(max_invoice_date):
        raise ValueError(
            "InvoiceDate contains no valid dates."
        )

    if cutoff_date is None:
        cutoff_date = (
            max_invoice_date
            - pd.Timedelta(days=90)
        )
    else:
        cutoff_date = pd.Timestamp(cutoff_date)

    if cutoff_date >= max_invoice_date:
        raise ValueError(
            "cutoff_date must be earlier than max InvoiceDate."
        )

    # ============================================================
    # 4. Verified non-product StockCodes
    # ============================================================

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

    # ============================================================
    # 5. Historical transactions
    # ============================================================

    historical_df = working_df[
        working_df["InvoiceDate"] <= cutoff_date
    ].copy()

    # ============================================================
    # 6. Reconstruct the SAME customer population used by RFM
    # ============================================================

    historical_customer_transactions = historical_df[
        (historical_df["Customer ID"] != 0)
        & (historical_df["Customer ID"].notna())
        & (historical_df["Transaction Type"] == "Purchase")
        & (historical_df["Include In Sales Analysis"] == True)
        & (historical_df["Quantity"] > 0)
    ].copy()

    if historical_customer_transactions.empty:
        raise ValueError(
            "No qualifying historical customer transactions found."
        )

    historical_customer_transactions["Customer ID"] = (
        historical_customer_transactions["Customer ID"]
        .astype("int64")
    )

    historical_customer_population = set(
        historical_customer_transactions["Customer ID"]
        .unique()
    )

    # ============================================================
    # 7. Price-analysis transactions
    # ============================================================

    price_df = historical_customer_transactions[
        (historical_customer_transactions["Price"] > 0)
        & (historical_customer_transactions["StockCode"].notna())
    ].copy()

    price_df["StockCode_Key"] = (
        price_df["StockCode"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Exclude administrative/non-product transactions
    price_df = price_df[
        ~price_df["StockCode_Key"].isin(
            non_product_stock_codes
        )
    ].copy()

    if price_df.empty:
        raise ValueError(
            "No valid product-price transactions remain "
            "after filtering."
        )

    # ============================================================
    # 8. Product-level median price benchmark
    # ============================================================

    product_price_benchmark = (
        price_df
        .groupby("StockCode_Key")
        .agg(
            Price_Observation_Count=("Price", "size"),
            Distinct_Price_Count=("Price", "nunique"),
            Product_Median_Price=("Price", "median"),
            Product_Minimum_Price=("Price", "min"),
            Product_Maximum_Price=("Price", "max"),
        )
        .reset_index()
    )

    # ============================================================
    # 9. Merge product benchmark
    # ============================================================

    price_df = price_df.merge(
        product_price_benchmark[
            [
                "StockCode_Key",
                "Product_Median_Price",
            ]
        ],
        on="StockCode_Key",
        how="left",
        validate="many_to_one",
    )

    # ============================================================
    # 10. Calculate price deviation
    # ============================================================

    price_df["Price_Deviation_From_Median"] = (
        (
            price_df["Product_Median_Price"]
            - price_df["Price"]
        )
        / price_df["Product_Median_Price"]
    )

    # Only positive values represent below-median purchases.
    price_df["Below_Median_Price_Deviation"] = (
        price_df["Price_Deviation_From_Median"]
        .clip(lower=0)
    )

    # ============================================================
    # 11. Cap extreme deviations at 30%
    # ============================================================

    price_df["Capped_Below_Median_Deviation"] = (
        price_df["Below_Median_Price_Deviation"]
        .clip(upper=0.30)
    )

    # ============================================================
    # 12. Exploratory threshold flags
    # ============================================================

    price_df["Below_10pct_Median"] = (
        price_df["Below_Median_Price_Deviation"] >= 0.10
    )

    price_df["Below_20pct_Median"] = (
        price_df["Below_Median_Price_Deviation"] >= 0.20
    )

    price_df["Below_30pct_Median"] = (
        price_df["Below_Median_Price_Deviation"] >= 0.30
    )

    # ============================================================
    # 13. Value associated with low-price transactions
    # ============================================================

    price_df["Below_20pct_Value"] = np.where(
        price_df["Below_20pct_Median"],
        price_df["Analytical Transaction Value"],
        0.0,
    )

    price_df["Below_30pct_Value"] = np.where(
        price_df["Below_30pct_Median"],
        price_df["Analytical Transaction Value"],
        0.0,
    )

    price_df["Capped_Deviation_Value"] = (
        price_df["Capped_Below_Median_Deviation"]
        * price_df["Analytical Transaction Value"]
    )

    # ============================================================
    # 14. Customer-level price behavior
    # ============================================================

    customer_price_behavior = (
        price_df
        .groupby("Customer ID")
        .agg(
            Price_Transaction_Count=(
                "Invoice",
                "size",
            ),

            Price_Purchase_Order_Count=(
                "Invoice",
                "nunique",
            ),

            Total_Purchase_Value=(
                "Analytical Transaction Value",
                "sum",
            ),

            Mean_Below_Median_Deviation=(
                "Below_Median_Price_Deviation",
                "mean",
            ),

            Mean_Capped_Below_Median_Deviation=(
                "Capped_Below_Median_Deviation",
                "mean",
            ),

            Median_Below_Median_Deviation=(
                "Below_Median_Price_Deviation",
                "median",
            ),

            Transactions_Below_Median=(
                "Below_Median_Price_Deviation",
                lambda x: (x > 0).sum(),
            ),

            Transactions_Below_10pct=(
                "Below_10pct_Median",
                "sum",
            ),

            Transactions_Below_20pct=(
                "Below_20pct_Median",
                "sum",
            ),

            Transactions_Below_30pct=(
                "Below_30pct_Median",
                "sum",
            ),

            Value_Below_20pct=(
                "Below_20pct_Value",
                "sum",
            ),

            Value_Below_30pct=(
                "Below_30pct_Value",
                "sum",
            ),

            Capped_Deviation_Value=(
                "Capped_Deviation_Value",
                "sum",
            ),
        )
        .reset_index()
    )

    # ============================================================
    # 15. Customer-level exposure percentages
    # ============================================================

    customer_price_behavior[
        "Below_Median_Transaction_Share"
    ] = (
        customer_price_behavior[
            "Transactions_Below_Median"
        ]
        /
        customer_price_behavior[
            "Price_Transaction_Count"
        ]
    )

    customer_price_behavior[
        "Below_10pct_Transaction_Share"
    ] = (
        customer_price_behavior[
            "Transactions_Below_10pct"
        ]
        /
        customer_price_behavior[
            "Price_Transaction_Count"
        ]
    )

    customer_price_behavior[
        "Below_20pct_Transaction_Share"
    ] = (
        customer_price_behavior[
            "Transactions_Below_20pct"
        ]
        /
        customer_price_behavior[
            "Price_Transaction_Count"
        ]
    )

    customer_price_behavior[
        "Below_30pct_Transaction_Share"
    ] = (
        customer_price_behavior[
            "Transactions_Below_30pct"
        ]
        /
        customer_price_behavior[
            "Price_Transaction_Count"
        ]
    )

    customer_price_behavior[
        "Below_20pct_Value_Share"
    ] = (
        customer_price_behavior[
            "Value_Below_20pct"
        ]
        /
        customer_price_behavior[
            "Total_Purchase_Value"
        ]
    )

    customer_price_behavior[
        "Below_30pct_Value_Share"
    ] = (
        customer_price_behavior[
            "Value_Below_30pct"
        ]
        /
        customer_price_behavior[
            "Total_Purchase_Value"
        ]
    )

    customer_price_behavior[
        "Capped_Deviation_Value_Share"
    ] = (
        customer_price_behavior[
            "Capped_Deviation_Value"
        ]
        /
        customer_price_behavior[
            "Total_Purchase_Value"
        ]
    )

    # ============================================================
    # 16. Merge back to COMPLETE 5,281-customer population
    # ============================================================

    customer_population_df = pd.DataFrame(
        {
            "Customer ID": sorted(
                historical_customer_population
            )
        }
    )

    customer_behavior = customer_population_df.merge(
        customer_price_behavior,
        on="Customer ID",
        how="left",
        validate="one_to_one",
    )

    # ============================================================
    # 17. Customers without valid product-price observations
    # ============================================================

    numeric_behavior_columns = [
        column
        for column in customer_behavior.columns
        if column != "Customer ID"
    ]

    customer_behavior[
        numeric_behavior_columns
    ] = customer_behavior[
        numeric_behavior_columns
    ].fillna(0)

    customer_behavior[
        "Has_Price_Analysis"
    ] = (
        customer_behavior[
            "Price_Transaction_Count"
        ] > 0
    )

    # ============================================================
    # 18. Repeated behavior
    # ============================================================

    customer_behavior[
        "Repeated_20pct_Below_Median"
    ] = (
        customer_behavior[
            "Transactions_Below_20pct"
        ] >= 2
    )

    customer_behavior[
        "Repeated_30pct_Below_Median"
    ] = (
        customer_behavior[
            "Transactions_Below_30pct"
        ] >= 2
    )

    # ============================================================
    # 19. Exploratory behavioral bands
    # ============================================================

    customer_behavior[
        "Below_20pct_Behavior_Band"
    ] = pd.cut(
        customer_behavior[
            "Below_20pct_Transaction_Share"
        ],
        bins=[
            -0.000001,
            0.00,
            0.10,
            0.25,
            0.50,
            1.00,
        ],
        labels=[
            "No Exposure",
            "Low Exposure (0-10%)",
            "Moderate Exposure (10-25%)",
            "High Exposure (25-50%)",
            "Very High Exposure (>50%)",
        ],
        include_lowest=True,
    )

    # ============================================================
    # 20. Threshold summary
    # ============================================================

    threshold_summary = pd.DataFrame(
        {
            "Threshold": [
                "Below Median",
                "At Least 10% Below Median",
                "At Least 20% Below Median",
                "At Least 30% Below Median",
            ],
            "Customers_With_Exposure": [
                (
                    customer_behavior[
                        "Transactions_Below_Median"
                    ] > 0
                ).sum(),

                (
                    customer_behavior[
                        "Transactions_Below_10pct"
                    ] > 0
                ).sum(),

                (
                    customer_behavior[
                        "Transactions_Below_20pct"
                    ] > 0
                ).sum(),

                (
                    customer_behavior[
                        "Transactions_Below_30pct"
                    ] > 0
                ).sum(),
            ],
        }
    )

    threshold_summary[
        "Customer_Percentage"
    ] = (
        threshold_summary[
            "Customers_With_Exposure"
        ]
        / len(customer_behavior)
        * 100
    )

    # ============================================================
    # 21. Behavioral band summary
    # ============================================================

    behavioral_band_summary = (
        customer_behavior[
            "Below_20pct_Behavior_Band"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "Behavior_Band"
        )
        .reset_index(
            name="Customer_Count"
        )
    )

    behavioral_band_summary[
        "Customer_Percentage"
    ] = (
        behavioral_band_summary[
            "Customer_Count"
        ]
        / len(customer_behavior)
        * 100
    )

    # ============================================================
    # 22. Correlation between transaction and value exposure
    # ============================================================

    transaction_value_relationship = {
        "below_20pct_transaction_share_vs_value_share": (
            customer_behavior[
                "Below_20pct_Transaction_Share"
            ].corr(
                customer_behavior[
                    "Below_20pct_Value_Share"
                ],
                method="spearman",
            )
        ),

        "below_30pct_transaction_share_vs_value_share": (
            customer_behavior[
                "Below_30pct_Transaction_Share"
            ].corr(
                customer_behavior[
                    "Below_30pct_Value_Share"
                ],
                method="spearman",
            )
        ),

        "capped_deviation_vs_20pct_transaction_share": (
            customer_behavior[
                "Mean_Capped_Below_Median_Deviation"
            ].corr(
                customer_behavior[
                    "Below_20pct_Transaction_Share"
                ],
                method="spearman",
            )
        ),
    }

    # ============================================================
    # 23. Summary
    # ============================================================

    summary = {
        "max_invoice_date": max_invoice_date,
        "cutoff_date": cutoff_date,

        "historical_customer_population": len(
            historical_customer_population
        ),

        "customers_with_price_analysis": int(
            customer_behavior[
                "Has_Price_Analysis"
            ].sum()
        ),

        "customers_without_price_analysis": int(
            (
                ~customer_behavior[
                    "Has_Price_Analysis"
                ]
            ).sum()
        ),

        "product_count": int(
            price_df["StockCode_Key"].nunique()
        ),

        "products_with_multiple_prices": int(
            (
                product_price_benchmark[
                    "Distinct_Price_Count"
                ] > 1
            ).sum()
        ),

        "transactions_analyzed": len(price_df),

        "customers_with_any_below_median": int(
            (
                customer_behavior[
                    "Transactions_Below_Median"
                ] > 0
            ).sum()
        ),

        "customers_with_any_20pct_exposure": int(
            (
                customer_behavior[
                    "Transactions_Below_20pct"
                ] > 0
            ).sum()
        ),

        "customers_with_any_30pct_exposure": int(
            (
                customer_behavior[
                    "Transactions_Below_30pct"
                ] > 0
            ).sum()
        ),

        "customers_with_repeated_20pct_exposure": int(
            customer_behavior[
                "Repeated_20pct_Below_Median"
            ].sum()
        ),

        "customers_with_repeated_30pct_exposure": int(
            customer_behavior[
                "Repeated_30pct_Below_Median"
            ].sum()
        ),

        "mean_capped_deviation": (
            customer_behavior[
                "Mean_Capped_Below_Median_Deviation"
            ].mean()
        ),

        "median_capped_deviation": (
            customer_behavior[
                "Mean_Capped_Below_Median_Deviation"
            ].median()
        ),

        "maximum_capped_deviation": (
            customer_behavior[
                "Mean_Capped_Below_Median_Deviation"
            ].max()
        ),
    }

    # ============================================================
    # 24. Validation
    # ============================================================

    validation = {}

    validation["customer_population_is_5281_or_expected"] = (
        len(customer_behavior)
        == len(historical_customer_population)
    )

    validation["customer_ids_unique"] = (
        customer_behavior["Customer ID"].is_unique
    )

    validation["customer_ids_not_missing"] = (
        customer_behavior["Customer ID"].notna().all()
    )

    validation["customer_population_matches"] = (
        set(customer_behavior["Customer ID"])
        == historical_customer_population
    )

    validation["price_analysis_customers_subset_of_population"] = (
        set(
            customer_price_behavior[
                "Customer ID"
            ]
        ).issubset(
            historical_customer_population
        )
    )

    validation["capped_deviation_valid"] = (
        customer_behavior[
            "Mean_Capped_Below_Median_Deviation"
        ].between(0, 0.30).all()
    )

    validation["transaction_shares_valid"] = (
        customer_behavior[
            "Below_20pct_Transaction_Share"
        ].between(0, 1).all()
    )

    validation["value_shares_valid"] = (
        customer_behavior[
            "Below_20pct_Value_Share"
        ].between(0, 1).all()
    )

    validation["threshold_counts_consistent"] = (
        (
            customer_behavior[
                "Transactions_Below_30pct"
            ]
            <=
            customer_behavior[
                "Transactions_Below_20pct"
            ]
        ).all()
        and
        (
            customer_behavior[
                "Transactions_Below_20pct"
            ]
            <=
            customer_behavior[
                "Transactions_Below_10pct"
            ]
        ).all()
    )

    validation["no_missing_behavior_values"] = (
        customer_behavior[
            numeric_behavior_columns
        ].notna().all().all()
    )

    validation["passed"] = all(
        validation.values()
    )

    # ============================================================
    # 25. Failed checks
    # ============================================================

    failed_checks = [
        check
        for check, result in validation.items()
        if check != "passed" and not bool(result)
    ]

    # ============================================================
    # 26. Return complete analysis package
    # ============================================================

    return {
        "summary": summary,
        "validation": validation,
        "failed_checks": failed_checks,

        "customer_behavior": customer_behavior,

        "product_price_benchmark": (
            product_price_benchmark
        ),

        "transaction_price_analysis": price_df,

        "threshold_summary": threshold_summary,

        "behavioral_band_summary": (
            behavioral_band_summary
        ),

        "transaction_value_relationship": (
            transaction_value_relationship
        ),
    }

def analyze_customer_price_deviation(
    df_cleaned: pd.DataFrame,
    cutoff_date: pd.Timestamp | None = None,
) -> dict:
    """
    Analyze customer purchase prices relative to the historical median
    price of the same StockCode.

    This is an exploratory analysis for developing a defensible
    Discount / Markdown Sensitivity proxy.

    The analysis does NOT assume that a below-median price is an actual
    discount or promotion.

    Returns
    -------
    dict
        Product-level price benchmarks, transaction-level price deviation,
        customer-level exposure analysis, summary, and validation results.
    """

    # ============================================================
    # 1. Validate required columns
    # ============================================================

    required_columns = {
        "Invoice",
        "StockCode",
        "InvoiceDate",
        "Price",
        "Quantity",
        "Customer ID",
        "Transaction Type",
        "Include In Sales Analysis",
        "Analytical Transaction Value",
    }

    missing_columns = required_columns - set(df_cleaned.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # ============================================================
    # 2. Create working copy
    # ============================================================

    working_df = df_cleaned.copy()

    working_df["InvoiceDate"] = pd.to_datetime(
        working_df["InvoiceDate"],
        errors="coerce",
    )

    working_df["Price"] = pd.to_numeric(
        working_df["Price"],
        errors="coerce",
    )

    # ============================================================
    # 3. Determine point-in-time cutoff
    # ============================================================

    max_invoice_date = working_df["InvoiceDate"].max()

    if pd.isna(max_invoice_date):
        raise ValueError(
            "InvoiceDate contains no valid dates."
        )

    if cutoff_date is None:
        cutoff_date = max_invoice_date - pd.Timedelta(days=90)
    else:
        cutoff_date = pd.Timestamp(cutoff_date)

    if cutoff_date >= max_invoice_date:
        raise ValueError(
            "cutoff_date must be earlier than the maximum InvoiceDate."
        )

    # ============================================================
    # 4. Define verified non-product StockCodes
    # ============================================================

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

    # ============================================================
    # 5. Historical transactions up to cutoff
    # ============================================================

    historical_df = working_df[
        working_df["InvoiceDate"] <= cutoff_date
    ].copy()

    # ============================================================
    # 6. Select qualifying purchase transactions
    # ============================================================

    qualifying_df = historical_df[
        (historical_df["Customer ID"] != 0)
        & (historical_df["Customer ID"].notna())
        & (historical_df["Transaction Type"] == "Purchase")
        & (historical_df["Include In Sales Analysis"] == True)
        & (historical_df["Quantity"] > 0)
        & (historical_df["Price"] > 0)
        & (historical_df["StockCode"].notna())
    ].copy()

    if qualifying_df.empty:
        raise ValueError(
            "No qualifying purchase transactions were found."
        )

    # ============================================================
    # 7. Standardize keys
    # ============================================================

    qualifying_df["Customer ID"] = (
        qualifying_df["Customer ID"]
        .astype("int64")
    )

    qualifying_df["StockCode_Key"] = (
        qualifying_df["StockCode"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Exclude administrative/non-product StockCodes
    qualifying_df = qualifying_df[
        ~qualifying_df["StockCode_Key"].isin(
            non_product_stock_codes
        )
    ].copy()

    if qualifying_df.empty:
        raise ValueError(
            "No qualifying product transactions remain after "
            "excluding non-product StockCodes."
        )

    # ============================================================
    # 8. Establish product-level median price benchmark
    # ============================================================

    product_price_benchmark = (
        qualifying_df
        .groupby("StockCode_Key")
        .agg(
            Price_Observation_Count=("Price", "size"),
            Distinct_Price_Count=("Price", "nunique"),
            Product_Median_Price=("Price", "median"),
            Product_Minimum_Price=("Price", "min"),
            Product_Maximum_Price=("Price", "max"),
        )
        .reset_index()
    )

    # ============================================================
    # 9. Merge product benchmark onto transactions
    # ============================================================

    qualifying_df = qualifying_df.merge(
        product_price_benchmark[
            [
                "StockCode_Key",
                "Product_Median_Price",
            ]
        ],
        on="StockCode_Key",
        how="left",
        validate="many_to_one",
    )

    # ============================================================
    # 10. Calculate relative price deviation
    # ============================================================

    qualifying_df["Price_Deviation_From_Median"] = (
        (
            qualifying_df["Product_Median_Price"]
            - qualifying_df["Price"]
        )
        / qualifying_df["Product_Median_Price"]
    )

    # Positive value = customer paid below product median.
    # Negative value = customer paid above product median.

    qualifying_df["Below_Median_Price_Deviation"] = (
        qualifying_df["Price_Deviation_From_Median"]
        .clip(lower=0)
    )

    # ============================================================
    # 11. Create exploratory threshold flags
    # ============================================================

    qualifying_df["Below_10pct_Median_Price"] = (
        qualifying_df["Below_Median_Price_Deviation"] >= 0.10
    )

    qualifying_df["Below_20pct_Median_Price"] = (
        qualifying_df["Below_Median_Price_Deviation"] >= 0.20
    )

    qualifying_df["Below_30pct_Median_Price"] = (
        qualifying_df["Below_Median_Price_Deviation"] >= 0.30
    )

    # ============================================================
    # 12. Customer-level exposure analysis
    # ============================================================

    customer_price_exposure = (
        qualifying_df
        .groupby("Customer ID")
        .agg(
            Purchase_Transaction_Count=(
                "Invoice",
                "size",
            ),
            Purchase_Order_Count=(
                "Invoice",
                "nunique",
            ),
            Total_Purchase_Value=(
                "Analytical Transaction Value",
                "sum",
            ),

            Mean_Below_Median_Price_Deviation=(
                "Below_Median_Price_Deviation",
                "mean",
            ),

            Median_Below_Median_Price_Deviation=(
                "Below_Median_Price_Deviation",
                "median",
            ),

            Transactions_Below_Median=(
                "Below_Median_Price_Deviation",
                lambda x: (x > 0).sum(),
            ),

            Transactions_Below_10pct=(
                "Below_10pct_Median_Price",
                "sum",
            ),

            Transactions_Below_20pct=(
                "Below_20pct_Median_Price",
                "sum",
            ),

            Transactions_Below_30pct=(
                "Below_30pct_Median_Price",
                "sum",
            ),
        )
        .reset_index()
    )

    # ============================================================
    # 13. Customer-level transaction shares
    # ============================================================

    customer_price_exposure[
        "Below_Median_Transaction_Share"
    ] = (
        customer_price_exposure[
            "Transactions_Below_Median"
        ]
        / customer_price_exposure[
            "Purchase_Transaction_Count"
        ]
    )

    customer_price_exposure[
        "Below_10pct_Transaction_Share"
    ] = (
        customer_price_exposure[
            "Transactions_Below_10pct"
        ]
        / customer_price_exposure[
            "Purchase_Transaction_Count"
        ]
    )

    customer_price_exposure[
        "Below_20pct_Transaction_Share"
    ] = (
        customer_price_exposure[
            "Transactions_Below_20pct"
        ]
        / customer_price_exposure[
            "Purchase_Transaction_Count"
        ]
    )

    customer_price_exposure[
        "Below_30pct_Transaction_Share"
    ] = (
        customer_price_exposure[
            "Transactions_Below_30pct"
        ]
        / customer_price_exposure[
            "Purchase_Transaction_Count"
        ]
    )

    # ============================================================
    # 14. Value-weighted exposure
    # ============================================================

    qualifying_df["Below_20pct_Value"] = np.where(
        qualifying_df["Below_20pct_Median_Price"],
        qualifying_df["Analytical Transaction Value"],
        0.0,
    )

    qualifying_df["Below_30pct_Value"] = np.where(
        qualifying_df["Below_30pct_Median_Price"],
        qualifying_df["Analytical Transaction Value"],
        0.0,
    )

    customer_value_exposure = (
        qualifying_df
        .groupby("Customer ID")
        .agg(
            Total_Purchase_Value=(
                "Analytical Transaction Value",
                "sum",
            ),
            Value_Below_20pct_Median=(
                "Below_20pct_Value",
                "sum",
            ),
            Value_Below_30pct_Median=(
                "Below_30pct_Value",
                "sum",
            ),
        )
        .reset_index()
    )

    customer_value_exposure[
        "Below_20pct_Value_Share"
    ] = (
        customer_value_exposure[
            "Value_Below_20pct_Median"
        ]
        / customer_value_exposure[
            "Total_Purchase_Value"
        ]
    )

    customer_value_exposure[
        "Below_30pct_Value_Share"
    ] = (
        customer_value_exposure[
            "Value_Below_30pct_Median"
        ]
        / customer_value_exposure[
            "Total_Purchase_Value"
        ]
    )

    # ============================================================
    # 15. Merge customer-level analyses
    # ============================================================

    customer_price_exposure = customer_price_exposure.merge(
        customer_value_exposure[
            [
                "Customer ID",
                "Below_20pct_Value_Share",
                "Below_30pct_Value_Share",
            ]
        ],
        on="Customer ID",
        how="left",
        validate="one_to_one",
    )

    # ============================================================
    # 16. Distribution analysis
    # ============================================================

    deviation = qualifying_df[
        "Below_Median_Price_Deviation"
    ]

    summary = {
        "max_invoice_date": max_invoice_date,
        "cutoff_date": cutoff_date,

        "qualifying_transaction_count": len(
            qualifying_df
        ),

        "customer_count": qualifying_df[
            "Customer ID"
        ].nunique(),

        "product_count": qualifying_df[
            "StockCode_Key"
        ].nunique(),

        "products_with_multiple_prices": int(
            (
                product_price_benchmark[
                    "Distinct_Price_Count"
                ] > 1
            ).sum()
        ),

        "products_with_single_price": int(
            (
                product_price_benchmark[
                    "Distinct_Price_Count"
                ] == 1
            ).sum()
        ),

        "transactions_below_product_median": int(
            (deviation > 0).sum()
        ),

        "transactions_below_median_percentage": (
            (deviation > 0).mean() * 100
        ),

        "transactions_below_10pct": int(
            (
                qualifying_df[
                    "Below_10pct_Median_Price"
                ]
            ).sum()
        ),

        "transactions_below_10pct_percentage": (
            qualifying_df[
                "Below_10pct_Median_Price"
            ].mean() * 100
        ),

        "transactions_below_20pct": int(
            (
                qualifying_df[
                    "Below_20pct_Median_Price"
                ]
            ).sum()
        ),

        "transactions_below_20pct_percentage": (
            qualifying_df[
                "Below_20pct_Median_Price"
            ].mean() * 100
        ),

        "transactions_below_30pct": int(
            (
                qualifying_df[
                    "Below_30pct_Median_Price"
                ]
            ).sum()
        ),

        "transactions_below_30pct_percentage": (
            qualifying_df[
                "Below_30pct_Median_Price"
            ].mean() * 100
        ),

        "median_below_median_deviation": (
            deviation.median()
        ),

        "mean_below_median_deviation": (
            deviation.mean()
        ),

        "maximum_below_median_deviation": (
            deviation.max()
        ),

        "customers_with_below_median_purchases": int(
            (
                customer_price_exposure[
                    "Transactions_Below_Median"
                ] > 0
            ).sum()
        ),

        "customers_with_20pct_below_median_purchases": int(
            (
                customer_price_exposure[
                    "Transactions_Below_20pct"
                ] > 0
            ).sum()
        ),

        "customers_with_30pct_below_median_purchases": int(
            (
                customer_price_exposure[
                    "Transactions_Below_30pct"
                ] > 0
            ).sum()
        ),
    }

    # ============================================================
    # 17. Validation
    # ============================================================

    validation = {}

    validation["qualifying_transactions_present"] = (
        len(qualifying_df) > 0
    )

    validation["customer_ids_present"] = (
        qualifying_df["Customer ID"].notna().all()
    )

    validation["positive_prices_only"] = (
        (qualifying_df["Price"] > 0).all()
    )

    validation["product_median_prices_present"] = (
        qualifying_df[
            "Product_Median_Price"
        ].notna().all()
    )

    validation["product_median_prices_positive"] = (
        (
            qualifying_df[
                "Product_Median_Price"
            ] > 0
        ).all()
    )

    validation["price_deviation_present"] = (
        qualifying_df[
            "Price_Deviation_From_Median"
        ].notna().all()
    )

    validation["below_median_deviation_non_negative"] = (
        (
            qualifying_df[
                "Below_Median_Price_Deviation"
            ] >= 0
        ).all()
    )

    validation["customer_exposure_present"] = (
        not customer_price_exposure.empty
    )

    validation["all_transactions_within_cutoff"] = (
        qualifying_df["InvoiceDate"] <= cutoff_date
    ).all()

    validation["passed"] = all(
        validation.values()
    )

    # ============================================================
    # 18. Failed checks
    # ============================================================

    failed_checks = [
        check
        for check, result in validation.items()
        if check != "passed" and not bool(result)
    ]

    # ============================================================
    # 19. Return complete analysis package
    # ============================================================

    return {
        "summary": summary,
        "validation": validation,
        "failed_checks": failed_checks,
        "product_price_benchmark": product_price_benchmark,
        "transaction_price_analysis": qualifying_df,
        "customer_price_exposure": customer_price_exposure,
    }

def analyze_discount_sensitivity_behavior(
    discount_price_analysis: dict,
    df_cleaned: pd.DataFrame,
) -> dict:
    """
    Analyze customer-level behavior associated with purchases made
    below the historical median price of the corresponding product.

    This is an exploratory analysis for developing the final
    Discount / Markdown Sensitivity customer-level feature.

    The analysis does NOT interpret below-median prices as confirmed
    discounts or promotions.

    Parameters
    ----------
    discount_price_analysis : dict
        Output returned by analyze_customer_price_deviation().

    df_cleaned : pd.DataFrame
        Complete cleaned transaction dataframe.

    Returns
    -------
    dict
        Customer behavioral analysis, behavioral bands, coverage,
        extreme customers, relationships between measures,
        summary statistics, and validation results.
    """

    # ============================================================
    # 1. Validate input analysis package
    # ============================================================

    required_analysis_keys = {
        "customer_price_exposure",
        "transaction_price_analysis",
    }

    missing_analysis_keys = (
        required_analysis_keys
        - set(discount_price_analysis.keys())
    )

    if missing_analysis_keys:
        raise ValueError(
            "discount_price_analysis is missing required keys: "
            f"{sorted(missing_analysis_keys)}"
        )

    customer_exposure = (
        discount_price_analysis[
            "customer_price_exposure"
        ].copy()
    )

    transaction_analysis = (
        discount_price_analysis[
            "transaction_price_analysis"
        ].copy()
    )

    # ============================================================
    # 2. Validate required customer-level columns
    # ============================================================

    required_customer_columns = {
        "Customer ID",
        "Purchase_Transaction_Count",
        "Purchase_Order_Count",
        "Total_Purchase_Value",
        "Below_Median_Transaction_Share",
        "Below_10pct_Transaction_Share",
        "Below_20pct_Transaction_Share",
        "Below_30pct_Transaction_Share",
        "Below_20pct_Value_Share",
        "Below_30pct_Value_Share",
        "Transactions_Below_Median",
        "Transactions_Below_10pct",
        "Transactions_Below_20pct",
        "Transactions_Below_30pct",
    }

    missing_customer_columns = (
        required_customer_columns
        - set(customer_exposure.columns)
    )

    if missing_customer_columns:
        raise ValueError(
            "Customer exposure analysis is missing required "
            f"columns: {sorted(missing_customer_columns)}"
        )

    # ============================================================
    # 3. Copy and standardize Customer ID
    # ============================================================

    customer_exposure["Customer ID"] = (
        customer_exposure["Customer ID"]
        .astype("int64")
    )

    # ============================================================
    # 4. Determine full historical customer population
    # ============================================================

    if "Customer ID" not in df_cleaned.columns:
        raise ValueError(
            "Missing required column: Customer ID"
        )

    full_customer_population = (
        df_cleaned.loc[
            df_cleaned["Customer ID"] != 0,
            "Customer ID",
        ]
        .dropna()
        .astype("int64")
        .unique()
    )

    full_customer_population = set(
        full_customer_population
    )

    analyzed_customer_population = set(
        customer_exposure["Customer ID"]
    )

    customers_without_price_analysis = (
        full_customer_population
        - analyzed_customer_population
    )

    # ============================================================
    # 5. Create exploratory behavioral bands
    # ============================================================

    customer_exposure[
        "Below_20pct_Behavior_Band"
    ] = pd.cut(
        customer_exposure[
            "Below_20pct_Transaction_Share"
        ],
        bins=[
            -0.000001,
            0.00,
            0.10,
            0.25,
            0.50,
            1.00,
        ],
        labels=[
            "No Exposure",
            "Low Exposure (0-10%)",
            "Moderate Exposure (10-25%)",
            "High Exposure (25-50%)",
            "Very High Exposure (>50%)",
        ],
        include_lowest=True,
    )

    customer_exposure[
        "Below_30pct_Behavior_Band"
    ] = pd.cut(
        customer_exposure[
            "Below_30pct_Transaction_Share"
        ],
        bins=[
            -0.000001,
            0.00,
            0.10,
            0.25,
            0.50,
            1.00,
        ],
        labels=[
            "No Exposure",
            "Low Exposure (0-10%)",
            "Moderate Exposure (10-25%)",
            "High Exposure (25-50%)",
            "Very High Exposure (>50%)",
        ],
        include_lowest=True,
    )

    # ============================================================
    # 6. Customer behavioral band summary
    # ============================================================

    behavioral_band_summary = (
        customer_exposure[
            "Below_20pct_Behavior_Band"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "Behavior_Band"
        )
        .reset_index(
            name="Customer_Count"
        )
    )

    behavioral_band_summary[
        "Customer_Percentage"
    ] = (
        behavioral_band_summary[
            "Customer_Count"
        ]
        / len(customer_exposure)
        * 100
    )

    # ============================================================
    # 7. Threshold exposure summary
    # ============================================================

    threshold_summary = pd.DataFrame(
        {
            "Threshold": [
                "Below Median",
                "At Least 10% Below Median",
                "At Least 20% Below Median",
                "At Least 30% Below Median",
            ],
            "Customers_With_Exposure": [
                (
                    customer_exposure[
                        "Transactions_Below_Median"
                    ] > 0
                ).sum(),

                (
                    customer_exposure[
                        "Transactions_Below_10pct"
                    ] > 0
                ).sum(),

                (
                    customer_exposure[
                        "Transactions_Below_20pct"
                    ] > 0
                ).sum(),

                (
                    customer_exposure[
                        "Transactions_Below_30pct"
                    ] > 0
                ).sum(),
            ],
        }
    )

    threshold_summary[
        "Customer_Percentage"
    ] = (
        threshold_summary[
            "Customers_With_Exposure"
        ]
        / len(customer_exposure)
        * 100
    )

    # ============================================================
    # 8. Repeated low-price behavior
    # ============================================================

    customer_exposure[
        "Repeated_20pct_Below_Median"
    ] = (
        customer_exposure[
            "Transactions_Below_20pct"
        ] >= 2
    )

    customer_exposure[
        "Repeated_30pct_Below_Median"
    ] = (
        customer_exposure[
            "Transactions_Below_30pct"
        ] >= 2
    )

    repeated_behavior_summary = {
        "customers_with_2plus_20pct_below_transactions": int(
            customer_exposure[
                "Repeated_20pct_Below_Median"
            ].sum()
        ),

        "customers_with_2plus_30pct_below_transactions": int(
            customer_exposure[
                "Repeated_30pct_Below_Median"
            ].sum()
        ),

        "customers_with_5plus_20pct_below_transactions": int(
            (
                customer_exposure[
                    "Transactions_Below_20pct"
                ] >= 5
            ).sum()
        ),

        "customers_with_5plus_30pct_below_transactions": int(
            (
                customer_exposure[
                    "Transactions_Below_30pct"
                ] >= 5
            ).sum()
        ),
    }

    # ============================================================
    # 9. Transaction-share versus value-share relationship
    # ============================================================

    transaction_value_relationship = {
        "below_20pct_transaction_share_vs_value_share": (
            customer_exposure[
                "Below_20pct_Transaction_Share"
            ].corr(
                customer_exposure[
                    "Below_20pct_Value_Share"
                ],
                method="spearman",
            )
        ),

        "below_30pct_transaction_share_vs_value_share": (
            customer_exposure[
                "Below_30pct_Transaction_Share"
            ].corr(
                customer_exposure[
                    "Below_30pct_Value_Share"
                ],
                method="spearman",
            )
        ),
    }

    # ============================================================
    # 10. Identify high-exposure customers
    # ============================================================

    high_exposure_customers = (
        customer_exposure[
            (
                customer_exposure[
                    "Below_20pct_Transaction_Share"
                ] > 0
            )
        ]
        .sort_values(
            [
                "Below_20pct_Transaction_Share",
                "Below_20pct_Value_Share",
                "Transactions_Below_20pct",
            ],
            ascending=False,
        )
        .copy()
    )

    # ============================================================
    # 11. Identify strongest repeated behavior
    # ============================================================

    repeated_low_price_customers = (
        customer_exposure[
            (
                customer_exposure[
                    "Transactions_Below_20pct"
                ] >= 2
            )
        ]
        .sort_values(
            [
                "Transactions_Below_20pct",
                "Below_20pct_Transaction_Share",
                "Below_20pct_Value_Share",
            ],
            ascending=False,
        )
        .copy()
    )

    # ============================================================
    # 12. Compare transaction-based and value-based behavior
    # ============================================================

    customer_exposure[
        "Transaction_Value_Exposure_Difference_20pct"
    ] = (
        customer_exposure[
            "Below_20pct_Transaction_Share"
        ]
        - customer_exposure[
            "Below_20pct_Value_Share"
        ]
    )

    customer_exposure[
        "Transaction_Value_Exposure_Difference_30pct"
    ] = (
        customer_exposure[
            "Below_30pct_Transaction_Share"
        ]
        - customer_exposure[
            "Below_30pct_Value_Share"
        ]
    )

    # ============================================================
    # 13. Identify customers with strong value exposure
    # ============================================================

    high_value_exposure_customers = (
        customer_exposure[
            (
                customer_exposure[
                    "Below_20pct_Value_Share"
                ] > 0
            )
        ]
        .sort_values(
            [
                "Below_20pct_Value_Share",
                "Below_20pct_Transaction_Share",
                "Transactions_Below_20pct",
            ],
            ascending=False,
        )
        .copy()
    )

    # ============================================================
    # 14. Summary statistics
    # ============================================================

    summary = {
        "full_customer_population": len(
            full_customer_population
        ),

        "customers_with_price_analysis": len(
            analyzed_customer_population
        ),

        "customers_without_price_analysis": len(
            customers_without_price_analysis
        ),

        "below_20pct_share_mean": (
            customer_exposure[
                "Below_20pct_Transaction_Share"
            ].mean()
        ),

        "below_20pct_share_median": (
            customer_exposure[
                "Below_20pct_Transaction_Share"
            ].median()
        ),

        "below_20pct_share_max": (
            customer_exposure[
                "Below_20pct_Transaction_Share"
            ].max()
        ),

        "below_30pct_share_mean": (
            customer_exposure[
                "Below_30pct_Transaction_Share"
            ].mean()
        ),

        "below_30pct_share_median": (
            customer_exposure[
                "Below_30pct_Transaction_Share"
            ].median()
        ),

        "below_30pct_share_max": (
            customer_exposure[
                "Below_30pct_Transaction_Share"
            ].max()
        ),

        "customers_with_any_20pct_exposure": int(
            (
                customer_exposure[
                    "Transactions_Below_20pct"
                ] > 0
            ).sum()
        ),

        "customers_with_any_30pct_exposure": int(
            (
                customer_exposure[
                    "Transactions_Below_30pct"
                ] > 0
            ).sum()
        ),

        "customers_with_repeated_20pct_exposure": (
            repeated_behavior_summary[
                "customers_with_2plus_20pct_below_transactions"
            ]
        ),

        "customers_with_repeated_30pct_exposure": (
            repeated_behavior_summary[
                "customers_with_2plus_30pct_below_transactions"
            ]
        ),
    }

    # ============================================================
    # 15. Validation
    # ============================================================

    validation = {}

    validation["customer_exposure_not_empty"] = (
        not customer_exposure.empty
    )

    validation["customer_ids_unique"] = (
        customer_exposure["Customer ID"].is_unique
    )

    validation["customer_ids_not_missing"] = (
        customer_exposure["Customer ID"].notna().all()
    )

    validation["purchase_counts_positive"] = (
        (
            customer_exposure[
                "Purchase_Transaction_Count"
            ] > 0
        ).all()
    )

    validation["transaction_shares_valid"] = (
        (
            customer_exposure[
                "Below_20pct_Transaction_Share"
            ].between(0, 1)
        ).all()
        and
        (
            customer_exposure[
                "Below_30pct_Transaction_Share"
            ].between(0, 1)
        ).all()
    )

    validation["value_shares_valid"] = (
        (
            customer_exposure[
                "Below_20pct_Value_Share"
            ].between(0, 1)
        ).all()
        and
        (
            customer_exposure[
                "Below_30pct_Value_Share"
            ].between(0, 1)
        ).all()
    )

    validation["threshold_counts_consistent"] = (
        (
            customer_exposure[
                "Transactions_Below_30pct"
            ]
            <=
            customer_exposure[
                "Transactions_Below_20pct"
            ]
        ).all()
        and
        (
            customer_exposure[
                "Transactions_Below_20pct"
            ]
            <=
            customer_exposure[
                "Transactions_Below_10pct"
            ]
        ).all()
    )

    validation["full_population_identified"] = (
        len(
            analyzed_customer_population
            | customers_without_price_analysis
        )
        == len(full_customer_population)
    )

    validation["behavioral_bands_created"] = (
        customer_exposure[
            "Below_20pct_Behavior_Band"
        ].notna().all()
    )

    validation["passed"] = all(
        validation.values()
    )

    # ============================================================
    # 16. Failed checks
    # ============================================================

    failed_checks = [
        check
        for check, result in validation.items()
        if check != "passed" and not bool(result)
    ]

    # ============================================================
    # 17. Return complete analysis package
    # ============================================================

    return {
        "summary": summary,
        "validation": validation,
        "failed_checks": failed_checks,

        "customer_behavior": customer_exposure,

        "behavioral_band_summary": behavioral_band_summary,

        "threshold_summary": threshold_summary,

        "repeated_behavior_summary": (
            repeated_behavior_summary
        ),

        "transaction_value_relationship": (
            transaction_value_relationship
        ),

        "high_exposure_customers": (
            high_exposure_customers
        ),

        "repeated_low_price_customers": (
            repeated_low_price_customers
        ),

        "high_value_exposure_customers": (
            high_value_exposure_customers
        ),

        "customers_without_price_analysis": (
            customers_without_price_analysis
        ),
    }


#----------- 9.2 create discount sensitivity---------
def create_discount_markdown_sensitivity_features(
    df_cleaned: pd.DataFrame,
    cutoff_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Create customer-level Discount / Markdown Sensitivity features.

    Discount / Markdown Sensitivity is a price-based promotional
    sensitivity proxy because Online Retail II does not contain an
    explicit discount, markdown, or promotion field.

    For each qualifying purchase transaction:

        Price Deviation =
            (Product Median Price - Paid Price)
            / Product Median Price

    Only positive deviations are retained and capped at 30%.

    Customer-level feature:

        Discount_Markdown_Sensitivity =
            mean(Capped Positive Price Deviation)

    Customers belonging to the point-in-time customer population but
    having no eligible product-price observations receive 0.0.

    Parameters
    ----------
    df_cleaned : pd.DataFrame
        Complete cleaned transaction dataframe.

    cutoff_date : pd.Timestamp | None
        Point-in-time cutoff. If None, automatically calculated as
        max InvoiceDate - 90 days.

    Returns
    -------
    pd.DataFrame
        One row per customer with:

        Customer ID
        Discount_Markdown_Sensitivity
    """

    # ============================================================
    # 1. Validate required columns
    # ============================================================

    required_columns = {
        "Invoice",
        "StockCode",
        "InvoiceDate",
        "Price",
        "Quantity",
        "Customer ID",
        "Transaction Type",
        "Include In Sales Analysis",
    }

    missing_columns = required_columns - set(df_cleaned.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # ============================================================
    # 2. Create working copy
    # ============================================================

    working_df = df_cleaned.copy()

    working_df["InvoiceDate"] = pd.to_datetime(
        working_df["InvoiceDate"],
        errors="coerce",
    )

    working_df["Price"] = pd.to_numeric(
        working_df["Price"],
        errors="coerce",
    )

    # ============================================================
    # 3. Determine point-in-time cutoff
    # ============================================================

    max_invoice_date = working_df["InvoiceDate"].max()

    if pd.isna(max_invoice_date):
        raise ValueError(
            "InvoiceDate contains no valid dates."
        )

    if cutoff_date is None:
        cutoff_date = (
            max_invoice_date
            - pd.Timedelta(days=90)
        )
    else:
        cutoff_date = pd.Timestamp(cutoff_date)

    if cutoff_date >= max_invoice_date:
        raise ValueError(
            "cutoff_date must be earlier than max InvoiceDate."
        )

    # ============================================================
    # 4. Verified non-product StockCodes
    # ============================================================

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

    # ============================================================
    # 5. Historical qualifying customer transactions
    # ============================================================

    historical_df = working_df[
        working_df["InvoiceDate"] <= cutoff_date
    ].copy()

    historical_customer_transactions = historical_df[
        (historical_df["Customer ID"] != 0)
        & (historical_df["Customer ID"].notna())
        & (historical_df["Transaction Type"] == "Purchase")
        & (historical_df["Include In Sales Analysis"] == True)
        & (historical_df["Quantity"] > 0)
    ].copy()

    if historical_customer_transactions.empty:
        raise ValueError(
            "No qualifying historical customer transactions found."
        )

    historical_customer_transactions["Customer ID"] = (
        historical_customer_transactions["Customer ID"]
        .astype("int64")
    )

    # ============================================================
    # 6. Establish complete point-in-time customer population
    # ============================================================

    customer_population = (
        historical_customer_transactions[
            "Customer ID"
        ]
        .drop_duplicates()
        .sort_values()
        .to_frame()
        .reset_index(drop=True)
    )

    # ============================================================
    # 7. Select transactions eligible for price analysis
    # ============================================================

    price_df = historical_customer_transactions[
        (historical_customer_transactions["Price"] > 0)
        & (historical_customer_transactions["StockCode"].notna())
    ].copy()

    price_df["StockCode_Key"] = (
        price_df["StockCode"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    price_df = price_df[
        ~price_df["StockCode_Key"].isin(
            non_product_stock_codes
        )
    ].copy()

    # ============================================================
    # 8. Product-level historical median price
    # ============================================================

    product_median_prices = (
        price_df
        .groupby("StockCode_Key")["Price"]
        .median()
        .rename("Product_Median_Price")
        .reset_index()
    )

    # ============================================================
    # 9. Attach product benchmark
    # ============================================================

    price_df = price_df.merge(
        product_median_prices,
        on="StockCode_Key",
        how="left",
        validate="many_to_one",
    )

    # ============================================================
    # 10. Calculate price deviation
    # ============================================================

    price_df["Price_Deviation"] = (
        (
            price_df["Product_Median_Price"]
            - price_df["Price"]
        )
        / price_df["Product_Median_Price"]
    )

    # ============================================================
    # 11. Retain only positive deviation
    # ============================================================

    price_df["Positive_Price_Deviation"] = (
        price_df["Price_Deviation"]
        .clip(lower=0)
    )

    # ============================================================
    # 12. Cap extreme deviations at 30%
    # ============================================================

    price_df["Capped_Price_Deviation"] = (
        price_df["Positive_Price_Deviation"]
        .clip(upper=0.30)
    )

    # ============================================================
    # 13. Aggregate to customer level
    # ============================================================

    customer_features = (
        price_df
        .groupby("Customer ID")[
            "Capped_Price_Deviation"
        ]
        .mean()
        .rename(
            "Discount_Markdown_Sensitivity"
        )
        .reset_index()
    )

    # ============================================================
    # 14. Restore complete customer population
    # ============================================================

    customer_features = customer_population.merge(
        customer_features,
        on="Customer ID",
        how="left",
        validate="one_to_one",
    )

    # Customers with no eligible product-price observations
    # receive zero sensitivity.
    customer_features[
        "Discount_Markdown_Sensitivity"
    ] = (
        customer_features[
            "Discount_Markdown_Sensitivity"
        ]
        .fillna(0.0)
    )

    # ============================================================
    # 15. Final ordering and type
    # ============================================================

    customer_features = customer_features.sort_values(
        "Customer ID"
    ).reset_index(drop=True)

    customer_features[
        "Customer ID"
    ] = customer_features[
        "Customer ID"
    ].astype("int64")

    customer_features[
        "Discount_Markdown_Sensitivity"
    ] = customer_features[
        "Discount_Markdown_Sensitivity"
    ].astype("float64")

    # ============================================================
    # 16. Basic production checks
    # ============================================================

    if customer_features["Customer ID"].duplicated().any():
        raise ValueError(
            "Duplicate Customer ID detected."
        )

    if customer_features[
        "Discount_Markdown_Sensitivity"
    ].isna().any():
        raise ValueError(
            "Missing Discount_Markdown_Sensitivity detected."
        )

    if (
        customer_features[
            "Discount_Markdown_Sensitivity"
        ] < 0
    ).any():
        raise ValueError(
            "Negative Discount_Markdown_Sensitivity detected."
        )

    if (
        customer_features[
            "Discount_Markdown_Sensitivity"
        ] > 0.30
    ).any():
        raise ValueError(
            "Discount_Markdown_Sensitivity exceeds "
            "the 0.30 cap."
        )

    return customer_features


#----------- 9.3 validate discount sensitivity --------
def validate_discount_markdown_sensitivity_features(
    discount_markdown_features: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    cutoff_date: pd.Timestamp | None = None,
) -> dict:
    """
    Independently validate the production
    Discount / Markdown Sensitivity feature.

    The validator reconstructs the expected customer population,
    product-level median prices, capped price deviations, and
    customer-level sensitivity values from the cleaned transaction
    data.

    Validation checks include:

    - Required output columns
    - Customer population consistency
    - Customer ID uniqueness
    - Missing values
    - Numeric feature values
    - Feature range [0, 0.30]
    - Correct point-in-time cutoff
    - No post-cutoff transactions used
    - Correct product median prices
    - Correct customer-level aggregation
    - Correct handling of customers without eligible
      product-price observations

    Parameters
    ----------
    discount_markdown_features : pd.DataFrame
        Output of create_discount_markdown_sensitivity_features().

    df_cleaned : pd.DataFrame
        Complete cleaned transaction dataframe.

    cutoff_date : pd.Timestamp | None
        Point-in-time cutoff. If None, automatically calculated
        as max InvoiceDate - 90 days.

    Returns
    -------
    dict
        Validation checks, summary, and failed checks.
    """

    # ============================================================
    # 1. Required source columns
    # ============================================================

    required_source_columns = {
        "Invoice",
        "StockCode",
        "InvoiceDate",
        "Price",
        "Quantity",
        "Customer ID",
        "Transaction Type",
        "Include In Sales Analysis",
    }

    missing_source_columns = (
        required_source_columns
        - set(df_cleaned.columns)
    )

    if missing_source_columns:
        raise ValueError(
            "df_cleaned is missing required columns: "
            f"{sorted(missing_source_columns)}"
        )

    # ============================================================
    # 2. Required output columns
    # ============================================================

    required_feature_columns = {
        "Customer ID",
        "Discount_Markdown_Sensitivity",
    }

    missing_feature_columns = (
        required_feature_columns
        - set(discount_markdown_features.columns)
    )

    validation = {}

    validation[
        "customer_id_column_present"
    ] = "Customer ID" in discount_markdown_features.columns

    validation[
        "discount_markdown_sensitivity_column_present"
    ] = (
        "Discount_Markdown_Sensitivity"
        in discount_markdown_features.columns
    )

    if missing_feature_columns:
        return {
            "validation": {
                **validation,
                "passed": False,
            },
            "summary": {},
            "failed_checks": [
                f"Missing output columns: "
                f"{sorted(missing_feature_columns)}"
            ],
        }

    # ============================================================
    # 3. Working copy
    # ============================================================

    source_df = df_cleaned.copy()

    feature_df = discount_markdown_features.copy()

    source_df["InvoiceDate"] = pd.to_datetime(
        source_df["InvoiceDate"],
        errors="coerce",
    )

    source_df["Price"] = pd.to_numeric(
        source_df["Price"],
        errors="coerce",
    )

    # ============================================================
    # 4. Determine cutoff
    # ============================================================

    max_invoice_date = source_df["InvoiceDate"].max()

    if pd.isna(max_invoice_date):
        raise ValueError(
            "InvoiceDate contains no valid dates."
        )

    if cutoff_date is None:
        cutoff_date = (
            max_invoice_date
            - pd.Timedelta(days=90)
        )
    else:
        cutoff_date = pd.Timestamp(cutoff_date)

    validation[
        "cutoff_before_max_invoice_date"
    ] = cutoff_date < max_invoice_date

    # ============================================================
    # 5. Basic output validation
    # ============================================================

    validation[
        "feature_output_not_empty"
    ] = not feature_df.empty

    validation[
        "customer_id_is_unique"
    ] = feature_df[
        "Customer ID"
    ].is_unique

    validation[
        "no_missing_customer_id"
    ] = feature_df[
        "Customer ID"
    ].notna().all()

    validation[
        "no_missing_sensitivity"
    ] = feature_df[
        "Discount_Markdown_Sensitivity"
    ].notna().all()

    validation[
        "sensitivity_is_numeric"
    ] = pd.api.types.is_numeric_dtype(
        feature_df[
            "Discount_Markdown_Sensitivity"
        ]
    )

    # ============================================================
    # 6. Establish point-in-time historical population
    # ============================================================

    historical_df = source_df[
        source_df["InvoiceDate"] <= cutoff_date
    ].copy()

    historical_customer_transactions = historical_df[
        (historical_df["Customer ID"] != 0)
        & (historical_df["Customer ID"].notna())
        & (historical_df["Transaction Type"] == "Purchase")
        & (historical_df["Include In Sales Analysis"] == True)
        & (historical_df["Quantity"] > 0)
    ].copy()

    historical_customer_transactions[
        "Customer ID"
    ] = (
        historical_customer_transactions[
            "Customer ID"
        ].astype("int64")
    )

    expected_customer_ids = set(
        historical_customer_transactions[
            "Customer ID"
        ].unique()
    )

    actual_customer_ids = set(
        feature_df[
            "Customer ID"
        ].astype("int64")
    )

    # ============================================================
    # 7. Customer population validation
    # ============================================================

    validation[
        "customer_count_matches"
    ] = (
        len(feature_df)
        == len(expected_customer_ids)
    )

    validation[
        "customer_population_matches"
    ] = (
        actual_customer_ids
        == expected_customer_ids
    )

    # ============================================================
    # 8. Verify no post-cutoff transactions are relevant
    # ============================================================

    source_transactions_after_cutoff = (
        source_df[
            source_df["InvoiceDate"] > cutoff_date
        ]
    )

    validation[
        "no_source_transaction_after_cutoff_used"
    ] = (
        len(source_transactions_after_cutoff)
        >= 0
    )

    # This check is intentionally structural. The reconstructed
    # expected feature below only uses historical_df.
    # ============================================================

    # ============================================================
    # 9. Verified non-product StockCodes
    # ============================================================

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

    # ============================================================
    # 10. Eligible product-price transactions
    # ============================================================

    price_df = historical_customer_transactions[
        (historical_customer_transactions["Price"] > 0)
        & (historical_customer_transactions["StockCode"].notna())
    ].copy()

    price_df["StockCode_Key"] = (
        price_df["StockCode"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    price_df = price_df[
        ~price_df["StockCode_Key"].isin(
            non_product_stock_codes
        )
    ].copy()

    validation[
        "qualifying_price_transactions_present"
    ] = not price_df.empty

    # ============================================================
    # 11. Product median price reconstruction
    # ============================================================

    product_median_prices = (
        price_df
        .groupby("StockCode_Key")["Price"]
        .median()
        .rename("Product_Median_Price")
        .reset_index()
    )

    validation[
        "product_median_prices_present"
    ] = not product_median_prices.empty

    validation[
        "product_median_prices_positive"
    ] = (
        product_median_prices[
            "Product_Median_Price"
        ] > 0
    ).all()

    # ============================================================
    # 12. Attach independently reconstructed product median
    # ============================================================

    price_df = price_df.merge(
        product_median_prices,
        on="StockCode_Key",
        how="left",
        validate="many_to_one",
    )

    # ============================================================
    # 13. Reconstruct price deviation
    # ============================================================

    price_df[
        "Expected_Price_Deviation"
    ] = (
        (
            price_df["Product_Median_Price"]
            - price_df["Price"]
        )
        / price_df["Product_Median_Price"]
    )

    price_df[
        "Expected_Positive_Deviation"
    ] = (
        price_df[
            "Expected_Price_Deviation"
        ]
        .clip(lower=0)
    )

    price_df[
        "Expected_Capped_Deviation"
    ] = (
        price_df[
            "Expected_Positive_Deviation"
        ]
        .clip(upper=0.30)
    )

    # ============================================================
    # 14. Independently reconstruct customer feature
    # ============================================================

    expected_customer_features = (
        price_df
        .groupby("Customer ID")[
            "Expected_Capped_Deviation"
        ]
        .mean()
        .rename(
            "Expected_Discount_Markdown_Sensitivity"
        )
        .reset_index()
    )

    # ============================================================
    # 15. Restore complete expected population
    # ============================================================

    expected_population_df = pd.DataFrame(
        {
            "Customer ID": sorted(
                expected_customer_ids
            )
        }
    )

    expected_customer_features = (
        expected_population_df
        .merge(
            expected_customer_features,
            on="Customer ID",
            how="left",
            validate="one_to_one",
        )
    )

    expected_customer_features[
        "Expected_Discount_Markdown_Sensitivity"
    ] = (
        expected_customer_features[
            "Expected_Discount_Markdown_Sensitivity"
        ]
        .fillna(0.0)
    )

    # ============================================================
    # 16. Align actual and expected output
    # ============================================================

    actual_comparison = feature_df[
        [
            "Customer ID",
            "Discount_Markdown_Sensitivity",
        ]
    ].copy()

    actual_comparison[
        "Customer ID"
    ] = actual_comparison[
        "Customer ID"
    ].astype("int64")

    actual_comparison = actual_comparison.sort_values(
        "Customer ID"
    ).reset_index(drop=True)

    expected_customer_features = (
        expected_customer_features.sort_values(
            "Customer ID"
        )
        .reset_index(drop=True)
    )

    # ============================================================
    # 17. Verify Customer ID alignment
    # ============================================================

    validation[
        "customer_ids_aligned"
    ] = (
        actual_comparison[
            "Customer ID"
        ].equals(
            expected_customer_features[
                "Customer ID"
            ]
        )
    )

    # ============================================================
    # 18. Compare feature values
    # ============================================================

    if validation["customer_ids_aligned"]:

        actual_values = (
            actual_comparison[
                "Discount_Markdown_Sensitivity"
            ].astype(float)
        )

        expected_values = (
            expected_customer_features[
                "Expected_Discount_Markdown_Sensitivity"
            ].astype(float)
        )

        validation[
            "sensitivity_values_match"
        ] = np.allclose(
            actual_values,
            expected_values,
            atol=1e-12,
            rtol=1e-9,
        )

        maximum_absolute_difference = float(
            np.max(
                np.abs(
                    actual_values
                    - expected_values
                )
            )
        )

    else:

        validation[
            "sensitivity_values_match"
        ] = False

        maximum_absolute_difference = None

    # ============================================================
    # 19. Range validation
    # ============================================================

    validation[
        "no_negative_sensitivity"
    ] = (
        feature_df[
            "Discount_Markdown_Sensitivity"
        ] >= 0
    ).all()

    validation[
        "sensitivity_within_30pct_cap"
    ] = (
        feature_df[
            "Discount_Markdown_Sensitivity"
        ] <= 0.30
    ).all()

    # ============================================================
    # 20. Check customers without price observations
    # ============================================================

    customers_with_price_analysis = set(
        price_df[
            "Customer ID"
        ].unique()
    )

    customers_without_price_analysis = (
        expected_customer_ids
        - customers_with_price_analysis
    )

    actual_zero_for_no_price_customers = True

    if customers_without_price_analysis:

        no_price_customer_values = (
            feature_df[
                feature_df[
                    "Customer ID"
                ].isin(
                    customers_without_price_analysis
                )
            ][
                "Discount_Markdown_Sensitivity"
            ]
        )

        actual_zero_for_no_price_customers = (
            no_price_customer_values
            .eq(0.0)
            .all()
        )

    validation[
        "customers_without_price_analysis_are_zero"
    ] = actual_zero_for_no_price_customers

    # ============================================================
    # 21. Verify expected customer count
    # ============================================================

    validation[
        "expected_customer_feature_count_matches"
    ] = (
        len(expected_customer_features)
        == len(expected_customer_ids)
    )

    # ============================================================
    # 22. Overall validation
    # ============================================================

    validation[
        "passed"
    ] = all(
        bool(result)
        for result in validation.values()
    )

    # ============================================================
    # 23. Failed checks
    # ============================================================

    failed_checks = [
        check
        for check, result in validation.items()
        if check != "passed"
        and not bool(result)
    ]

    # ============================================================
    # 24. Summary
    # ============================================================

    summary = {
        "max_invoice_date": max_invoice_date,

        "cutoff_date": cutoff_date,

        "customer_count": len(
            feature_df
        ),

        "expected_customer_count": len(
            expected_customer_ids
        ),

        "price_analysis_customer_count": len(
            customers_with_price_analysis
        ),

        "customers_without_price_analysis": len(
            customers_without_price_analysis
        ),

        "product_count": int(
            product_median_prices[
                "StockCode_Key"
            ].nunique()
        ),

        "qualifying_price_transaction_count": len(
            price_df
        ),

        "minimum_sensitivity": float(
            feature_df[
                "Discount_Markdown_Sensitivity"
            ].min()
        ),

        "median_sensitivity": float(
            feature_df[
                "Discount_Markdown_Sensitivity"
            ].median()
        ),

        "mean_sensitivity": float(
            feature_df[
                "Discount_Markdown_Sensitivity"
            ].mean()
        ),

        "maximum_sensitivity": float(
            feature_df[
                "Discount_Markdown_Sensitivity"
            ].max()
        ),

        "maximum_absolute_difference": (
            maximum_absolute_difference
        ),
    }

    # ============================================================
    # 25. Return validation package
    # ============================================================

    return {
        "validation": validation,
        "summary": summary,
        "failed_checks": failed_checks,
    }


###############################################################
#-------------------- 10. Engagement score --------------
###############################################################


#---- 10.1 analyze engagement score ----
def analyze_engagement_score(
    df_cleaned: pd.DataFrame,
    cutoff_date: pd.Timestamp | None = None,
) -> dict:
    """
    Analyze a candidate customer-level Engagement Score
    based on Recency, Frequency, and Monetary value.

    The PRD defines Engagement Score as a composite of:
        - Recency
        - Frequency
        - Monetary

    This function is exploratory and does not freeze the
    final Engagement Score definition.

    Candidate approach:
        - Recency: lower is better
        - Frequency: higher is better
        - Monetary: higher is better

    Each component is converted to a 0-100 percentile-based
    score, after which an equal-weight candidate Engagement
    Score is calculated.

    Parameters
    ----------
    df_cleaned : pd.DataFrame
        Complete cleaned transaction dataframe.

    cutoff_date : pd.Timestamp | None
        Point-in-time cutoff. If None, automatically calculated
        as max InvoiceDate - 90 days.

    Returns
    -------
    dict
        Engagement score analysis package.
    """

    # ============================================================
    # 1. Validate required columns
    # ============================================================

    required_columns = {
        "Invoice",
        "InvoiceDate",
        "Quantity",
        "Customer ID",
        "Transaction Type",
        "Include In Sales Analysis",
        "Analytical Transaction Value",
    }

    missing_columns = (
        required_columns
        - set(df_cleaned.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # ============================================================
    # 2. Create working copy
    # ============================================================

    working_df = df_cleaned.copy()

    working_df["InvoiceDate"] = pd.to_datetime(
        working_df["InvoiceDate"],
        errors="coerce",
    )

    working_df["Analytical Transaction Value"] = pd.to_numeric(
        working_df["Analytical Transaction Value"],
        errors="coerce",
    )

    # ============================================================
    # 3. Determine point-in-time cutoff
    # ============================================================

    max_invoice_date = working_df["InvoiceDate"].max()

    if pd.isna(max_invoice_date):
        raise ValueError(
            "InvoiceDate contains no valid dates."
        )

    if cutoff_date is None:
        cutoff_date = (
            max_invoice_date
            - pd.Timedelta(days=90)
        )
    else:
        cutoff_date = pd.Timestamp(cutoff_date)

    if cutoff_date >= max_invoice_date:
        raise ValueError(
            "cutoff_date must be earlier than max InvoiceDate."
        )

    # ============================================================
    # 4. Select qualifying historical purchases
    # ============================================================

    historical_df = working_df[
        working_df["InvoiceDate"] <= cutoff_date
    ].copy()

    qualifying_df = historical_df[
        (historical_df["Customer ID"] != 0)
        & (historical_df["Customer ID"].notna())
        & (historical_df["Transaction Type"] == "Purchase")
        & (historical_df["Include In Sales Analysis"] == True)
        & (historical_df["Quantity"] > 0)
    ].copy()

    if qualifying_df.empty:
        raise ValueError(
            "No qualifying historical purchase transactions found."
        )

    qualifying_df["Customer ID"] = (
        qualifying_df["Customer ID"]
        .astype("int64")
    )

    # ============================================================
    # 5. Calculate customer-level RFM measures
    # ============================================================

    engagement_base = (
        qualifying_df
        .groupby("Customer ID")
        .agg(
            Last_Purchase_Date=(
                "InvoiceDate",
                "max",
            ),

            Frequency=(
                "Invoice",
                "nunique",
            ),

            Monetary=(
                "Analytical Transaction Value",
                "sum",
            ),
        )
        .reset_index()
    )

    # ============================================================
    # 6. Calculate Recency
    # ============================================================

    engagement_base["Recency"] = (
        cutoff_date
        - engagement_base["Last_Purchase_Date"]
    ).dt.total_seconds() / 86400

    # ============================================================
    # 7. Arrange RFM measures
    # ============================================================

    engagement_base = engagement_base[
        [
            "Customer ID",
            "Recency",
            "Frequency",
            "Monetary",
            "Last_Purchase_Date",
        ]
    ].copy()

    # ============================================================
    # 8. Create percentile-based component scores
    # ============================================================

    # ------------------------------------------------------------
    # Recency
    # Lower recency = better engagement
    # ------------------------------------------------------------

    engagement_base[
        "Recency_Percentile"
    ] = (
        engagement_base["Recency"]
        .rank(
            method="average",
            pct=True,
            ascending=True,
        )
    )

    engagement_base[
        "Recency_Score"
    ] = (
        1
        - engagement_base["Recency_Percentile"]
    ) * 100

    # ------------------------------------------------------------
    # Frequency
    # Higher frequency = better engagement
    # ------------------------------------------------------------

    engagement_base[
        "Frequency_Percentile"
    ] = (
        engagement_base["Frequency"]
        .rank(
            method="average",
            pct=True,
            ascending=True,
        )
    )

    engagement_base[
        "Frequency_Score"
    ] = (
        engagement_base["Frequency_Percentile"]
        * 100
    )

    # ------------------------------------------------------------
    # Monetary
    # Higher monetary value = better engagement
    # ------------------------------------------------------------

    engagement_base[
        "Monetary_Percentile"
    ] = (
        engagement_base["Monetary"]
        .rank(
            method="average",
            pct=True,
            ascending=True,
        )
    )

    engagement_base[
        "Monetary_Score"
    ] = (
        engagement_base["Monetary_Percentile"]
        * 100
    )

    # ============================================================
    # 9. Candidate equal-weight Engagement Score
    # ============================================================

    engagement_base[
        "Candidate_Engagement_Score"
    ] = (
        engagement_base[
            [
                "Recency_Score",
                "Frequency_Score",
                "Monetary_Score",
            ]
        ]
        .mean(axis=1)
    )

    # ============================================================
    # 10. Exploratory engagement bands
    # ============================================================

    engagement_base[
        "Candidate_Engagement_Band"
    ] = pd.cut(
        engagement_base[
            "Candidate_Engagement_Score"
        ],
        bins=[
            -0.000001,
            20,
            40,
            60,
            80,
            100.000001,
        ],
        labels=[
            "Very Low",
            "Low",
            "Moderate",
            "High",
            "Very High",
        ],
        include_lowest=True,
    )

    # ============================================================
    # 11. RFM distribution
    # ============================================================

    rfm_distribution = (
        engagement_base[
            [
                "Recency",
                "Frequency",
                "Monetary",
            ]
        ]
        .describe()
    )

    # ============================================================
    # 12. Component score distribution
    # ============================================================

    score_distribution = (
        engagement_base[
            [
                "Recency_Score",
                "Frequency_Score",
                "Monetary_Score",
                "Candidate_Engagement_Score",
            ]
        ]
        .describe()
    )

    # ============================================================
    # 13. Engagement band summary
    # ============================================================

    engagement_band_summary = (
        engagement_base[
            "Candidate_Engagement_Band"
        ]
        .value_counts(
            sort=False,
            dropna=False,
        )
        .rename_axis(
            "Engagement_Band"
        )
        .reset_index(
            name="Customer_Count"
        )
    )

    engagement_band_summary[
        "Customer_Percentage"
    ] = (
        engagement_band_summary[
            "Customer_Count"
        ]
        / len(engagement_base)
        * 100
    )

    # ============================================================
    # 14. Component correlation
    # ============================================================

    component_correlation = (
        engagement_base[
            [
                "Recency",
                "Frequency",
                "Monetary",
                "Recency_Score",
                "Frequency_Score",
                "Monetary_Score",
                "Candidate_Engagement_Score",
            ]
        ]
        .corr(
            method="spearman"
        )
    )

    # ============================================================
    # 15. Component summary
    # ============================================================

    component_summary = pd.DataFrame(
        {
            "Component": [
                "Recency",
                "Frequency",
                "Monetary",
            ],

            "Mean_Score": [
                engagement_base[
                    "Recency_Score"
                ].mean(),

                engagement_base[
                    "Frequency_Score"
                ].mean(),

                engagement_base[
                    "Monetary_Score"
                ].mean(),
            ],

            "Median_Score": [
                engagement_base[
                    "Recency_Score"
                ].median(),

                engagement_base[
                    "Frequency_Score"
                ].median(),

                engagement_base[
                    "Monetary_Score"
                ].median(),
            ],

            "Minimum_Score": [
                engagement_base[
                    "Recency_Score"
                ].min(),

                engagement_base[
                    "Frequency_Score"
                ].min(),

                engagement_base[
                    "Monetary_Score"
                ].min(),
            ],

            "Maximum_Score": [
                engagement_base[
                    "Recency_Score"
                ].max(),

                engagement_base[
                    "Frequency_Score"
                ].max(),

                engagement_base[
                    "Monetary_Score"
                ].max(),
            ],
        }
    )

    # ============================================================
    # 16. Highest engagement customers
    # ============================================================

    highest_engagement_customers = (
        engagement_base
        .sort_values(
            "Candidate_Engagement_Score",
            ascending=False,
        )
        .head(20)
        .copy()
    )

    # ============================================================
    # 17. Lowest engagement customers
    # ============================================================

    lowest_engagement_customers = (
        engagement_base
        .sort_values(
            "Candidate_Engagement_Score",
            ascending=True,
        )
        .head(20)
        .copy()
    )

    # ============================================================
    # 18. Validation
    # ============================================================

    validation = {}

    validation[
        "qualifying_transactions_present"
    ] = not qualifying_df.empty

    validation[
        "customer_ids_present"
    ] = qualifying_df[
        "Customer ID"
    ].notna().all()

    validation[
        "customer_ids_unique"
    ] = engagement_base[
        "Customer ID"
    ].is_unique

    validation[
        "recency_values_valid"
    ] = (
        engagement_base[
            "Recency"
        ] >= 0
    ).all()

    validation[
        "frequency_values_positive"
    ] = (
        engagement_base[
            "Frequency"
        ] > 0
    ).all()

    validation[
        "monetary_values_positive"
    ] = (
        engagement_base[
            "Monetary"
        ] > 0
    ).all()

    validation[
        "recency_scores_valid"
    ] = (
        engagement_base[
            "Recency_Score"
        ]
        .between(0, 100)
        .all()
    )

    validation[
        "frequency_scores_valid"
    ] = (
        engagement_base[
            "Frequency_Score"
        ]
        .between(0, 100)
        .all()
    )

    validation[
        "monetary_scores_valid"
    ] = (
        engagement_base[
            "Monetary_Score"
        ]
        .between(0, 100)
        .all()
    )

    validation[
        "candidate_engagement_scores_valid"
    ] = (
        engagement_base[
            "Candidate_Engagement_Score"
        ]
        .between(0, 100)
        .all()
    )

    validation[
        "no_missing_component_scores"
    ] = (
        engagement_base[
            [
                "Recency_Score",
                "Frequency_Score",
                "Monetary_Score",
            ]
        ]
        .notna()
        .all()
        .all()
    )

    validation[
        "no_missing_candidate_scores"
    ] = (
        engagement_base[
            "Candidate_Engagement_Score"
        ]
        .notna()
        .all()
    )

    validation[
        "all_transactions_within_cutoff"
    ] = (
        qualifying_df[
            "InvoiceDate"
        ] <= cutoff_date
    ).all()

    validation[
        "cutoff_before_max_invoice_date"
    ] = cutoff_date < max_invoice_date

    validation["passed"] = all(
        validation.values()
    )

    # ============================================================
    # 19. Failed checks
    # ============================================================

    failed_checks = [
        check
        for check, result in validation.items()
        if check != "passed"
        and not bool(result)
    ]

    # ============================================================
    # 20. Summary
    # ============================================================

    summary = {
        "max_invoice_date": max_invoice_date,

        "cutoff_date": cutoff_date,

        "qualifying_transaction_count": len(
            qualifying_df
        ),

        "customer_count": len(
            engagement_base
        ),

        "minimum_recency_days": float(
            engagement_base[
                "Recency"
            ].min()
        ),

        "median_recency_days": float(
            engagement_base[
                "Recency"
            ].median()
        ),

        "maximum_recency_days": float(
            engagement_base[
                "Recency"
            ].max()
        ),

        "minimum_frequency": int(
            engagement_base[
                "Frequency"
            ].min()
        ),

        "median_frequency": float(
            engagement_base[
                "Frequency"
            ].median()
        ),

        "maximum_frequency": int(
            engagement_base[
                "Frequency"
            ].max()
        ),

        "minimum_monetary": float(
            engagement_base[
                "Monetary"
            ].min()
        ),

        "median_monetary": float(
            engagement_base[
                "Monetary"
            ].median()
        ),

        "maximum_monetary": float(
            engagement_base[
                "Monetary"
            ].max()
        ),

        "candidate_engagement_mean": float(
            engagement_base[
                "Candidate_Engagement_Score"
            ].mean()
        ),

        "candidate_engagement_median": float(
            engagement_base[
                "Candidate_Engagement_Score"
            ].median()
        ),

        "candidate_engagement_minimum": float(
            engagement_base[
                "Candidate_Engagement_Score"
            ].min()
        ),

        "candidate_engagement_maximum": float(
            engagement_base[
                "Candidate_Engagement_Score"
            ].max()
        ),
    }

    # ============================================================
    # 21. Return complete analysis package
    # ============================================================

    return {
        "summary": summary,

        "validation": validation,

        "failed_checks": failed_checks,

        "engagement_base": engagement_base,

        "rfm_distribution": rfm_distribution,

        "score_distribution": score_distribution,

        "engagement_band_summary": (
            engagement_band_summary
        ),

        "component_correlation": (
            component_correlation
        ),

        "component_summary": (
            component_summary
        ),

        "highest_engagement_customers": (
            highest_engagement_customers
        ),

        "lowest_engagement_customers": (
            lowest_engagement_customers
        ),
    }

#---- 10.2 create engagement score features ----
def create_engagement_score_features(
    rfm_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create customer-level Engagement Score from frozen RFM features.

    Engagement Score is an equal-weight composite of:
        - Recency
        - Frequency
        - Total Monetary Value

    Component scoring:
        Recency:
            Lower recency = higher score.

        Frequency:
            Higher frequency = higher score.

        Total Monetary Value:
            Higher monetary value = higher score.

    Final score:
        Mean of the three component scores.

    Score range:
        0 to 100.

    Parameters
    ----------
    rfm_features : pd.DataFrame
        Frozen RFM feature dataframe.

    Returns
    -------
    pd.DataFrame
        Customer ID, Engagement_Score and Engagement_Band.
    """

    # ============================================================
    # 1. Validate required RFM columns
    # ============================================================

    required_columns = {
        "Customer ID",
        "Recency",
        "Frequency",
        "Total_Monetary_Value",
    }

    missing_columns = (
        required_columns
        - set(rfm_features.columns)
    )

    if missing_columns:
        raise ValueError(
            "rfm_features is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # ============================================================
    # 2. Create working copy
    # ============================================================

    working_df = rfm_features[
        [
            "Customer ID",
            "Recency",
            "Frequency",
            "Total_Monetary_Value",
        ]
    ].copy()

    # ============================================================
    # 3. Validate RFM input
    # ============================================================

    if working_df.empty:
        raise ValueError(
            "rfm_features is empty."
        )

    if working_df["Customer ID"].isna().any():
        raise ValueError(
            "rfm_features contains missing Customer ID values."
        )

    if working_df["Customer ID"].duplicated().any():
        raise ValueError(
            "rfm_features contains duplicate Customer IDs."
        )

    if working_df[
        [
            "Recency",
            "Frequency",
            "Total_Monetary_Value",
        ]
    ].isna().any().any():

        raise ValueError(
            "RFM features contain missing values."
        )

    # ============================================================
    # 4. Recency score
    # Lower Recency = better engagement
    # ============================================================

    working_df["Recency_Percentile"] = (
        working_df["Recency"]
        .rank(
            method="average",
            pct=True,
            ascending=True,
        )
    )

    working_df["Recency_Score"] = (
        1
        - working_df["Recency_Percentile"]
    ) * 100

    # ============================================================
    # 5. Frequency score
    # Higher Frequency = better engagement
    # ============================================================

    working_df["Frequency_Percentile"] = (
        working_df["Frequency"]
        .rank(
            method="average",
            pct=True,
            ascending=True,
        )
    )

    working_df["Frequency_Score"] = (
        working_df["Frequency_Percentile"]
        * 100
    )

    # ============================================================
    # 6. Monetary score
    # Higher monetary value = better engagement
    # ============================================================

    working_df["Monetary_Percentile"] = (
        working_df["Total_Monetary_Value"]
        .rank(
            method="average",
            pct=True,
            ascending=True,
        )
    )

    working_df["Monetary_Score"] = (
        working_df["Monetary_Percentile"]
        * 100
    )

    # ============================================================
    # 7. Final Engagement Score
    # ============================================================

    working_df["Engagement_Score"] = (
        working_df[
            [
                "Recency_Score",
                "Frequency_Score",
                "Monetary_Score",
            ]
        ]
        .mean(axis=1)
    )

    # ============================================================
    # 8. Dashboard-friendly Engagement Band
    # ============================================================

    working_df["Engagement_Band"] = pd.cut(
        working_df["Engagement_Score"],
        bins=[
            -0.000001,
            20,
            40,
            60,
            80,
            100.000001,
        ],
        labels=[
            "Very Low",
            "Low",
            "Moderate",
            "High",
            "Very High",
        ],
        include_lowest=True,
    )

    # ============================================================
    # 9. Final production output
    # ============================================================

    engagement_features = working_df[
        [
            "Customer ID",
            "Engagement_Score",
            "Engagement_Band",
        ]
    ].copy()

    # ============================================================
    # 10. Final checks
    # ============================================================

    if engagement_features[
        "Customer ID"
    ].duplicated().any():

        raise ValueError(
            "Duplicate Customer ID detected."
        )

    if engagement_features[
        "Engagement_Score"
    ].isna().any():

        raise ValueError(
            "Missing Engagement Score detected."
        )

    if (
        engagement_features[
            "Engagement_Score"
        ] < 0
    ).any():

        raise ValueError(
            "Negative Engagement Score detected."
        )

    if (
        engagement_features[
            "Engagement_Score"
        ] > 100
    ).any():

        raise ValueError(
            "Engagement Score exceeds 100."
        )

    return engagement_features

#----- 10.3 validation of engagement score features ----
def validate_engagement_score_features(
    engagement_features: pd.DataFrame,
    rfm_features: pd.DataFrame,
) -> dict:
    """
    Independently validate the production Engagement Score feature.

    Engagement Score is reconstructed from the frozen RFM features:

        Recency Score =
            (1 - Recency Percentile) * 100

        Frequency Score =
            Frequency Percentile * 100

        Monetary Score =
            Total Monetary Value Percentile * 100

        Engagement Score =
            Mean of the three component scores.

    Engagement Score range:
        0 to 100

    Parameters
    ----------
    engagement_features : pd.DataFrame
        Output of create_engagement_score_features().

    rfm_features : pd.DataFrame
        Frozen RFM feature dataframe.

    Returns
    -------
    dict
        Validation checks, summary, and failed checks.
    """

    # ============================================================
    # 1. Required RFM columns
    # ============================================================

    required_rfm_columns = {
        "Customer ID",
        "Recency",
        "Frequency",
        "Total_Monetary_Value",
    }

    missing_rfm_columns = (
        required_rfm_columns
        - set(rfm_features.columns)
    )

    if missing_rfm_columns:
        raise ValueError(
            "rfm_features is missing required columns: "
            f"{sorted(missing_rfm_columns)}"
        )

    # ============================================================
    # 2. Required Engagement columns
    # ============================================================

    required_engagement_columns = {
        "Customer ID",
        "Engagement_Score",
        "Engagement_Band",
    }

    missing_engagement_columns = (
        required_engagement_columns
        - set(engagement_features.columns)
    )

    validation = {}

    validation[
        "customer_id_column_present"
    ] = (
        "Customer ID"
        in engagement_features.columns
    )

    validation[
        "engagement_score_column_present"
    ] = (
        "Engagement_Score"
        in engagement_features.columns
    )

    validation[
        "engagement_band_column_present"
    ] = (
        "Engagement_Band"
        in engagement_features.columns
    )

    # ============================================================
    # 3. Stop early if required output columns are missing
    # ============================================================

    if missing_engagement_columns:
        failed_checks = [
            "Missing output columns: "
            f"{sorted(missing_engagement_columns)}"
        ]

        validation["passed"] = False

        return {
            "validation": validation,
            "summary": {},
            "failed_checks": failed_checks,
        }

    # ============================================================
    # 4. Create working copies
    # ============================================================

    rfm_df = rfm_features[
        [
            "Customer ID",
            "Recency",
            "Frequency",
            "Total_Monetary_Value",
        ]
    ].copy()

    actual_df = engagement_features[
        [
            "Customer ID",
            "Engagement_Score",
            "Engagement_Band",
        ]
    ].copy()

    # ============================================================
    # 5. Validate RFM input
    # ============================================================

    validation[
        "rfm_output_not_empty"
    ] = not rfm_df.empty

    validation[
        "rfm_customer_ids_unique"
    ] = rfm_df[
        "Customer ID"
    ].is_unique

    validation[
        "rfm_customer_ids_not_missing"
    ] = rfm_df[
        "Customer ID"
    ].notna().all()

    validation[
        "rfm_values_not_missing"
    ] = (
        rfm_df[
            [
                "Recency",
                "Frequency",
                "Total_Monetary_Value",
            ]
        ]
        .notna()
        .all()
        .all()
    )

    # ============================================================
    # 6. Validate Engagement output
    # ============================================================

    validation[
        "engagement_output_not_empty"
    ] = not actual_df.empty

    validation[
        "customer_ids_unique"
    ] = actual_df[
        "Customer ID"
    ].is_unique

    validation[
        "no_missing_customer_id"
    ] = actual_df[
        "Customer ID"
    ].notna().all()

    validation[
        "no_missing_engagement_score"
    ] = actual_df[
        "Engagement_Score"
    ].notna().all()

    validation[
        "engagement_score_is_numeric"
    ] = pd.api.types.is_numeric_dtype(
        actual_df[
            "Engagement_Score"
        ]
    )

    # ============================================================
    # 7. Validate customer population
    # ============================================================

    expected_customer_ids = set(
        rfm_df[
            "Customer ID"
        ]
    )

    actual_customer_ids = set(
        actual_df[
            "Customer ID"
        ]
    )

    validation[
        "customer_count_matches"
    ] = (
        len(actual_df)
        == len(rfm_df)
    )

    validation[
        "customer_population_matches"
    ] = (
        actual_customer_ids
        == expected_customer_ids
    )

    # ============================================================
    # 8. Validate RFM values
    # ============================================================

    validation[
        "recency_values_non_negative"
    ] = (
        rfm_df[
            "Recency"
        ] >= 0
    ).all()

    validation[
        "frequency_values_positive"
    ] = (
        rfm_df[
            "Frequency"
        ] > 0
    ).all()

    validation[
        "monetary_values_positive"
    ] = (
        rfm_df[
            "Total_Monetary_Value"
        ] > 0
    ).all()

    # ============================================================
    # 9. Independently calculate Recency Score
    # ============================================================

    rfm_df[
        "Expected_Recency_Percentile"
    ] = (
        rfm_df[
            "Recency"
        ]
        .rank(
            method="average",
            pct=True,
            ascending=True,
        )
    )

    rfm_df[
        "Expected_Recency_Score"
    ] = (
        1
        - rfm_df[
            "Expected_Recency_Percentile"
        ]
    ) * 100

    # ============================================================
    # 10. Independently calculate Frequency Score
    # ============================================================

    rfm_df[
        "Expected_Frequency_Percentile"
    ] = (
        rfm_df[
            "Frequency"
        ]
        .rank(
            method="average",
            pct=True,
            ascending=True,
        )
    )

    rfm_df[
        "Expected_Frequency_Score"
    ] = (
        rfm_df[
            "Expected_Frequency_Percentile"
        ]
        * 100
    )

    # ============================================================
    # 11. Independently calculate Monetary Score
    # ============================================================

    rfm_df[
        "Expected_Monetary_Percentile"
    ] = (
        rfm_df[
            "Total_Monetary_Value"
        ]
        .rank(
            method="average",
            pct=True,
            ascending=True,
        )
    )

    rfm_df[
        "Expected_Monetary_Score"
    ] = (
        rfm_df[
            "Expected_Monetary_Percentile"
        ]
        * 100
    )

    # ============================================================
    # 12. Independently calculate Engagement Score
    # ============================================================

    rfm_df[
        "Expected_Engagement_Score"
    ] = (
        rfm_df[
            [
                "Expected_Recency_Score",
                "Expected_Frequency_Score",
                "Expected_Monetary_Score",
            ]
        ]
        .mean(axis=1)
    )

    # ============================================================
    # 13. Independently calculate Engagement Band
    # ============================================================

    rfm_df[
        "Expected_Engagement_Band"
    ] = pd.cut(
        rfm_df[
            "Expected_Engagement_Score"
        ],
        bins=[
            -0.000001,
            20,
            40,
            60,
            80,
            100.000001,
        ],
        labels=[
            "Very Low",
            "Low",
            "Moderate",
            "High",
            "Very High",
        ],
        include_lowest=True,
    )

    # ============================================================
    # 14. Validate component score ranges
    # ============================================================

    validation[
        "recency_scores_valid"
    ] = (
        rfm_df[
            "Expected_Recency_Score"
        ]
        .between(0, 100)
        .all()
    )

    validation[
        "frequency_scores_valid"
    ] = (
        rfm_df[
            "Expected_Frequency_Score"
        ]
        .between(0, 100)
        .all()
    )

    validation[
        "monetary_scores_valid"
    ] = (
        rfm_df[
            "Expected_Monetary_Score"
        ]
        .between(0, 100)
        .all()
    )

    # ============================================================
    # 15. Validate expected Engagement Score
    # ============================================================

    validation[
        "expected_engagement_score_valid"
    ] = (
        rfm_df[
            "Expected_Engagement_Score"
        ]
        .between(0, 100)
        .all()
    )

    validation[
        "actual_engagement_score_valid"
    ] = (
        actual_df[
            "Engagement_Score"
        ]
        .between(0, 100)
        .all()
    )

    # ============================================================
    # 16. Align actual and expected customers
    # ============================================================

    actual_comparison = (
        actual_df[
            [
                "Customer ID",
                "Engagement_Score",
                "Engagement_Band",
            ]
        ]
        .sort_values(
            "Customer ID"
        )
        .reset_index(drop=True)
    )

    expected_comparison = (
        rfm_df[
            [
                "Customer ID",
                "Expected_Engagement_Score",
                "Expected_Engagement_Band",
            ]
        ]
        .sort_values(
            "Customer ID"
        )
        .reset_index(drop=True)
    )

    validation[
        "customer_ids_aligned"
    ] = (
        actual_comparison[
            "Customer ID"
        ].equals(
            expected_comparison[
                "Customer ID"
            ]
        )
    )

    # ============================================================
    # 17. Compare Engagement Score values
    # ============================================================

    if validation[
        "customer_ids_aligned"
    ]:

        actual_scores = (
            actual_comparison[
                "Engagement_Score"
            ]
            .astype(float)
        )

        expected_scores = (
            expected_comparison[
                "Expected_Engagement_Score"
            ]
            .astype(float)
        )

        validation[
            "engagement_scores_match"
        ] = np.allclose(
            actual_scores,
            expected_scores,
            atol=1e-12,
            rtol=1e-9,
        )

        maximum_absolute_difference = float(
            np.max(
                np.abs(
                    actual_scores
                    - expected_scores
                )
            )
        )

    else:

        validation[
            "engagement_scores_match"
        ] = False

        maximum_absolute_difference = None

    # ============================================================
    # 18. Compare Engagement Bands
    # ============================================================

    if validation[
        "customer_ids_aligned"
    ]:

        actual_bands = (
            actual_comparison[
                "Engagement_Band"
            ]
            .astype(str)
            .reset_index(drop=True)
        )

        expected_bands = (
            expected_comparison[
                "Expected_Engagement_Band"
            ]
            .astype(str)
            .reset_index(drop=True)
        )

        validation[
            "engagement_bands_match"
        ] = actual_bands.equals(
            expected_bands
        )

    else:

        validation[
            "engagement_bands_match"
        ] = False

    # ============================================================
    # 19. Validate Engagement Band categories
    # ============================================================

    expected_band_categories = {
        "Very Low",
        "Low",
        "Moderate",
        "High",
        "Very High",
    }

    actual_band_categories = set(
        actual_df[
            "Engagement_Band"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    validation[
        "engagement_band_categories_valid"
    ] = (
        actual_band_categories
        .issubset(
            expected_band_categories
        )
    )

    # ============================================================
    # 20. Overall validation
    # ============================================================

    validation[
        "passed"
    ] = all(
        bool(result)
        for result in validation.values()
    )

    # ============================================================
    # 21. Failed checks
    # ============================================================

    failed_checks = [
        check
        for check, result in validation.items()
        if check != "passed"
        and not bool(result)
    ]

    # ============================================================
    # 22. Engagement Band Summary
    # ============================================================

    engagement_band_summary = (
        actual_df[
            "Engagement_Band"
        ]
        .value_counts(
            sort=False,
            dropna=False,
        )
        .rename_axis(
            "Engagement_Band"
        )
        .reset_index(
            name="Customer_Count"
        )
    )

    engagement_band_summary[
        "Customer_Percentage"
    ] = (
        engagement_band_summary[
            "Customer_Count"
        ]
        / len(actual_df)
        * 100
    )

    # ============================================================
    # 23. Summary
    # ============================================================

    summary = {
        "customer_count": len(
            actual_df
        ),

        "expected_customer_count": len(
            rfm_df
        ),

        "minimum_engagement_score": float(
            actual_df[
                "Engagement_Score"
            ].min()
        ),

        "median_engagement_score": float(
            actual_df[
                "Engagement_Score"
            ].median()
        ),

        "mean_engagement_score": float(
            actual_df[
                "Engagement_Score"
            ].mean()
        ),

        "maximum_engagement_score": float(
            actual_df[
                "Engagement_Score"
            ].max()
        ),

        "maximum_absolute_difference": (
            maximum_absolute_difference
        ),

        "very_low_customer_count": int(
            (
                actual_df[
                    "Engagement_Band"
                ].astype(str)
                == "Very Low"
            ).sum()
        ),

        "low_customer_count": int(
            (
                actual_df[
                    "Engagement_Band"
                ].astype(str)
                == "Low"
            ).sum()
        ),

        "moderate_customer_count": int(
            (
                actual_df[
                    "Engagement_Band"
                ].astype(str)
                == "Moderate"
            ).sum()
        ),

        "high_customer_count": int(
            (
                actual_df[
                    "Engagement_Band"
                ].astype(str)
                == "High"
            ).sum()
        ),

        "very_high_customer_count": int(
            (
                actual_df[
                    "Engagement_Band"
                ].astype(str)
                == "Very High"
            ).sum()
        ),
    }

    # ============================================================
    # 24. Return validation package
    # ============================================================

    return {
        "validation": validation,

        "summary": summary,

        "failed_checks": failed_checks,

        "engagement_band_summary": (
            engagement_band_summary
        ),
    }

################################################################################


################################################################################
#--- Master Feature Engineering Functions to create feature datframe -------
################################################################################

def create_customer_feature_table(
    df_cleaned: pd.DataFrame,
    cutoff_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Create the complete customer-level feature table.

    This master production function orchestrates all frozen
    customer-level feature engineering functions.

    The user only needs to provide df_cleaned.

    Point-in-time cutoff behavior
    -----------------------------
    If cutoff_date is not provided:
        cutoff_date = max(InvoiceDate) - 90 days

    If cutoff_date is provided:
        the supplied cutoff date is used.

    The same cutoff date is passed to all feature groups.

    RFM features are created once and internally passed to
    the feature functions that depend on RFM, including
    Historical CLV and Engagement Score.

    Parameters
    ----------
    df_cleaned : pd.DataFrame
        Complete cleaned transaction dataframe.

    cutoff_date : pd.Timestamp | None, optional
        Point-in-time cutoff date.

        If None, the cutoff is automatically calculated as:

            max(InvoiceDate) - 90 days

        If provided, the supplied cutoff date is used.

    Returns
    -------
    pd.DataFrame
        Final customer-level feature table containing all
        frozen PRD features.
    """

    # ============================================================
    # 1. Validate InvoiceDate
    # ============================================================

    if "InvoiceDate" not in df_cleaned.columns:
        raise ValueError(
            "df_cleaned must contain InvoiceDate."
        )

    working_df = df_cleaned.copy()

    working_df["InvoiceDate"] = pd.to_datetime(
        working_df["InvoiceDate"],
        errors="coerce",
    )

    max_invoice_date = working_df["InvoiceDate"].max()

    if pd.isna(max_invoice_date):
        raise ValueError(
            "InvoiceDate contains no valid dates."
        )

    # ============================================================
    # 2. Determine point-in-time cutoff
    # ============================================================

    if cutoff_date is None:

        cutoff_date = (
            max_invoice_date
            - pd.Timedelta(days=90)
        )

    else:

        cutoff_date = pd.Timestamp(
            cutoff_date
        )

    # ============================================================
    # 3. Validate cutoff date
    # ============================================================

    if cutoff_date > max_invoice_date:

        raise ValueError(
            "cutoff_date cannot be later than "
            "the maximum InvoiceDate."
        )

    # ============================================================
    # 4. Create RFM
    # ============================================================

    rfm_features = create_rfm_features(
        working_df,
        cutoff_date=cutoff_date,
    )

    # ============================================================
    # 5. Create Historical CLV from RFM
    # ============================================================

    clv_features = create_clv_features(
        working_df,
        rfm_features,
        cutoff_date=cutoff_date,
    )

    # ============================================================
    # 6. Create Average Basket Size
    # ============================================================

    average_basket_features = (
        create_avg_basket_size_features(
            working_df,
            cutoff_date=cutoff_date,
        )
    )

    # ============================================================
    # 7. Create Purchase Frequency Trend
    # ============================================================

    purchase_frequency_features = (
        create_purchase_frequency_trend_features(
            working_df,
            cutoff_date=cutoff_date,
        )
    )

    # ============================================================
    # 8. Create Interpurchase Variance
    # ============================================================

    interpurchase_features = (
        create_interpurchase_variance_features(
            working_df,
            cutoff_date=cutoff_date,
        )
    )

    # ============================================================
    # 9. Create Seasonal Purchase Concentration
    # ============================================================

    seasonal_features = (
        create_seasonal_purchase_concentration_features(
            working_df,
            cutoff_date=cutoff_date,
        )
    )

    # ============================================================
    # 10. Create Product Category Affinity
    # ============================================================

    product_category_affinity_features = (
        create_product_category_affinity_features(
            working_df,
            cutoff_date=cutoff_date,
        )
    )

    # ============================================================
    # 11. Create Return Rate
    # ============================================================

    return_rate_features = (
        create_return_rate_features(
            working_df,
            cutoff_date=cutoff_date,
        )
    )

    # ============================================================
    # 12. Create Discount / Markdown Sensitivity
    # ============================================================

    discount_markdown_features = (
        create_discount_markdown_sensitivity_features(
            working_df,
            cutoff_date=cutoff_date,
        )
    )

    # ============================================================
    # 13. Create Engagement Score from RFM
    # ============================================================

    engagement_features = (
        create_engagement_score_features(
            rfm_features
        )
    )

    # ============================================================
    # 14. Collect all feature tables
    # ============================================================

    feature_tables = [
        rfm_features,
        clv_features,
        average_basket_features,
        purchase_frequency_features,
        interpurchase_features,
        seasonal_features,
        product_category_affinity_features,
        return_rate_features,
        discount_markdown_features,
        engagement_features,
    ]

    # ============================================================
    # 15. Validate Customer ID before merging
    # ============================================================

    for index, feature_table in enumerate(
        feature_tables,
        start=1,
    ):

        if "Customer ID" not in feature_table.columns:

            raise ValueError(
                f"Feature table {index} does not contain "
                "Customer ID."
            )

        if feature_table[
            "Customer ID"
        ].duplicated().any():

            raise ValueError(
                f"Feature table {index} contains "
                "duplicate Customer IDs."
            )

        if feature_table[
            "Customer ID"
        ].isna().any():

            raise ValueError(
                f"Feature table {index} contains "
                "missing Customer IDs."
            )

    # ============================================================
    # 16. Validate customer populations
    # ============================================================

    reference_customer_ids = set(
        rfm_features["Customer ID"]
    )

    for index, feature_table in enumerate(
        feature_tables[1:],
        start=2,
    ):

        current_customer_ids = set(
            feature_table["Customer ID"]
        )

        if current_customer_ids != reference_customer_ids:

            raise ValueError(
                f"Feature table {index} has a different "
                "customer population from the RFM feature table."
            )

    # ============================================================
    # 17. Merge all feature tables
    # ============================================================

    customer_features = feature_tables[0].copy()

    for feature_table in feature_tables[1:]:

        customer_features = customer_features.merge(
            feature_table,
            on="Customer ID",
            how="inner",
            validate="one_to_one",
        )

    # ============================================================
    # 18. Final Customer ID validation
    # ============================================================

    if customer_features[
        "Customer ID"
    ].duplicated().any():

        raise ValueError(
            "Duplicate Customer IDs detected in "
            "final customer feature table."
        )

    if customer_features[
        "Customer ID"
    ].isna().any():

        raise ValueError(
            "Missing Customer IDs detected in "
            "final customer feature table."
        )

    # ============================================================
    # 19. Store pipeline metadata
    # ============================================================

    customer_features.attrs[
        "max_invoice_date"
    ] = max_invoice_date

    customer_features.attrs[
        "cutoff_date"
    ] = cutoff_date

    customer_features.attrs[
        "customer_count"
    ] = len(customer_features)

    customer_features.attrs[
        "feature_table_count"
    ] = len(feature_tables)

    # ============================================================
    # 20. Return final feature table
    # ============================================================

    return customer_features





################################################################################################
#===============================================================================================
# ------ FEATURE ENGINEERING SEPARATE FOR PRODUCT PURCHSE TIMMING BEHAVIOUR----------------------------------------------------------------
# -------FOR LSTM Sequential model
#===============================================================================================
################################################################################################

# ---- 1. Preparing lstm product taxonomy --------------------------------
def prepare_lstm_product_taxonomy(df_transactions, taxonomy_result=None):
    """
    Prepare product taxonomy and eligible transaction rows
    for the LSTM purchase-behaviour pipeline.

    Excludes:
        - Missing customer IDs
        - Customer ID = 0
        - Non-purchase transactions
        - Transactions excluded from sales analysis
        - Missing invoice dates
        - Known administrative/non-product stock codes

    Returns
    -------
    dict containing:
        product_catalog
        lstm_transactions
        unmatched_products
        excluded_admin_transactions
        taxonomy_audit
    """

    import pandas as pd

    # --------------------------------------------------
    # 1. Validate input columns
    # --------------------------------------------------
    required_columns = [
        "Customer ID",
        "StockCode",
        "InvoiceDate",
        "Transaction Type",
        "Include In Sales Analysis",
        "Analytical Transaction Value",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df_transactions.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------
    # 2. Generate or reuse product taxonomy
    # --------------------------------------------------
    if taxonomy_result is None:
        taxonomy_result = analyze_product_taxonomy(
            df_transactions
        )

    if "product_catalog" not in taxonomy_result:
        raise KeyError(
            "Taxonomy result must contain 'product_catalog'."
        )

    product_catalog = taxonomy_result[
        "product_catalog"
    ].copy()

    catalog_required_columns = [
        "StockCode_Key",
        "Product_Category",
    ]

    missing_catalog_columns = [
        column
        for column in catalog_required_columns
        if column not in product_catalog.columns
    ]

    if missing_catalog_columns:
        raise ValueError(
            "Product catalog is missing columns: "
            f"{missing_catalog_columns}"
        )

    # --------------------------------------------------
    # 3. Validate product catalog
    # --------------------------------------------------
    if product_catalog["StockCode_Key"].isna().any():
        raise ValueError(
            "Product catalog contains missing stock-code keys."
        )

    if product_catalog["StockCode_Key"].duplicated().any():
        raise ValueError(
            "Product catalog must contain one row per product."
        )

    if product_catalog["Product_Category"].isna().any():
        raise ValueError(
            "Product catalog contains missing product categories."
        )

    # --------------------------------------------------
    # 4. Standardize stock codes
    # --------------------------------------------------
    transactions = df_transactions.copy()

    transactions["StockCode_Key"] = (
        transactions["StockCode"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------
    # 5. Define administrative/non-product stock codes
    # --------------------------------------------------
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

    transactions["Is_Admin_StockCode"] = (
        transactions["StockCode_Key"].isin(
            admin_stock_codes
        )
    )

    # --------------------------------------------------
    # 6. Select eligible transactions
    # --------------------------------------------------
    eligible_mask = (
        transactions["Customer ID"].notna()
        & transactions["Customer ID"].ne(0)
        & transactions["Transaction Type"].eq("Purchase")
        & transactions["Include In Sales Analysis"].eq(True)
        & transactions["InvoiceDate"].notna()
    )

    # Count eligible administrative rows before exclusion.
    eligible_admin_mask = (
        eligible_mask
        & transactions["Is_Admin_StockCode"]
    )

    eligible_admin_rows_excluded = int(
        eligible_admin_mask.sum()
    )

    # Keep only eligible retail purchase transactions.
    retail_mask = (
        eligible_mask
        & ~transactions["Is_Admin_StockCode"]
    )

    eligible_transactions = transactions.loc[
        retail_mask
    ].copy()

    # --------------------------------------------------
    # 7. Attach product categories
    # --------------------------------------------------
    eligible_transactions = eligible_transactions.merge(
        product_catalog[
            ["StockCode_Key", "Product_Category"]
        ],
        on="StockCode_Key",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    eligible_transactions["Taxonomy_Match_Status"] = (
        eligible_transactions["_merge"]
        .map({
            "both": "Matched",
            "left_only": "Unmatched",
        })
    )

    unmatched_mask = (
        eligible_transactions["Taxonomy_Match_Status"]
        .eq("Unmatched")
    )

    unmatched_transaction_rows = int(
        unmatched_mask.sum()
    )

    unmatched_products = (
        eligible_transactions.loc[
            unmatched_mask,
            ["StockCode_Key"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    eligible_transactions = eligible_transactions.drop(
        columns=["_merge"]
    )

    # --------------------------------------------------
    # 8. Retain excluded administrative rows for auditing
    # --------------------------------------------------
    excluded_admin_transactions = transactions.loc[
        eligible_admin_mask
    ].copy()

    # --------------------------------------------------
    # 9. Sort chronologically
    # --------------------------------------------------
    lstm_transactions = (
        eligible_transactions
        .sort_values(
            [
                "Customer ID",
                "InvoiceDate",
                "StockCode_Key",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # 10. Build audit summary
    # --------------------------------------------------
    taxonomy_audit = {
        "input_transaction_rows": len(df_transactions),

        "eligible_purchase_rows_before_admin_exclusion": int(
            eligible_mask.sum()
        ),

        "eligible_admin_rows_excluded": (
            eligible_admin_rows_excluded
        ),

        "eligible_retail_rows_after_exclusion": (
            len(lstm_transactions)
        ),

        "matched_transaction_rows": int(
            lstm_transactions[
                "Taxonomy_Match_Status"
            ].eq("Matched").sum()
        ),

        "unmatched_transaction_rows": (
            unmatched_transaction_rows
        ),

        "unmatched_unique_products": len(
            unmatched_products
        ),

        "unique_customers": int(
            lstm_transactions["Customer ID"].nunique()
        ),

        "unique_products_in_transactions": int(
            lstm_transactions["StockCode_Key"].nunique()
        ),

        "unique_products_in_catalog": int(
            product_catalog["StockCode_Key"].nunique()
        ),

        "category_counts": (
            lstm_transactions["Product_Category"]
            .value_counts(dropna=False)
            .to_dict()
        ),
    }

    # --------------------------------------------------
    # 11. Return outputs
    # --------------------------------------------------
    return {
        "product_catalog": product_catalog,
        "lstm_transactions": lstm_transactions,
        "unmatched_products": unmatched_products,
        "excluded_admin_transactions": (
            excluded_admin_transactions
        ),
        "taxonomy_audit": taxonomy_audit,
    }


#---- 2. build lstm_purchase_events----------------------------------------------------------------
def build_lstm_purchase_events(lstm_taxonomy_result, invoice_col=None):
    """
    Aggregate transaction-level data into customer purchase events
    for the LSTM purchase-timing pipeline.

    Parameters
    ----------
    lstm_taxonomy_result : dict
        Output from prepare_lstm_product_taxonomy().

    invoice_col : str or None
        Invoice identifier column. If None, detect a supported name.

    Returns
    -------
    dict containing:
        purchase_events:
            One row per customer invoice.

        event_category_features:
            Category line counts and category purchase values
            for each customer invoice.

        event_audit:
            Summary of event aggregation and validation.
    """

    import re
    import pandas as pd

    # --------------------------------------------------
    # 1. Validate input
    # --------------------------------------------------
    if "lstm_transactions" not in lstm_taxonomy_result:
        raise KeyError(
            "Input must contain 'lstm_transactions'."
        )

    transactions = lstm_taxonomy_result[
        "lstm_transactions"
    ].copy()

    if transactions.empty:
        raise ValueError(
            "No transactions available for purchase-event creation."
        )

    # --------------------------------------------------
    # 2. Identify invoice column
    # --------------------------------------------------
    if invoice_col is None:
        possible_invoice_columns = [
            "Invoice",
            "InvoiceNo",
            "Invoice ID",
            "Invoice_ID",
        ]

        invoice_col = next(
            (
                column
                for column in possible_invoice_columns
                if column in transactions.columns
            ),
            None,
        )

    if invoice_col is None or invoice_col not in transactions.columns:
        raise ValueError(
            "Invoice column not found. Pass the correct column "
            "using invoice_col='your_column_name'."
        )

    # --------------------------------------------------
    # 3. Validate required columns
    # --------------------------------------------------
    required_columns = [
        "Customer ID",
        "InvoiceDate",
        "StockCode_Key",
        "Product_Category",
        "Analytical Transaction Value",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in transactions.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if transactions[
        ["Customer ID", invoice_col, "InvoiceDate"]
    ].isna().any().any():
        raise ValueError(
            "Missing customer IDs, invoice IDs, or invoice dates "
            "were found in the prepared transactions."
        )

    if transactions["Product_Category"].isna().any():
        raise ValueError(
            "Some transaction rows have no product category. "
            "Resolve unmatched taxonomy rows before aggregation."
        )

    # --------------------------------------------------
    # 4. Validate purchase values
    # --------------------------------------------------
    transactions["Analytical Transaction Value"] = pd.to_numeric(
        transactions["Analytical Transaction Value"],
        errors="raise",
    )

    if not transactions[
        "Analytical Transaction Value"
    ].map(
        lambda value: pd.notna(value) and value >= 0
    ).all():
        raise ValueError(
            "Analytical Transaction Value contains missing "
            "or negative values."
        )

    if "Quantity" in transactions.columns:
        transactions["Quantity"] = pd.to_numeric(
            transactions["Quantity"],
            errors="raise",
        )

        if transactions["Quantity"].isna().any():
            raise ValueError(
                "Quantity contains missing values."
            )

    # --------------------------------------------------
    # 5. Define event keys
    # --------------------------------------------------
    event_keys = [
        "Customer ID",
        invoice_col,
    ]

    # --------------------------------------------------
    # 6. Aggregate transaction rows into purchase events
    # --------------------------------------------------
    aggregations = {
        "InvoiceDate": ("InvoiceDate", "min"),
        "Purchase_Value": (
            "Analytical Transaction Value",
            "sum",
        ),
        "Product_Line_Count": (
            "StockCode_Key",
            "size",
        ),
        "Unique_Product_Count": (
            "StockCode_Key",
            "nunique",
        ),
    }

    if "Quantity" in transactions.columns:
        aggregations["Total_Quantity"] = (
            "Quantity",
            "sum",
        )

    purchase_events = (
        transactions
        .groupby(event_keys, dropna=False)
        .agg(**aggregations)
        .reset_index()
        .rename(
            columns={invoice_col: "Invoice_ID"}
        )
    )

    # --------------------------------------------------
    # 7. Build category-level event features
    # --------------------------------------------------
    def make_category_column(category, suffix):
        safe_name = re.sub(
            r"[^A-Za-z0-9]+",
            "_",
            str(category),
        ).strip("_")

        return f"Category_{safe_name}_{suffix}"

    # Count transaction lines in each category per invoice.
    category_line_counts = (
        transactions
        .groupby(
            event_keys + ["Product_Category"],
            observed=True,
        )
        .size()
        .unstack(
            "Product_Category",
            fill_value=0,
        )
    )

    category_line_counts.columns = [
        make_category_column(category, "Line_Count")
        for category in category_line_counts.columns
    ]

    # Sum purchase value by category per invoice.
    category_purchase_values = (
        transactions
        .groupby(
            event_keys + ["Product_Category"],
            observed=True,
        )["Analytical Transaction Value"]
        .sum()
        .unstack(
            "Product_Category",
            fill_value=0,
        )
    )

    category_purchase_values.columns = [
        make_category_column(category, "Value")
        for category in category_purchase_values.columns
    ]

    # Combine category line counts and category values.
    event_category_features = pd.concat(
        [
            category_line_counts,
            category_purchase_values,
        ],
        axis=1,
    ).reset_index()

    event_category_features = (
        event_category_features
        .rename(columns={invoice_col: "Invoice_ID"})
    )

    # --------------------------------------------------
    # 8. Merge event features into purchase-event table
    # --------------------------------------------------
    purchase_events = purchase_events.merge(
        event_category_features,
        on=["Customer ID", "Invoice_ID"],
        how="left",
        validate="one_to_one",
    )

    category_feature_columns = [
        column
        for column in purchase_events.columns
        if column.startswith("Category_")
    ]

    purchase_events[category_feature_columns] = (
        purchase_events[category_feature_columns]
        .fillna(0)
    )

    # --------------------------------------------------
    # 9. Sort events chronologically
    # --------------------------------------------------
    purchase_events = (
        purchase_events
        .sort_values(
            [
                "Customer ID",
                "InvoiceDate",
                "Invoice_ID",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # 10. Validate event uniqueness
    # --------------------------------------------------
    duplicate_event_count = int(
        purchase_events.duplicated(
            ["Customer ID", "Invoice_ID"]
        ).sum()
    )

    if duplicate_event_count > 0:
        raise ValueError(
            "Duplicate customer-invoice events detected."
        )

    if not purchase_events[
        "Purchase_Value"
    ].map(
        lambda value: pd.notna(value) and value >= 0
    ).all():
        raise ValueError(
            "Invalid purchase values found after aggregation."
        )

    # Verify category line counts reconcile to source rows.
    category_line_columns = [
        column
        for column in purchase_events.columns
        if column.startswith("Category_")
        and column.endswith("_Line_Count")
    ]

    total_category_lines = int(
        purchase_events[category_line_columns]
        .sum()
        .sum()
    )

    if total_category_lines != len(transactions):
        raise ValueError(
            "Category line counts do not reconcile with "
            "the source transaction row count."
        )

    # --------------------------------------------------
    # 11. Build audit summary
    # --------------------------------------------------
    event_audit = {
        "input_transaction_rows": len(transactions),
        "purchase_event_count": len(purchase_events),
        "unique_customers": int(
            purchase_events["Customer ID"].nunique()
        ),
        "unique_invoices": int(
            purchase_events["Invoice_ID"].nunique()
        ),
        "duplicate_customer_invoice_events": (
            duplicate_event_count
        ),
        "category_line_count_total": total_category_lines,
        "category_line_counts_reconcile": (
            total_category_lines == len(transactions)
        ),
        "total_purchase_value_transactions": float(
            transactions[
                "Analytical Transaction Value"
            ].sum()
        ),
        "total_purchase_value_events": float(
            purchase_events["Purchase_Value"].sum()
        ),
        "category_feature_column_count": len(
            category_feature_columns
        ),
    }

    # --------------------------------------------------
    # 12. Return results
    # --------------------------------------------------
    return {
        "purchase_events": purchase_events,
        "event_category_features": event_category_features,
        "event_audit": event_audit,
    }


#----- 3. create_lstm_purchase_timming_targets----------------------------------------------------------------
def create_lstm_purchase_timing_targets(purchase_events):
    """
    Create next-purchase timing targets for the LSTM pipeline.

    Automatically converts InvoiceDate to datetime so the function
    can run in an automated pipeline without manual notebook steps.

    Parameters
    ----------
    purchase_events : pandas.DataFrame
        Output DataFrame from build_lstm_purchase_events().

    Returns
    -------
    dict containing:
        purchase_events_with_targets
        modeling_targets
        target_audit
    """

    import pandas as pd

    # --------------------------------------------------
    # 1. Validate input
    # --------------------------------------------------
    if not isinstance(purchase_events, pd.DataFrame):
        raise TypeError(
            "purchase_events must be a pandas DataFrame."
        )

    required_columns = [
        "Customer ID",
        "Invoice_ID",
        "InvoiceDate",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in purchase_events.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if purchase_events.empty:
        raise ValueError(
            "purchase_events DataFrame is empty."
        )

    events = purchase_events.copy()

    # --------------------------------------------------
    # 2. Normalize InvoiceDate inside the function
    # --------------------------------------------------
    # Handles strings, pandas datetime values, and compatible
    # datetime-like values without requiring a notebook step.
    original_dates = events["InvoiceDate"]

    converted_dates = pd.to_datetime(
        original_dates,
        errors="coerce",
    )

    invalid_date_mask = (
        original_dates.notna()
        & converted_dates.isna()
    )

    missing_date_mask = original_dates.isna()

    invalid_date_count = int(invalid_date_mask.sum())
    missing_date_count = int(missing_date_mask.sum())

    if invalid_date_count or missing_date_count:
        raise ValueError(
            "InvoiceDate validation failed: "
            f"{invalid_date_count} unparseable dates and "
            f"{missing_date_count} missing dates. "
            "Correct these records upstream before continuing."
        )

    events["InvoiceDate"] = converted_dates

    # --------------------------------------------------
    # 3. Validate customer and invoice identifiers
    # --------------------------------------------------
    if events[
        ["Customer ID", "Invoice_ID"]
    ].isna().any().any():
        raise ValueError(
            "Customer ID and Invoice_ID must not be missing."
        )

    if events["Customer ID"].eq(0).any():
        raise ValueError(
            "Customer ID = 0 found. Remove unknown customers "
            "during LSTM taxonomy preparation."
        )

    duplicate_count = int(
        events.duplicated(
            ["Customer ID", "Invoice_ID"]
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate customer-invoice events."
        )

    # --------------------------------------------------
    # 4. Sort events chronologically per customer
    # --------------------------------------------------
    events = (
        events
        .sort_values(
            ["Customer ID", "InvoiceDate", "Invoice_ID"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # 5. Calculate next purchase date and invoice
    # --------------------------------------------------
    customer_groups = events.groupby(
        "Customer ID",
        sort=False,
    )

    events["Next_Purchase_Date"] = (
        customer_groups["InvoiceDate"].shift(-1)
    )

    events["Next_Invoice_ID"] = (
        customer_groups["Invoice_ID"].shift(-1)
    )

    # --------------------------------------------------
    # 6. Calculate time until the next purchase
    # --------------------------------------------------
    events["Days_Until_Next_Purchase"] = (
        events["Next_Purchase_Date"]
        - events["InvoiceDate"]
    ).dt.total_seconds() / 86400

    valid_target_mask = (
        events["Next_Purchase_Date"].notna()
    )

    negative_target_count = int(
        events.loc[
            valid_target_mask,
            "Days_Until_Next_Purchase",
        ].lt(0).sum()
    )

    if negative_target_count:
        raise ValueError(
            "Negative purchase-timing targets detected."
        )

    # Keep every event for auditing.
    purchase_events_with_targets = events.copy()

    # Use only events with an observed next purchase for training.
    modeling_targets = (
        events.loc[valid_target_mask]
        .copy()
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # 7. Build audit summary
    # --------------------------------------------------
    events_per_customer = (
        events.groupby("Customer ID").size()
    )

    target_values = modeling_targets[
        "Days_Until_Next_Purchase"
    ]

    target_audit = {
        "input_purchase_events": len(events),

        "unique_customers": int(
            events["Customer ID"].nunique()
        ),

        "invoice_date_dtype": str(
            events["InvoiceDate"].dtype
        ),

        "invalid_dates": invalid_date_count,

        "missing_dates": missing_date_count,

        "events_with_valid_next_purchase_target": len(
            modeling_targets
        ),

        "events_without_next_purchase_target": int(
            (~valid_target_mask).sum()
        ),

        "customers_with_one_event": int(
            events_per_customer.eq(1).sum()
        ),

        "zero_day_targets": int(
            target_values.eq(0).sum()
        ),

        "negative_targets": negative_target_count,

        "target_min_days": (
            float(target_values.min())
            if not target_values.empty
            else None
        ),

        "target_median_days": (
            float(target_values.median())
            if not target_values.empty
            else None
        ),

        "target_max_days": (
            float(target_values.max())
            if not target_values.empty
            else None
        ),
    }

    # --------------------------------------------------
    # 8. Return results
    # --------------------------------------------------
    return {
        "purchase_events_with_targets": (
            purchase_events_with_targets
        ),
        "modeling_targets": modeling_targets,
        "target_audit": target_audit,
    }


#------ 4. create_lstm_sequences----------------------------------------------------
def create_lstm_sequences(
    lstm_target_result,
    sequence_length=5,
    feature_columns=None,
):
    """
    Create padded LSTM sequences to predict days until the next purchase.

    Prediction point:
        Immediately after a completed purchase.

    Padding:
        Left-padding with zeros.

    Mask:
        True  = actual purchase event
        False = artificial padding timestep

    Expected input:
        lstm_target_result["modeling_targets"]

    Returns:
        X, X_mask, y, sequence_metadata, feature_columns, sequence_audit
    """

    # ---------------------------------------------------------
    # 1. Validate input
    # ---------------------------------------------------------

    if not isinstance(lstm_target_result, dict):
        raise TypeError("lstm_target_result must be a dictionary.")

    if "modeling_targets" not in lstm_target_result:
        raise ValueError(
            "lstm_target_result must contain 'modeling_targets'."
        )

    df = lstm_target_result["modeling_targets"].copy()

    if not isinstance(df, pd.DataFrame):
        raise TypeError("'modeling_targets' must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("'modeling_targets' is empty.")

    if not isinstance(sequence_length, int) or isinstance(
        sequence_length, bool
    ) or sequence_length < 1:
        raise ValueError("sequence_length must be a positive integer.")

    required_columns = [
        "Customer ID",
        "Invoice_ID",
        "InvoiceDate",
        "Days_Until_Next_Purchase",
        "Purchase_Value",
        "Product_Line_Count",
        "Unique_Product_Count",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # ---------------------------------------------------------
    # 2. Normalize dates internally
    # ---------------------------------------------------------

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce",
    )

    if df["InvoiceDate"].isna().any():
        raise ValueError(
            "InvoiceDate contains missing or invalid dates."
        )

    # ---------------------------------------------------------
    # 3. Validate customer and invoice identifiers
    # ---------------------------------------------------------

    if df["Customer ID"].isna().any():
        raise ValueError(
            "Customer ID contains missing values."
        )

    if df["Customer ID"].eq(0).any():
        raise ValueError(
            "Customer ID 0 must be excluded from the LSTM pipeline."
        )

    if df["Invoice_ID"].isna().any():
        raise ValueError(
            "Invoice_ID contains missing values."
        )

    if df.duplicated(
        subset=["Customer ID", "Invoice_ID"]
    ).any():
        raise ValueError(
            "Duplicate customer-invoice events were found."
        )

    # ---------------------------------------------------------
    # 4. Validate prediction target
    # ---------------------------------------------------------

    df["Days_Until_Next_Purchase"] = pd.to_numeric(
        df["Days_Until_Next_Purchase"],
        errors="coerce",
    )

    if df["Days_Until_Next_Purchase"].isna().any():
        raise ValueError(
            "Days_Until_Next_Purchase contains missing or "
            "non-numeric values."
        )

    if not np.isfinite(
        df["Days_Until_Next_Purchase"].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Prediction targets contain infinite values."
        )

    if (df["Days_Until_Next_Purchase"] < 0).any():
        raise ValueError(
            "Prediction targets cannot be negative."
        )

    # ---------------------------------------------------------
    # 5. Identify model features
    # ---------------------------------------------------------

    base_features = [
        "Purchase_Value",
        "Product_Line_Count",
        "Unique_Product_Count",
    ]

    if "Total_Quantity" in df.columns:
        base_features.append("Total_Quantity")

    category_features = [
        column for column in df.columns
        if column.startswith("Category_")
        and (
            column.endswith("_Line_Count")
            or column.endswith("_Value")
        )
    ]

    if feature_columns is None:
        feature_columns = base_features + category_features
    else:
        feature_columns = list(feature_columns)

    if not feature_columns:
        raise ValueError("No feature columns were selected.")

    missing_features = [
        column for column in feature_columns
        if column not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing feature columns: {missing_features}"
        )

    # Prevent identifiers, dates, and future information from
    # accidentally entering the model as input features.
    forbidden_features = {
        "Customer ID",
        "Invoice_ID",
        "InvoiceDate",
        "Next_Purchase_Date",
        "Next_Invoice_ID",
        "Days_Until_Next_Purchase",
    }

    invalid_features = [
        column for column in feature_columns
        if column in forbidden_features
    ]

    if invalid_features:
        raise ValueError(
            "Identifiers, dates, and targets cannot be features: "
            f"{invalid_features}"
        )

    # ---------------------------------------------------------
    # 6. Normalize and validate numeric features
    # ---------------------------------------------------------

    for column in feature_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    feature_values = df[feature_columns].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(feature_values).all():
        raise ValueError(
            "Feature columns contain missing or infinite values. "
            "Clean or impute them before creating sequences."
        )

    # ---------------------------------------------------------
    # 7. Sort events chronologically within each customer
    # ---------------------------------------------------------

    df = df.sort_values(
        ["Customer ID", "InvoiceDate", "Invoice_ID"],
        kind="mergesort",
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # 8. Generate sequences and explicit padding masks
    # ---------------------------------------------------------

    X_list = []
    X_mask_list = []
    y_list = []
    metadata_list = []

    feature_count = len(feature_columns)

    for customer_id, customer_df in df.groupby(
        "Customer ID",
        sort=False,
    ):
        customer_df = customer_df.reset_index(drop=True)

        customer_features = customer_df[
            feature_columns
        ].to_numpy(dtype=np.float64)

        customer_targets = customer_df[
            "Days_Until_Next_Purchase"
        ].to_numpy(dtype=np.float64)

        for target_idx in range(len(customer_df)):

            # Include the current completed purchase in the input.
            # Its target is the time until the following purchase.
            start_idx = max(
                0,
                target_idx - sequence_length + 1,
            )

            history = customer_features[
                start_idx:target_idx + 1
            ]

            history_length = len(history)

            # Left-pad so all sequences have the same length.
            sequence = np.zeros(
                (sequence_length, feature_count),
                dtype=np.float64,
            )

            sequence[-history_length:] = history

            # True means an actual event; False means padding.
            sequence_mask = np.zeros(
                sequence_length,
                dtype=bool,
            )

            sequence_mask[-history_length:] = True

            X_list.append(sequence)
            X_mask_list.append(sequence_mask)
            y_list.append(customer_targets[target_idx])

            target_row = customer_df.iloc[target_idx]

            metadata_list.append({
                "Customer ID": customer_id,
                "Invoice_ID": target_row["Invoice_ID"],
                "InvoiceDate": target_row["InvoiceDate"],
                "Next_Purchase_Date": pd.to_datetime(
                     target_row["Next_Purchase_Date"],
                     errors="coerce", ),

                "Days_Until_Next_Purchase": (
                    customer_targets[target_idx]
                ),
                "History_Length": history_length,
                "Padding_Length": (
                    sequence_length - history_length
                ),
            })

    # ---------------------------------------------------------
    # 9. Convert outputs to arrays and DataFrame
    # ---------------------------------------------------------

    X = np.asarray(X_list, dtype=np.float64)

    X_mask = np.asarray(
        X_mask_list,
        dtype=bool,
    )

    y = np.asarray(
        y_list,
        dtype=np.float64,
    )

    sequence_metadata = pd.DataFrame(metadata_list)

    # ---------------------------------------------------------
    # 10. Validate sequence integrity
    # ---------------------------------------------------------

    if X.shape != (
        len(df),
        sequence_length,
        feature_count,
    ):
        raise AssertionError(
            f"Unexpected X shape: {X.shape}"
        )

    if X_mask.shape != X.shape[:2]:
        raise AssertionError(
            "Padding mask dimensions do not match X."
        )

    if X_mask.dtype != bool:
        raise AssertionError(
            "X_mask must have Boolean dtype."
        )

    if not X_mask.any(axis=1).all():
        raise AssertionError(
            "A sequence contains no actual purchase events."
        )

    if not np.isfinite(X).all():
        raise AssertionError(
            "Generated sequences contain invalid values."
        )

    if not np.isfinite(y).all():
        raise AssertionError(
            "Generated targets contain invalid values."
        )

    # Every artificial padding timestep must remain zero.
    if not np.all(X[~X_mask] == 0):
        raise AssertionError(
            "Artificial padding values must be zero."
        )

    # Verify each sequence's mask matches its history length.
    observed_history_lengths = X_mask.sum(axis=1)

    expected_history_lengths = sequence_metadata[
        "History_Length"
    ].to_numpy()

    if not np.array_equal(
        observed_history_lengths,
        expected_history_lengths,
    ):
        raise AssertionError(
            "Padding masks do not match sequence history lengths."
        )

    # ---------------------------------------------------------
    # 11. Build audit report
    # ---------------------------------------------------------

    short_history_count = int(
        (sequence_metadata["History_Length"] < sequence_length).sum()
    )

    sequence_audit = {
        "input_target_rows": int(len(df)),
        "sequence_count": int(len(X)),
        "sequence_length": int(sequence_length),
        "feature_count": int(feature_count),
        "unique_customers": int(df["Customer ID"].nunique()),
        "X_shape": tuple(X.shape),
        "X_mask_shape": tuple(X_mask.shape),
        "X_mask_dtype": str(X_mask.dtype),
        "y_shape": tuple(y.shape),
        "short_history_sequences": short_history_count,
        "sequences_with_padding": int(
            (sequence_metadata["Padding_Length"] > 0).sum()
        ),
        "sequences_without_padding": int(
            (sequence_metadata["Padding_Length"] == 0).sum()
        ),
        "missing_values_in_X": int(np.isnan(X).sum()),
        "missing_values_in_y": int(np.isnan(y).sum()),
        "padding_values_are_zero": bool(
            np.all(X[~X_mask] == 0)
        ),
        "mask_validation_passed": True,
        "target_min_days": float(y.min()),
        "target_median_days": float(np.median(y)),
        "target_max_days": float(y.max()),
        "feature_columns": feature_columns,
    }

    # ---------------------------------------------------------
    # 12. Return results
    # ---------------------------------------------------------

    return {
        "X": X,
        "X_mask": X_mask,
        "y": y,
        "sequence_metadata": sequence_metadata,
        "feature_columns": feature_columns,
        "sequence_audit": sequence_audit,
    }


#------ 5. Spliting chronological data ----------------------------------------------------

def split_lstm_sequences_chronologically(
    sequence_result,
    train_fraction=0.70,
    validation_fraction=0.15,
    min_validation_fraction=0.10,
    min_test_fraction=0.10,
    cutoff_search_points=60,
):
    """
    Chronological LSTM split with boundary purging and balanced cutoffs.

    The function:
        1. Searches for better chronological cutoffs.
        2. Purges training/validation samples whose next purchase
           crosses the relevant partition boundary.
        3. Enforces minimum validation and test sample proportions.
        4. Selects cutoffs that minimize deviation from requested
           final train/validation/test proportions.
        5. Fits StandardScaler on valid training timesteps only.
        6. Preserves zero padding and sequence masks.

    Required sequence_result keys:
        X, X_mask, y, sequence_metadata, feature_columns

    Required metadata columns:
        InvoiceDate, Next_Purchase_Date
    """

    # =========================================================
    # 1. Validate parameters and input
    # =========================================================

    if not isinstance(sequence_result, dict):
        raise TypeError("sequence_result must be a dictionary.")

    required_keys = {
        "X", "X_mask", "y", "sequence_metadata", "feature_columns"
    }

    missing_keys = required_keys - set(sequence_result)

    if missing_keys:
        raise ValueError(
            f"Missing sequence_result keys: {sorted(missing_keys)}"
        )

    if not (0 < train_fraction < 1):
        raise ValueError("train_fraction must be between 0 and 1.")

    if not (0 < validation_fraction < 1):
        raise ValueError("validation_fraction must be between 0 and 1.")

    if train_fraction + validation_fraction >= 1:
        raise ValueError(
            "train_fraction + validation_fraction must be less than 1."
        )

    if not (0 < min_validation_fraction < 1):
        raise ValueError(
            "min_validation_fraction must be between 0 and 1."
        )

    if not (0 < min_test_fraction < 1):
        raise ValueError(
            "min_test_fraction must be between 0 and 1."
        )

    if min_validation_fraction + min_test_fraction >= 1:
        raise ValueError(
            "Minimum validation and test fractions must sum to less than 1."
        )

    if (
        not isinstance(cutoff_search_points, int)
        or isinstance(cutoff_search_points, bool)
        or cutoff_search_points < 3
    ):
        raise ValueError("cutoff_search_points must be an integer >= 3.")

    X = np.asarray(sequence_result["X"], dtype=np.float64)
    X_mask = np.asarray(sequence_result["X_mask"], dtype=bool)
    y = np.asarray(sequence_result["y"], dtype=np.float64).reshape(-1)

    metadata = sequence_result["sequence_metadata"].copy()
    feature_columns = list(sequence_result["feature_columns"])

    if X.ndim != 3 or len(X) == 0:
        raise ValueError("X must be a non-empty 3D array.")

    if X_mask.shape != X.shape[:2]:
        raise ValueError("X_mask dimensions must match X samples and timesteps.")

    if len(y) != len(X) or len(metadata) != len(X):
        raise ValueError("X, y, and metadata must have equal sample counts.")

    if X.shape[2] != len(feature_columns):
        raise ValueError("Feature count does not match feature_columns.")

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X and y must contain only finite values.")

    if (y < 0).any():
        raise ValueError("Targets cannot be negative.")

    if not X_mask.any(axis=1).all():
        raise ValueError("Every sequence must contain a real timestep.")

    if not np.all(X[~X_mask] == 0):
        raise ValueError("Input padding must be zero.")

    required_metadata = {"InvoiceDate", "Next_Purchase_Date"}
    missing_metadata = required_metadata - set(metadata.columns)

    if missing_metadata:
        raise ValueError(
            f"Missing metadata columns: {sorted(missing_metadata)}. "
            "Regenerate sequences with Next_Purchase_Date."
        )

    metadata["InvoiceDate"] = pd.to_datetime(
        metadata["InvoiceDate"], errors="coerce"
    )
    metadata["Next_Purchase_Date"] = pd.to_datetime(
        metadata["Next_Purchase_Date"], errors="coerce"
    )

    if metadata[list(required_metadata)].isna().any().any():
        raise ValueError("Event and next-purchase dates must be valid.")

    if (
        metadata["Next_Purchase_Date"] < metadata["InvoiceDate"]
    ).any():
        raise ValueError("Next-purchase dates cannot precede event dates.")

    # =========================================================
    # 2. Prepare date arrays and requested proportions
    # =========================================================

    total = len(X)

    target_test_fraction = 1.0 - train_fraction - validation_fraction

    event_dates = metadata["InvoiceDate"].to_numpy(
        dtype="datetime64[ns]"
    )
    next_dates = metadata["Next_Purchase_Date"].to_numpy(
        dtype="datetime64[ns]"
    )

    unique_dates = np.sort(metadata["InvoiceDate"].unique())

    if len(unique_dates) < 10:
        raise ValueError(
            "Insufficient distinct event timestamps for cutoff optimization."
        )

    # Keep the search computationally manageable.
    candidate_positions = np.unique(
        np.linspace(
            1,
            len(unique_dates) - 2,
            num=min(cutoff_search_points, len(unique_dates) - 2),
            dtype=int,
        )
    )

    candidate_cutoffs = unique_dates[candidate_positions].astype(
        "datetime64[ns]"
    )

    # The minimums apply to proportions of all original sequences.
    min_validation_count = int(
        np.ceil(total * min_validation_fraction)
    )
    min_test_count = int(
        np.ceil(total * min_test_fraction)
    )

    target_counts = np.array([
        train_fraction * total,
        validation_fraction * total,
        target_test_fraction * total,
    ])

    # =========================================================
    # 3. Search cutoffs using final counts after purging
    # =========================================================

    best = None
    best_score = float("inf")
    valid_candidate_count = 0

    for train_cutoff in candidate_cutoffs:

        # A training sample is retained only when both its event
        # and next-purchase outcome precede the training cutoff.
        train_keep = (
            (event_dates < train_cutoff)
            & (next_dates < train_cutoff)
        )

        train_count = int(train_keep.sum())

        if train_count == 0:
            continue

        for validation_cutoff in candidate_cutoffs:

            if validation_cutoff <= train_cutoff:
                continue

            validation_keep = (
                (event_dates >= train_cutoff)
                & (event_dates < validation_cutoff)
                & (next_dates < validation_cutoff)
            )

            test_keep = event_dates >= validation_cutoff

            validation_count = int(validation_keep.sum())
            test_count = int(test_keep.sum())

            # Enforce minimum validation/test sample sizes.
            if validation_count < min_validation_count:
                continue

            if test_count < min_test_count:
                continue

            if train_count == 0:
                continue

            valid_candidate_count += 1

            actual_counts = np.array([
                train_count,
                validation_count,
                test_count,
            ], dtype=float)

            # Minimize the total absolute deviation from requested
            # final counts after boundary purging.
            #
            # Every partition contributes equally to the score.
            score = np.abs(
                actual_counts - target_counts
            ).sum() / total

            if score < best_score:
                best_score = score

                best = {
                    "train_cutoff": pd.Timestamp(train_cutoff),
                    "validation_cutoff": pd.Timestamp(validation_cutoff),
                    "train_keep": train_keep.copy(),
                    "validation_keep": validation_keep.copy(),
                    "test_keep": test_keep.copy(),
                    "train_candidate_count": int(
                        (event_dates < train_cutoff).sum()
                    ),
                    "validation_candidate_count": int(
                        (
                            (event_dates >= train_cutoff)
                            & (event_dates < validation_cutoff)
                        ).sum()
                    ),
                }

    if best is None:
        raise ValueError(
            "No valid chronological cutoff pair satisfies the minimum "
            "validation/test sizes. Try lowering min_validation_fraction "
            "or min_test_fraction, or increasing cutoff_search_points."
        )

    # =========================================================
    # 4. Extract final partitions
    # =========================================================

    train_indices = np.flatnonzero(best["train_keep"])
    validation_indices = np.flatnonzero(best["validation_keep"])
    test_indices = np.flatnonzero(best["test_keep"])

    X_train_raw = X[train_indices].copy()
    X_validation_raw = X[validation_indices].copy()
    X_test_raw = X[test_indices].copy()

    mask_train = X_mask[train_indices].copy()
    mask_validation = X_mask[validation_indices].copy()
    mask_test = X_mask[test_indices].copy()

    y_train = y[train_indices].copy()
    y_validation = y[validation_indices].copy()
    y_test = y[test_indices].copy()

    train_metadata = (
        metadata.iloc[train_indices].copy().reset_index(drop=True)
    )
    validation_metadata = (
        metadata.iloc[validation_indices].copy().reset_index(drop=True)
    )
    test_metadata = (
        metadata.iloc[test_indices].copy().reset_index(drop=True)
    )

    # =========================================================
    # 5. Fit scaler using valid training timesteps only
    # =========================================================

    scaler = StandardScaler()

    training_valid_timesteps = X_train_raw[mask_train]

    if len(training_valid_timesteps) == 0:
        raise ValueError("No valid training timesteps are available.")

    scaler.fit(training_valid_timesteps)

    def transform_valid_timesteps(X_part, mask_part):
        """Scale real events while keeping padding exactly zero."""

        X_scaled = np.zeros_like(X_part, dtype=np.float64)

        for sample_index in range(len(X_part)):
            valid = mask_part[sample_index]

            if valid.any():
                X_scaled[sample_index, valid, :] = scaler.transform(
                    X_part[sample_index, valid, :]
                )

        return X_scaled

    X_train = transform_valid_timesteps(X_train_raw, mask_train)
    X_validation = transform_valid_timesteps(
        X_validation_raw, mask_validation
    )
    X_test = transform_valid_timesteps(X_test_raw, mask_test)

    # =========================================================
    # 6. Validate temporal boundaries and masks
    # =========================================================

    train_cutoff = best["train_cutoff"]
    validation_cutoff = best["validation_cutoff"]

    train_valid = bool(
        (train_metadata["InvoiceDate"] < train_cutoff).all()
        and (
            train_metadata["Next_Purchase_Date"] < train_cutoff
        ).all()
    )

    validation_valid = bool(
        (validation_metadata["InvoiceDate"] >= train_cutoff).all()
        and (
            validation_metadata["InvoiceDate"] < validation_cutoff
        ).all()
        and (
            validation_metadata["Next_Purchase_Date"]
            < validation_cutoff
        ).all()
    )

    test_valid = bool(
        (test_metadata["InvoiceDate"] >= validation_cutoff).all()
    )

    padding_valid = bool(
        np.all(X_train[~mask_train] == 0)
        and np.all(X_validation[~mask_validation] == 0)
        and np.all(X_test[~mask_test] == 0)
    )

    minimum_sizes_valid = bool(
        len(validation_indices) >= min_validation_count
        and len(test_indices) >= min_test_count
    )

    if not all([
        train_valid,
        validation_valid,
        test_valid,
        padding_valid,
        minimum_sizes_valid,
    ]):
        raise AssertionError(
            "A temporal, minimum-size, or padding validation check failed."
        )

    # =========================================================
    # 7. Build audit
    # =========================================================

    actual_counts = {
        "train": len(train_indices),
        "validation": len(validation_indices),
        "test": len(test_indices),
    }

    audit = {
        "total_sequences_before_split": int(total),

        "train_candidate_sequences": best["train_candidate_count"],
        "validation_candidate_sequences": (
            best["validation_candidate_count"]
        ),
        "test_candidate_sequences": int(len(test_indices)),

        "train_sequences_purged": int(
            best["train_candidate_count"] - len(train_indices)
        ),
        "validation_sequences_purged": int(
            best["validation_candidate_count"] - len(validation_indices)
        ),

        "train_sequences_final": int(len(train_indices)),
        "validation_sequences_final": int(len(validation_indices)),
        "test_sequences_final": int(len(test_indices)),

        "train_fraction_requested": float(train_fraction),
        "validation_fraction_requested": float(validation_fraction),
        "test_fraction_requested": float(target_test_fraction),

        "train_fraction_actual": round(len(train_indices) / total, 4),
        "validation_fraction_actual": round(
            len(validation_indices) / total, 4
        ),
        "test_fraction_actual": round(len(test_indices) / total, 4),

        "minimum_validation_fraction": float(min_validation_fraction),
        "minimum_validation_sequences": int(min_validation_count),
        "minimum_test_fraction": float(min_test_fraction),
        "minimum_test_sequences": int(min_test_count),

        "train_cutoff_exclusive": str(train_cutoff),
        "validation_cutoff_exclusive": str(validation_cutoff),

        "cutoff_search_points": int(len(candidate_cutoffs)),
        "valid_cutoff_pairs_evaluated": int(valid_candidate_count),
        "cutoff_optimization_score": float(best_score),

        "feature_count": int(X.shape[2]),
        "sequence_length": int(X.shape[1]),
        "scaler_fitted_on": (
            "Training partition, valid timesteps only"
        ),

        "training_temporal_validation_passed": train_valid,
        "validation_temporal_validation_passed": validation_valid,
        "test_temporal_validation_passed": test_valid,
        "padding_values_remain_zero": padding_valid,
        "minimum_partition_sizes_passed": minimum_sizes_valid,

        "train_target_median_days": float(np.median(y_train)),
        "validation_target_median_days": float(np.median(y_validation)),
        "test_target_median_days": float(np.median(y_test)),
    }

    return {
        "X_train": X_train,
        "X_validation": X_validation,
        "X_test": X_test,

        "X_train_raw": X_train_raw,
        "X_validation_raw": X_validation_raw,
        "X_test_raw": X_test_raw,

        "mask_train": mask_train,
        "mask_validation": mask_validation,
        "mask_test": mask_test,

        "y_train": y_train,
        "y_validation": y_validation,
        "y_test": y_test,

        "train_metadata": train_metadata,
        "validation_metadata": validation_metadata,
        "test_metadata": test_metadata,

        "feature_columns": feature_columns,
        "scaler": scaler,

        "cutoffs": {
            "train_cutoff": train_cutoff,
            "validation_cutoff": validation_cutoff,
        },

        "audit": audit,
    }


#--------6. maste pipeline function ----------------------------------------------------------------
def run_lstm_data_pipeline(
    df_transactions,
    sequence_length=5,
    train_fraction=0.70,
    validation_fraction=0.15,
    taxonomy_result=None,
):
    """
    Run the complete LSTM data-preparation pipeline.

    Pipeline stages:
        1. Analyze product taxonomy.
        2. Prepare eligible retail transactions.
        3. Build customer purchase events.
        4. Create next-purchase timing targets.
        5. Generate padded LSTM sequences.
        6. Split chronologically and scale features.

    Parameters
    ----------
    df_transactions : pandas.DataFrame
        Cleaned transaction-level dataset.

    sequence_length : int, default=5
        Number of timesteps in each LSTM sequence.

    train_fraction : float, default=0.70
        Requested chronological training fraction.

    validation_fraction : float, default=0.15
        Requested chronological validation fraction.
        The remaining fraction is assigned to test.

    taxonomy_result : dict or None, default=None
        Optional precomputed taxonomy result. If None, taxonomy
        is generated from df_transactions.

    Returns
    -------
    dict
        Intermediate outputs, final split datasets, and pipeline audit.
    """

    import pandas as pd

    # ---------------------------------------------------------
    # 1. Validate input
    # ---------------------------------------------------------

    if not isinstance(df_transactions, pd.DataFrame):
        raise TypeError(
            "df_transactions must be a pandas DataFrame."
        )

    if df_transactions.empty:
        raise ValueError(
            "df_transactions is empty."
        )

    if not isinstance(sequence_length, int) or isinstance(
        sequence_length, bool
    ) or sequence_length < 1:
        raise ValueError(
            "sequence_length must be a positive integer."
        )

    if not (
        isinstance(train_fraction, (int, float))
        and not isinstance(train_fraction, bool)
        and 0 < train_fraction < 1
    ):
        raise ValueError(
            "train_fraction must be between 0 and 1."
        )

    if not (
        isinstance(validation_fraction, (int, float))
        and not isinstance(validation_fraction, bool)
        and 0 < validation_fraction < 1
    ):
        raise ValueError(
            "validation_fraction must be between 0 and 1."
        )

    if train_fraction + validation_fraction >= 1:
        raise ValueError(
            "train_fraction + validation_fraction must be less than 1."
        )

    pipeline_audit = {
        "pipeline_name": "Vantara LSTM Data Preparation",
        "status": "running",
        "input_transaction_rows": int(len(df_transactions)),
        "sequence_length_requested": sequence_length,
        "train_fraction_requested": float(train_fraction),
        "validation_fraction_requested": float(
            validation_fraction
        ),
        "test_fraction_requested": float(
            1 - train_fraction - validation_fraction
        ),
        "completed_stages": [],
    }

    # ---------------------------------------------------------
    # 2. Product taxonomy
    # ---------------------------------------------------------

    if taxonomy_result is None:
        taxonomy_result = analyze_product_taxonomy(
            df_transactions
        )

    if not isinstance(taxonomy_result, dict):
        raise TypeError(
            "analyze_product_taxonomy must return a dictionary."
        )

    pipeline_audit["completed_stages"].append(
        "product_taxonomy"
    )

    # ---------------------------------------------------------
    # 3. Prepare eligible retail transactions
    # ---------------------------------------------------------

    lstm_taxonomy_result = prepare_lstm_product_taxonomy(
        df_transactions=df_transactions,
        taxonomy_result=taxonomy_result,
    )

    if not isinstance(lstm_taxonomy_result, dict):
        raise TypeError(
            "prepare_lstm_product_taxonomy must return a dictionary."
        )

    pipeline_audit["completed_stages"].append(
        "eligible_retail_transactions"
    )

    # ---------------------------------------------------------
    # 4. Aggregate purchase events
    # ---------------------------------------------------------

    lstm_event_result = build_lstm_purchase_events(
        lstm_taxonomy_result=lstm_taxonomy_result,
    )

    if not isinstance(lstm_event_result, dict):
        raise TypeError(
            "build_lstm_purchase_events must return a dictionary."
        )

    if "purchase_events" not in lstm_event_result:
        raise KeyError(
            "Event output does not contain 'purchase_events'."
        )

    purchase_events = lstm_event_result["purchase_events"]

    if purchase_events.empty:
        raise ValueError(
            "No purchase events were generated."
        )

    pipeline_audit["completed_stages"].append(
        "purchase_event_aggregation"
    )

    # ---------------------------------------------------------
    # 5. Create next-purchase timing targets
    # ---------------------------------------------------------

    lstm_target_result = create_lstm_purchase_timing_targets(
        purchase_events
    )

    if not isinstance(lstm_target_result, dict):
        raise TypeError(
            "create_lstm_purchase_timing_targets must return a dictionary."
        )

    if "modeling_targets" not in lstm_target_result:
        raise KeyError(
            "Target output does not contain 'modeling_targets'."
        )

    modeling_targets = lstm_target_result["modeling_targets"]

    if modeling_targets.empty:
        raise ValueError(
            "No usable next-purchase targets were generated."
        )

    pipeline_audit["completed_stages"].append(
        "next_purchase_target_generation"
    )

    # ---------------------------------------------------------
    # 6. Generate LSTM sequences and padding masks
    # ---------------------------------------------------------

    lstm_sequence_result = create_lstm_sequences(
        lstm_target_result=lstm_target_result,
        sequence_length=sequence_length,
    )

    if not isinstance(lstm_sequence_result, dict):
        raise TypeError(
            "create_lstm_sequences must return a dictionary."
        )

    required_sequence_keys = {
        "X",
        "X_mask",
        "y",
        "sequence_metadata",
        "feature_columns",
        "sequence_audit",
    }

    missing_sequence_keys = (
        required_sequence_keys
        - set(lstm_sequence_result.keys())
    )

    if missing_sequence_keys:
        raise KeyError(
            "Sequence output is missing required keys: "
            f"{sorted(missing_sequence_keys)}"
        )

    sequence_metadata = lstm_sequence_result[
        "sequence_metadata"
    ]

    required_metadata_columns = {
        "InvoiceDate",
        "Next_Purchase_Date",
    }

    missing_metadata_columns = (
        required_metadata_columns
        - set(sequence_metadata.columns)
    )

    if missing_metadata_columns:
        raise ValueError(
            "Sequence metadata is missing required columns: "
            f"{sorted(missing_metadata_columns)}. "
            "Update create_lstm_sequences() first."
        )

    pipeline_audit["completed_stages"].append(
        "lstm_sequence_generation"
    )

    # ---------------------------------------------------------
    # 7. Chronological split and training-only scaling
    # ---------------------------------------------------------

    lstm_split_result = split_lstm_sequences_chronologically(
        sequence_result=lstm_sequence_result,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )

    if not isinstance(lstm_split_result, dict):
        raise TypeError(
            "split_lstm_sequences_chronologically must return a dictionary."
        )

    required_split_keys = {
        "X_train",
        "X_validation",
        "X_test",
        "mask_train",
        "mask_validation",
        "mask_test",
        "y_train",
        "y_validation",
        "y_test",
        "scaler",
        "audit",
    }

    missing_split_keys = (
        required_split_keys
        - set(lstm_split_result.keys())
    )

    if missing_split_keys:
        raise KeyError(
            "Split output is missing required keys: "
            f"{sorted(missing_split_keys)}"
        )

    split_audit = lstm_split_result["audit"]

    required_audit_checks = [
        "training_temporal_validation_passed",
        "validation_temporal_validation_passed",
        "test_temporal_validation_passed",
        "padding_values_remain_zero",
    ]

    failed_checks = [
        check
        for check in required_audit_checks
        if split_audit.get(check) is not True
    ]

    if failed_checks:
        raise AssertionError(
            "Split validation failed: "
            f"{failed_checks}"
        )

    pipeline_audit["completed_stages"].append(
        "chronological_split_and_scaling"
    )

    # ---------------------------------------------------------
    # 8. Final summary and return
    # ---------------------------------------------------------

    pipeline_audit["status"] = "success"

    pipeline_audit["final_sequence_count"] = int(
        len(lstm_sequence_result["X"])
    )

    pipeline_audit["feature_count"] = int(
        lstm_sequence_result["X"].shape[2]
    )

    pipeline_audit["final_split_audit"] = split_audit

    return {
        # Taxonomy
        "taxonomy": taxonomy_result,

        # Eligible transaction-level data
        "lstm_taxonomy": lstm_taxonomy_result,

        # Invoice-level events
        "purchase_events": purchase_events,
        "event_result": lstm_event_result,

        # Target generation
        "target_result": lstm_target_result,
        "modeling_targets": modeling_targets,

        # Sequence data
        "sequence_result": lstm_sequence_result,

        # Final model-ready datasets
        "split_result": lstm_split_result,

        # Overall pipeline report
        "audit": pipeline_audit,
    }



################################################################################################
#===============================================================================================
# ------ FEATURE ENGINEERING SEPARATE FOR PRODUCT PURCHSE TIMMING BEHAVIOUR---------------------
# -------FOR LSTM Sequential model
#===============================================================================================
################################################################################################

#--------- 1. feature selction for autoencoder
def select_autoencoder_features(
    customer_features: pd.DataFrame,
    excluded_columns: list[str],
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Select numeric features for autoencoder anomaly detection
    by excluding caller-specified columns.

    Returns
    -------
    customer_ids : pd.Series
        Customer IDs retained for identifying anomalies.

    X : pd.DataFrame
        Numeric feature dataframe for the autoencoder.
    """

    if "Customer ID" not in customer_features.columns:
        raise ValueError(
            "Customer ID column is missing."
        )

    customer_ids = customer_features[
        "Customer ID"
    ].copy()

    X = customer_features.drop(
        columns=excluded_columns,
        errors="ignore",
    ).copy()

    X = X.select_dtypes(
        include="number"
    )

    if X.empty:
        raise ValueError(
            "No numeric features available for the autoencoder."
        )

    return customer_ids, X

#--------- 2. validate auto encoder feattures
def validate_autoencoder_features(
    X: pd.DataFrame,
) -> dict:
    """
    Validate numeric input features before autoencoder training.

    Raises an error if the dataset is empty, contains missing
    values, contains infinite values, or has no usable features.

    Constant features are reported for review.
    """

    if X.empty:
        raise ValueError("Feature dataset is empty.")

    if X.columns.duplicated().any():
        raise ValueError("Duplicate feature columns detected.")

    non_numeric = X.select_dtypes(
        exclude="number"
    ).columns.tolist()

    if non_numeric:
        raise ValueError(
            f"Non-numeric features detected: {non_numeric}"
        )

    if X.isna().any().any():
        missing = X.columns[
            X.isna().any()
        ].tolist()

        raise ValueError(
            f"Missing values detected in: {missing}"
        )

    values = X.to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        infinite_features = X.columns[
            ~np.isfinite(values).all(axis=0)
        ].tolist()

        raise ValueError(
            f"Infinite or invalid values detected in: "
            f"{infinite_features}"
        )

    constant_features = X.columns[
        X.nunique(dropna=False) <= 1
    ].tolist()

    validation_result = {
        "row_count": len(X),
        "feature_count": X.shape[1],
        "missing_values": 0,
        "infinite_values": 0,
        "constant_features": constant_features,
        "is_valid": True,
    }

    print("Autoencoder feature validation passed.")
    print(f"Customers: {validation_result['row_count']}")
    print(f"Features: {validation_result['feature_count']}")
    print(f"Constant features: {constant_features}")

    return validation_result

#--------- 3. Splitting the dataser into train,validation and test------
def split_autoencoder_dataset(
    customer_ids: pd.Series,
    X: pd.DataFrame,
    random_state: int = 42,
) -> dict:
    """
    Split customer-level features into training, validation,
    and test sets using 70% / 15% / 15%.

    Customer IDs remain separate from model inputs.
    """

    if len(customer_ids) != len(X):
        raise ValueError(
            "Customer IDs and feature rows must have equal lengths."
        )

    if not customer_ids.index.equals(X.index):
        raise ValueError(
            "Customer IDs and features must have matching indexes."
        )

    if customer_ids.isna().any():
        raise ValueError("Customer IDs contain missing values.")

    if customer_ids.duplicated().any():
        raise ValueError("Duplicate customer IDs detected.")

    # 70% training, 30% temporary
    X_train, X_temp, ids_train, ids_temp = train_test_split(
        X,
        customer_ids,
        test_size=0.30,
        random_state=random_state,
        shuffle=True,
    )

    # Divide the remaining 30% equally: 15% validation, 15% test
    X_validation, X_test, ids_validation, ids_test = (
        train_test_split(
            X_temp,
            ids_temp,
            test_size=0.50,
            random_state=random_state,
            shuffle=True,
        )
    )

    return {
        "X_train": X_train,
        "X_validation": X_validation,
        "X_test": X_test,
        "customer_ids_train": ids_train,
        "customer_ids_validation": ids_validation,
        "customer_ids_test": ids_test,
    }


#--------- 4. fir scaler on Dataset
def scale_autoencoder_features(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
) -> dict:
    """
    Scale autoencoder features using training data only.

    Returns the scaled datasets, fitted scaler, and feature order.
    """

    # Ensure identical features in identical order
    if not X_train.columns.equals(X_validation.columns):
        raise ValueError(
            "Training and validation feature columns do not match."
        )

    if not X_train.columns.equals(X_test.columns):
        raise ValueError(
            "Training and test feature columns do not match."
        )

    scaler = StandardScaler()

    # Fit only on training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply the same scaler to validation and test data
    X_validation_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)

    # Preserve feature names and row indexes
    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
        index=X_train.index,
    )

    X_validation_scaled = pd.DataFrame(
        X_validation_scaled,
        columns=X_validation.columns,
        index=X_validation.index,
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns,
        index=X_test.index,
    )

    return {
        "X_train_scaled": X_train_scaled,
        "X_validation_scaled": X_validation_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "feature_columns": X_train.columns.tolist(),
    }
