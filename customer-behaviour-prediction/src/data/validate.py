from typing import Dict, Any
import pandas as pd
from IPython.display import display

EXPECTED_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate the Online Retail II dataset.

    This function performs structural and data-quality checks
    without modifying the original DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Combined Online Retail II transaction dataset.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing validation results and summary statistics.
    """

    # ---------------------------------------------------------
    # 1. Basic input validation
    # ---------------------------------------------------------

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("The dataset is empty.")

    # ---------------------------------------------------------
    # 2. Column validation
    # ---------------------------------------------------------

    actual_columns = list(df.columns)

    missing_columns = [column for column in EXPECTED_COLUMNS if column not in actual_columns]
    unexpected_columns = [column for column in actual_columns if column not in EXPECTED_COLUMNS]

    columns_valid = len(missing_columns) == 0

    # ---------------------------------------------------------
    # 3. Missing value analysis
    # ---------------------------------------------------------

    missing_values = df.isna().sum()
    missing_percentage = (df.isna().mean() * 100).round(2)

    # ---------------------------------------------------------
    # 4. Duplicate analysis
    # ---------------------------------------------------------

    duplicate_rows = int(df.duplicated().sum())

    # ---------------------------------------------------------
    # 5. Date validation
    # ---------------------------------------------------------

    date_range = {}

    if "InvoiceDate" in df.columns:

        invoice_dates = pd.to_datetime(df["InvoiceDate"],errors="coerce")
        invalid_dates = int(invoice_dates.isna().sum())

        if invoice_dates.notna().any():

            min_date = invoice_dates.min()
            max_date = invoice_dates.max()

            date_range = {
                "min_date": min_date,
                "max_date": max_date,
                "invalid_dates": invalid_dates,
            }

        else:
            date_range = {
                "min_date": None,
                "max_date": None,
                "invalid_dates": invalid_dates,
            }

    # ---------------------------------------------------------
    # 6. Quantity validation
    # ---------------------------------------------------------

    negative_quantity = None
    zero_quantity = None

    if "Quantity" in df.columns:

        negative_quantity = int((df["Quantity"] < 0).sum())
        zero_quantity = int((df["Quantity"] == 0).sum())

    # ---------------------------------------------------------
    # 7. Price validation
    # ---------------------------------------------------------

    zero_price = None
    negative_price = None

    if "Price" in df.columns:
        zero_price = int((df["Price"] == 0).sum())
        negative_price = int((df["Price"] < 0).sum())

    # ---------------------------------------------------------
    # 8. Customer validation
    # ---------------------------------------------------------

    missing_customer_id = None
    unique_customers = None

    if "Customer ID" in df.columns:
        missing_customer_id = int(df["Customer ID"].isna().sum())
        unique_customers = int(df["Customer ID"].nunique(dropna=True))

    # ---------------------------------------------------------
    # 9. Invoice validation
    # ---------------------------------------------------------

    unique_invoices = None

    if "Invoice" in df.columns:
        unique_invoices = int(df["Invoice"].nunique())

    # ---------------------------------------------------------
    # 10. Product / StockCode validation
    # ---------------------------------------------------------

    unique_stock_codes = None
    missing_stock_codes = None

    if "StockCode" in df.columns:

        unique_stock_codes = int(df["StockCode"].nunique())
        missing_stock_codes = int(df["StockCode"].isna().sum())

    # ---------------------------------------------------------
    # 11. Country validation
    # ---------------------------------------------------------

    unique_countries = None

    if "Country" in df.columns:
        unique_countries = int(df["Country"].nunique())

    # ---------------------------------------------------------
    # 12. Data type validation
    # ---------------------------------------------------------

    data_types = {
        column: str(dtype)
        for column, dtype in df.dtypes.items()
    }

    # ---------------------------------------------------------
    # 13. Overall validation status
    # ---------------------------------------------------------

    validation_passed = (
        columns_valid
        and duplicate_rows == 0
        and (date_range.get("invalid_dates", 0) == 0)
    )

    # ---------------------------------------------------------
    # 14. Create validation report
    # ---------------------------------------------------------

    report = {

        "validation_passed": validation_passed,

        "dataset": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },

        "columns": {
            "expected": EXPECTED_COLUMNS,
            "actual": actual_columns,
            "missing_columns": missing_columns,
            "unexpected_columns": unexpected_columns,
            "columns_valid": columns_valid,
        },

        "missing_values": {
            column: {
                "count": int(missing_values[column]),
                "percentage": float(missing_percentage[column]),
            }
            for column in actual_columns
            if missing_values[column] > 0
        },

        "duplicates": {
            "duplicate_rows": duplicate_rows,
        },

        "dates": date_range,

        "quantity": {
            "negative_quantity": negative_quantity,
            "zero_quantity": zero_quantity,
        },

        "price": {
            "zero_price": zero_price,
            "negative_price": negative_price,
        },

        "customers": {
            "unique_customers": unique_customers,
            "missing_customer_id": missing_customer_id,
        },

        "invoices": {
            "unique_invoices": unique_invoices,
        },

        "products": {
            "unique_stock_codes": unique_stock_codes,
            "missing_stock_codes": missing_stock_codes,
        },

        "countries": {
            "unique_countries": unique_countries,
        },

        "data_types": data_types,
    }

    return report

################################################################################################
#--------------------------------- display validation report -----------------------------------
################################################################################################
def display_validation_report(report: Dict[str, Any]) -> None:
    """
    Display the dataset validation report in a
    clean, human-readable format.

    Parameters
    ----------
    report : Dict[str, Any]
        Validation report returned by validate_dataset().
    """

    print("=" * 70)
    print("VANTARA CUSTOMER BEHAVIOR PLATFORM")
    print("DATASET VALIDATION REPORT")
    print("=" * 70)

    # ---------------------------------------------------------
    # Dataset Overview
    # ---------------------------------------------------------

    dataset = report["dataset"]

    print("\nDATASET OVERVIEW")
    print("-" * 70)

    print(f"{'Rows':<35}: {dataset['rows']:,}")
    print(f"{'Columns':<35}: {dataset['columns']}")

    # Customers
    customers = report["customers"]

    print(
        f"{'Unique Customers':<35}: "
        f"{customers['unique_customers']:,}"
        if customers["unique_customers"] is not None
        else f"{'Unique Customers':<35}: N/A"
    )

    # Invoices
    invoices = report["invoices"]

    print(
        f"{'Unique Invoices':<35}: "
        f"{invoices['unique_invoices']:,}"
        if invoices["unique_invoices"] is not None
        else f"{'Unique Invoices':<35}: N/A"
    )

    # Products
    products = report["products"]

    print(
        f"{'Unique Products':<35}: "
        f"{products['unique_stock_codes']:,}"
        if products["unique_stock_codes"] is not None
        else f"{'Unique Products':<35}: N/A"
    )

    # Countries
    countries = report["countries"]

    print(
        f"{'Countries':<35}: "
        f"{countries['unique_countries']:,}"
        if countries["unique_countries"] is not None
        else f"{'Countries':<35}: N/A"
    )

    # ---------------------------------------------------------
    # Date Range
    # ---------------------------------------------------------

    dates = report["dates"]

    print("\nDATE RANGE")
    print("-" * 70)

    if dates.get("min_date") is not None:

        print(
            f"{'Start Date':<35}: "
            f"{dates['min_date'].strftime('%Y-%m-%d')}"
        )

        print(
            f"{'End Date':<35}: "
            f"{dates['max_date'].strftime('%Y-%m-%d')}"
        )

    else:

        print(f"{'Start Date':<35}: N/A")
        print(f"{'End Date':<35}: N/A")

    print(
        f"{'Invalid Dates':<35}: "
        f"{dates.get('invalid_dates', 0):,}"
    )

    # ---------------------------------------------------------
    # Data Quality
    # ---------------------------------------------------------

    print("\nDATA QUALITY")
    print("-" * 70)

    missing_customer_id = customers["missing_customer_id"]

    print(
        f"{'Missing Customer ID':<35}: "
        f"{missing_customer_id:,}"
        if missing_customer_id is not None
        else f"{'Missing Customer ID':<35}: N/A"
    )

    if missing_customer_id is not None and dataset["rows"] > 0:

        percentage = (
            missing_customer_id / dataset["rows"]
        ) * 100

        print(
            f"{'Missing Customer ID (%)':<35}: "
            f"{percentage:.2f}%"
        )

    duplicates = report["duplicates"]

    print(
        f"{'Duplicate Rows':<35}: "
        f"{duplicates['duplicate_rows']:,}"
    )

    quantity = report["quantity"]

    print(
        f"{'Negative Quantity':<35}: "
        f"{quantity['negative_quantity']:,}"
        if quantity["negative_quantity"] is not None
        else f"{'Negative Quantity':<35}: N/A"
    )

    print(
        f"{'Zero Quantity':<35}: "
        f"{quantity['zero_quantity']:,}"
        if quantity["zero_quantity"] is not None
        else f"{'Zero Quantity':<35}: N/A"
    )

    price = report["price"]

    print(
        f"{'Zero Price':<35}: "
        f"{price['zero_price']:,}"
        if price["zero_price"] is not None
        else f"{'Zero Price':<35}: N/A"
    )

    print(
        f"{'Negative Price':<35}: "
        f"{price['negative_price']:,}"
        if price["negative_price"] is not None
        else f"{'Negative Price':<35}: N/A"
    )

    # ---------------------------------------------------------
    # Column Validation
    # ---------------------------------------------------------

    columns = report["columns"]

    print("\nCOLUMN VALIDATION")
    print("-" * 70)

    print(f"{'Expected Columns':<35}: "
          f"{len(columns['expected'])}")

    print(f"{'Actual Columns':<35}: "
          f"{len(columns['actual'])}")

    if columns["missing_columns"]:

        print(f"{'Missing Columns':<35}: "
              f"{', '.join(columns['missing_columns'])}")

    else:
        print(f"{'Missing Columns':<35}: None")

    if columns["unexpected_columns"]:

        print(f"{'Unexpected Columns':<35}: "
              f"{', '.join(columns['unexpected_columns'])}")

    else:
        print(f"{'Unexpected Columns':<35}: None")

    # ---------------------------------------------------------
    # Overall Status
    # ---------------------------------------------------------

    print("\n" + "=" * 70)

    if report["validation_passed"]:
        print(f"{'VALIDATION STATUS':<35}: PASSED")

    else:
        print(f"{'VALIDATION STATUS':<35}: REVIEW REQUIRED")

    print("=" * 70)



################################################################
#--- cleaned/featured dataset validating function---------------
################################################################
def validate_loaded_dataset(df, dataset_name="Dataset"):
    """
    Validate a loaded pandas DataFrame and display a notebook-friendly report.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to validate.
    dataset_name : str
        Name displayed in the validation report.

    Returns
    -------
    dict
        Dictionary containing validation summary and detailed results.
    """

    # --------------------------------------------------
    # 1. Validate input
    # --------------------------------------------------
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"{dataset_name} must be a pandas DataFrame."
        )

    if df.empty:
        print(f"WARNING: {dataset_name} is empty.")

    # --------------------------------------------------
    # 2. Calculate dataset-level metrics
    # --------------------------------------------------
    total_rows = len(df)
    total_columns = len(df.columns)
    total_cells = total_rows * total_columns

    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    missing_percentage = (
        (missing_cells / total_cells) * 100
        if total_cells > 0 else 0
    )

    memory_usage_mb = df.memory_usage(
        deep=True
    ).sum() / (1024 ** 2)

    # --------------------------------------------------
    # 3. Column-level quality report
    # --------------------------------------------------
    column_report = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Count": df.isna().sum().values,
        "Missing (%)": (
            df.isna().mean().values * 100
        ).round(2),
        "Unique Values": df.nunique().values,
        "Duplicate Values": [
            int(df[col].duplicated().sum())
            for col in df.columns
        ]
    })

    column_report["Missing Count"] = (
        column_report["Missing Count"].astype(int)
    )

    column_report["Unique Values"] = (
        column_report["Unique Values"].astype(int)
    )

    column_report["Duplicate Values"] = (
        column_report["Duplicate Values"].astype(int)
    )

    # Show columns with missing values first
    column_report = column_report.sort_values(
        by=["Missing Count", "Column"],
        ascending=[False, True]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # 4. Create summary report
    # --------------------------------------------------
    summary_report = pd.DataFrame({
        "Metric": [
            "Dataset Name",
            "Total Rows",
            "Total Columns",
            "Total Cells",
            "Missing Cells",
            "Missing Cells (%)",
            "Duplicate Rows",
            "Memory Usage (MB)",
            "Empty Dataset"
        ],
        "Value": [
            dataset_name,
            total_rows,
            total_columns,
            total_cells,
            missing_cells,
            round(missing_percentage, 2),
            duplicate_rows,
            round(memory_usage_mb, 2),
            df.empty
        ]
    })

    # --------------------------------------------------
    # 5. Notebook-friendly display
    # --------------------------------------------------
    print("=" * 70)
    print(f"DATASET VALIDATION REPORT: {dataset_name}")
    print("=" * 70)

    display(summary_report)

    print("\nCOLUMN-LEVEL DATA QUALITY REPORT")
    display(column_report)

    # --------------------------------------------------
    # 6. Validation warnings
    # --------------------------------------------------
    warnings = []

    if df.empty:
        warnings.append("Dataset contains no rows.")

    if missing_cells > 0:
        warnings.append(
            f"Dataset contains {missing_cells:,} missing cells."
        )

    if duplicate_rows > 0:
        warnings.append(
            f"Dataset contains {duplicate_rows:,} duplicate rows."
        )

    if warnings:
        print("\nVALIDATION WARNINGS")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("\nBasic validation passed: no missing cells or "
              "duplicate rows detected.")

    # --------------------------------------------------
    # 7. Return results without modifying the input
    # --------------------------------------------------
    validation_results = {
        "dataset_name": dataset_name,
        "summary": summary_report,
        "column_report": column_report,
        "warnings": warnings,
        "is_empty": df.empty,
        "has_missing_values": missing_cells > 0,
        "has_duplicate_rows": duplicate_rows > 0
    }

    return validation_results
