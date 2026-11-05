############################################## Data Cleaning Functions ################################################
import pandas as pd
 
    
# 1. -------------- duplicate removal ---------------
def clean_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate transaction rows.

    Legitimate repeated purchases are preserved because only rows that
    are completely identical across all columns are removed.

    Args:
        df: Transaction-level DataFrame.

    Returns:
        DataFrame with exact duplicate rows removed.
    """
    cleaned_df = df.copy()

    duplicate_count = int(cleaned_df.duplicated().sum())

    if duplicate_count > 0:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)

    return cleaned_df

# 2. ----------- Handle Quantity And Price ----------
def handle_quantity_and_price(
    df: pd.DataFrame,
    extreme_quantity_threshold: int = 10000,
    extreme_value_quantile: float = 0.999,
) -> pd.DataFrame:
    """
    Detect and treat Quantity and Price anomalies.

    Quantity rules
    --------------
    - Negative quantities are retained as returns/cancellations.
    - Positive IQR outliers are retained as potential bulk purchases.
    - Very large quantities are investigated using domain rules.
    - Repeated bulk purchasing by the same invoice or customer
      is treated as legitimate bulk behavior.
    - Isolated extreme transactions with exceptionally high
      transaction value are excluded from normal sales analysis.

    Price rules
    -----------
    - Positive prices are retained, including high-price IQR outliers.
    - Zero prices are excluded from normal sales analysis.
    - Negative prices are treated as administrative adjustments
      and excluded from normal sales analysis.

    Important
    ---------
    Original Quantity and Price values are never overwritten.
    Cleaning decisions are stored in separate treatment columns.
    """

    # =========================================================
    # 1. Validate required columns
    # =========================================================

    required_columns = {
        "Invoice",
        "StockCode",
        "Quantity",
        "Price",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    working_df = df.copy()

    # =========================================================
    # 2. Convert Quantity and Price to numeric
    # =========================================================

    working_df["Quantity"] = pd.to_numeric(
        working_df["Quantity"],
        errors="coerce",
    )

    working_df["Price"] = pd.to_numeric(
        working_df["Price"],
        errors="coerce",
    )

    # =========================================================
    # 3. Quantity IQR calculation
    # =========================================================

    quantity_q1 = working_df["Quantity"].quantile(0.25)
    quantity_q3 = working_df["Quantity"].quantile(0.75)

    quantity_iqr = quantity_q3 - quantity_q1

    quantity_lower_bound = (
        quantity_q1 - 1.5 * quantity_iqr
    )

    quantity_upper_bound = (
        quantity_q3 + 1.5 * quantity_iqr
    )

    # =========================================================
    # 4. Price IQR calculation
    #
    # Only positive prices are used to calculate the normal
    # product-price distribution.
    #
    # Zero and negative prices are handled separately because
    # they represent adjustments/manual corrections.
    # =========================================================

    positive_prices = working_df.loc[
        working_df["Price"] > 0,
        "Price",
    ]

    price_q1 = positive_prices.quantile(0.25)
    price_q3 = positive_prices.quantile(0.75)

    price_iqr = price_q3 - price_q1

    price_lower_bound = (
        price_q1 - 1.5 * price_iqr
    )

    price_upper_bound = (
        price_q3 + 1.5 * price_iqr
    )

    # =========================================================
    # 5. IQR outlier flags
    # =========================================================

    working_df["Quantity IQR Outlier"] = (
        (working_df["Quantity"] < quantity_lower_bound)
        | (working_df["Quantity"] > quantity_upper_bound)
    )

    working_df["Price IQR Outlier"] = (
        (working_df["Price"] > 0)
        & (
            (working_df["Price"] < price_lower_bound)
            | (working_df["Price"] > price_upper_bound)
        )
    )

    # =========================================================
    # 6. Identify transaction type
    #
    # In Online Retail II, invoices beginning with "C"
    # represent cancellations.
    # Other negative-quantity records are classified as returns.
    # =========================================================

    invoice_text = (
        working_df["Invoice"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    working_df["Transaction Type"] = "Purchase"

    cancellation_mask = invoice_text.str.startswith(
        "C",
        na=False,
    )

    return_mask = (
        (working_df["Quantity"] < 0)
        & ~cancellation_mask
    )

    working_df.loc[
        cancellation_mask,
        "Transaction Type",
    ] = "Cancellation"

    working_df.loc[
        return_mask,
        "Transaction Type",
    ] = "Return"

    # =========================================================
    # 7. Initial quantity treatment
    # =========================================================

    working_df["Quantity Treatment"] = "Normal"

    # Negative quantity records are retained because return
    # behavior is required later as a customer feature.

    working_df.loc[
        working_df["Quantity"] < 0,
        "Quantity Treatment",
    ] = "Return/Cancellation"

    # Positive IQR outliers are potential bulk purchases.
    # They are NOT removed automatically.

    potential_bulk_mask = (
        (working_df["Quantity"] > 0)
        & working_df["Quantity IQR Outlier"]
    )

    working_df.loc[
        potential_bulk_mask,
        "Quantity Treatment",
    ] = "Potential Bulk Purchase"

    # =========================================================
    # 8. Calculate transaction value
    # =========================================================

    working_df["Transaction Value"] = (
        working_df["Quantity"]
        * working_df["Price"]
    )

    # =========================================================
    # 9. Identify extreme quantities
    # =========================================================

    extreme_quantity_mask = (
        working_df["Quantity"]
        > extreme_quantity_threshold
    )

    # =========================================================
    # 10. Calculate exceptional transaction-value threshold
    #
    # Use positive sales only.
    # =========================================================

    positive_sales_mask = (
        (working_df["Quantity"] > 0)
        & (working_df["Price"] > 0)
    )

    positive_transaction_values = (
        working_df.loc[
            positive_sales_mask,
            "Transaction Value",
        ]
    )

    if len(positive_transaction_values) > 0:

        extreme_value_threshold = (
            positive_transaction_values.quantile(
                extreme_value_quantile
            )
        )

    else:

        extreme_value_threshold = float("inf")

    # =========================================================
    # 11. Identify coherent bulk invoices
    #
    # Multiple extreme-quantity lines on the same invoice
    # are evidence of a potential genuine bulk order.
    # =========================================================

    extreme_rows_per_invoice = (
        extreme_quantity_mask
        .groupby(working_df["Invoice"])
        .sum()
    )

    bulk_invoice_numbers = (
        extreme_rows_per_invoice[
            extreme_rows_per_invoice >= 2
        ]
        .index
    )

    coherent_bulk_invoice_mask = (
        extreme_quantity_mask
        & working_df["Invoice"].isin(
            bulk_invoice_numbers
        )
    )

    # =========================================================
    # 12. Identify repeated bulk behavior by customer
    #
    # If the same customer has multiple extreme-quantity
    # transactions, this provides additional evidence that the
    # behavior may be legitimate bulk purchasing.
    # =========================================================

    if "Customer ID" in working_df.columns:

        extreme_rows_per_customer = (
            extreme_quantity_mask
            .groupby(working_df["Customer ID"])
            .sum()
        )

        bulk_customer_ids = (
            extreme_rows_per_customer[
                extreme_rows_per_customer >= 2
            ]
            .index
        )

        repeated_bulk_customer_mask = (
            extreme_quantity_mask
            & working_df["Customer ID"].isin(
                bulk_customer_ids
            )
        )

    else:

        repeated_bulk_customer_mask = (
            pd.Series(
                False,
                index=working_df.index,
            )
        )

    # =========================================================
    # 13. Final legitimate bulk identification
    # =========================================================

    legitimate_bulk_mask = (
        coherent_bulk_invoice_mask
        | repeated_bulk_customer_mask
    )

    working_df.loc[
        legitimate_bulk_mask,
        "Quantity Treatment",
    ] = "Legitimate Bulk Purchase"

    # =========================================================
    # 14. Identify isolated extreme transactions
    #
    # These must satisfy ALL conditions:
    #
    # - Quantity > extreme threshold
    # - Positive price
    # - Exceptionally large transaction value
    # - Not part of a repeated bulk invoice
    # - Not part of repeated customer bulk behavior
    # =========================================================

    isolated_extreme_mask = (
        extreme_quantity_mask
        & (working_df["Price"] > 0)
        & (
            working_df["Transaction Value"]
            > extreme_value_threshold
        )
        & ~legitimate_bulk_mask
    )

    working_df.loc[
        isolated_extreme_mask,
        "Quantity Treatment",
    ] = "Extreme Quantity - Exclude"

    # =========================================================
    # 15. Price treatment
    # =========================================================

    working_df["Price Treatment"] = "Normal"

    high_price_mask = (
        (working_df["Price"] > 0)
        & working_df["Price IQR Outlier"]
    )

    zero_price_mask = (
        working_df["Price"] == 0
    )

    negative_price_mask = (
        working_df["Price"] < 0
    )

    working_df.loc[
        high_price_mask,
        "Price Treatment",
    ] = "High Price Outlier - Retained"

    working_df.loc[
        zero_price_mask,
        "Price Treatment",
    ] = "Zero Price - Excluded"

    working_df.loc[
        negative_price_mask,
        "Price Treatment",
    ] = "Negative Price - Administrative Adjustment"

    # =========================================================
    # 16. Final sales-analysis inclusion flag
    # =========================================================

    working_df["Include In Sales Analysis"] = True

    # Zero price → exclude

    working_df.loc[
        zero_price_mask,
        "Include In Sales Analysis",
    ] = False

    # Negative price → exclude

    working_df.loc[
        negative_price_mask,
        "Include In Sales Analysis",
    ] = False

    # Confirmed isolated extreme quantity → exclude

    working_df.loc[
        isolated_extreme_mask,
        "Include In Sales Analysis",
    ] = False

    # =========================================================
    # 17. Analytical transaction value
    #
    # Original Transaction Value is preserved.
    # Analytical Transaction Value is zero for records that
    # should not contribute to normal sales calculations.
    # =========================================================

    working_df["Analytical Transaction Value"] = (
        working_df["Transaction Value"]
        .where(
            working_df["Include In Sales Analysis"],
            0,
        )
    )

    # =========================================================
    # 18. Store IQR statistics for auditability
    # =========================================================

    working_df.attrs["quantity_iqr"] = {
        "q1": float(quantity_q1),
        "q3": float(quantity_q3),
        "iqr": float(quantity_iqr),
        "lower_bound": float(quantity_lower_bound),
        "upper_bound": float(quantity_upper_bound),
    }

    working_df.attrs["price_iqr"] = {
        "q1": float(price_q1),
        "q3": float(price_q3),
        "iqr": float(price_iqr),
        "lower_bound": float(price_lower_bound),
        "upper_bound": float(price_upper_bound),
    }

    working_df.attrs["domain_rules"] = {
        "extreme_quantity_threshold": (
            extreme_quantity_threshold
        ),
        "extreme_value_quantile": (
            extreme_value_quantile
        ),
        "extreme_value_threshold": float(
            extreme_value_threshold
        ),
    }

    return working_df


# 3. ----------- Standardize Descriptions ------------
def standardize_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize product descriptions using a lookup table keyed by StockCode.

    Rules
    -----
    1. Preserve the original Description for audit purposes.
    2. Standardize whitespace and text case.
    3. Identify obvious data-quality notes/placeholders that should
       not be used as canonical product descriptions.
    4. Build a canonical description lookup using StockCode.
    5. Select the most frequent valid product description for each StockCode.
    6. Replace inconsistent descriptions with the canonical description.
    7. Create Description Present Flag:
         1 = description was present
         0 = description was missing
    8. Replace remaining missing descriptions with "NO DESCRIPTION".
    9. StockCode itself is never modified.
    """

    required_columns = {
        "StockCode",
        "Description",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    working_df = df.copy()

    # ---------------------------------------------------------
    # 1. Preserve original description
    # ---------------------------------------------------------

    working_df["Original Description"] = (
        working_df["Description"].astype("string")
    )

    # ---------------------------------------------------------
    # 2. Create normalized StockCode key for lookup
    # ---------------------------------------------------------

    stockcode_key = (
        working_df["StockCode"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    # ---------------------------------------------------------
    # 3. Standardize description formatting
    # ---------------------------------------------------------

    standardized_description = (
        working_df["Description"]
        .astype("string")
        .str.strip()
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .str.upper()
    )

    working_df["Description"] = standardized_description

    # ---------------------------------------------------------
    # 4. Identify obvious data-quality notes/placeholders
    # ---------------------------------------------------------

    data_quality_note_mask = (
        standardized_description.str.contains(
            r"WRONGLY\s+(MARKED|CODED)"
            r"|MARKED\s+AS"
            r"|FOUND"
            r"|MISSING",
            regex=True,
            na=False,
        )
    )

    working_df["Description Data Quality Note"] = (
        data_quality_note_mask
    )

    # ---------------------------------------------------------
    # 5. Prepare lookup source
    # ---------------------------------------------------------

    lookup_source = pd.DataFrame(
        {
            "StockCode_Key": stockcode_key,
            "Description": standardized_description,
            "Is_Data_Quality_Note": data_quality_note_mask,
        }
    )

    lookup_source = lookup_source.dropna(
        subset=[
            "StockCode_Key",
            "Description",
        ]
    )

    lookup_source = lookup_source[
        lookup_source["StockCode_Key"] != ""
    ]

    lookup_source = lookup_source[
        lookup_source["Description"] != ""
    ]

    # ---------------------------------------------------------
    # 6. Remove obvious data-quality notes from canonical
    #    description candidates
    # ---------------------------------------------------------

    valid_description_source = lookup_source[
        ~lookup_source["Is_Data_Quality_Note"]
    ].copy()

    # ---------------------------------------------------------
    # 7. Select the most frequent valid description per StockCode
    # ---------------------------------------------------------

    description_lookup = (
        valid_description_source
        .groupby("StockCode_Key")["Description"]
        .agg(
            lambda values: values.value_counts().index[0]
        )
        .to_dict()
    )

    # ---------------------------------------------------------
    # 8. Apply canonical description lookup
    # ---------------------------------------------------------

    canonical_description = stockcode_key.map(
        description_lookup
    )

    working_df["Description"] = (
        canonical_description.fillna(
            working_df["Description"]
        )
    )

    # ---------------------------------------------------------
    # 9. Create Description Present Flag
    #
    # 1 = description was present
    # 0 = description was missing
    #
    # Use Original Description here so that the flag represents
    # the original source data, not the canonical lookup result.
    # ---------------------------------------------------------

    working_df["Description Present Flag"] = (
        working_df["Original Description"]
        .notna()
        .astype("int8")
    )

    # ---------------------------------------------------------
    # 10. Track whether description was standardized
    # ---------------------------------------------------------

    original_normalized = (
        working_df["Original Description"]
        .astype("string")
        .str.strip()
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .str.upper()
    )

    working_df["Description Standardized"] = (
        original_normalized.fillna("<MISSING>")
        != working_df["Description"].fillna("<MISSING>")
    )

    # ---------------------------------------------------------
    # 11. Replace remaining missing descriptions
    # ---------------------------------------------------------

    working_df["Description"] = (
        working_df["Description"]
        .fillna("NO DESCRIPTION")
        .astype("string")
    )

    # ---------------------------------------------------------
    # 12. Ensure final description columns have string type
    # ---------------------------------------------------------

    working_df["Original Description"] = (
        working_df["Original Description"].astype("string")
    )

    working_df["Description Data Quality Note"] = (
        working_df["Description Data Quality Note"].astype("int8")
    )

    return working_df


# 4.--------- product level dataset and customer level dataset -------------
def handle_customer_id(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Handle missing Customer ID values while preserving all transactions.

    Rules
    -----
    - Keep all transaction rows.
    - Preserve available Customer ID values.
    - Replace missing Customer ID values with 0.
    - Create Customer ID Missing Flag:
        0 = Customer ID was available
        1 = Customer ID was missing

    The flag preserves the original missingness information so that
    downstream analyses can filter or exclude unidentified customers
    when required.
    """

    if "Customer ID" not in df.columns:
        raise ValueError(
            "Missing required column: Customer ID"
        )

    working_df = df.copy()

    # Preserve whether Customer ID was originally missing.
    working_df["Customer ID Missing Flag"] = (
        working_df["Customer ID"]
        .isna()
        .astype("int8")
    )

    # Replace missing Customer IDs with sentinel value 0.
    working_df["Customer ID"] = (
        working_df["Customer ID"]
        .fillna(0).astype("int64")
    )

    return working_df


# 5. ---- Master cleaning function that calls all the above cleaning functions in sequence ----
def clean_transaction_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute complete transaction-data cleaning pipeline.
    """

    cleaned_df = df.copy()

    # Standardize columns that already exist in the raw dataset
    cleaned_df["Invoice"] = cleaned_df["Invoice"].astype("string")
    cleaned_df["StockCode"] = cleaned_df["StockCode"].astype("string")
    cleaned_df["Description"] = cleaned_df["Description"].astype("string")
    cleaned_df["Country"] = cleaned_df["Country"].astype("string")

    cleaned_df = clean_duplicates(cleaned_df)
    cleaned_df = handle_quantity_and_price(cleaned_df)
    cleaned_df = standardize_descriptions(cleaned_df)
    cleaned_df = handle_customer_id(cleaned_df)

    return cleaned_df


# 6. ----------------- Private methods --------------------------------
def validate_cleaned_data(
    df: pd.DataFrame,
) -> dict:
    """
    Validate the cleaned transaction dataframe.

    Validation checks
    -----------------
    1. Required columns are present.
    2. No exact duplicate rows remain.
    3. Customer ID handling is internally consistent.
    4. Customer ID Missing Flag contains only 0 or 1.
    5. InvoiceDate contains no missing values.
    6. Cleaning treatment columns contain no missing values.
    7. Sales-analysis columns contain no missing values.

    Notes
    -----
    Negative Quantity, zero Price, negative Price, and missing
    Description are reported as data-quality statistics but are
    not automatically treated as validation failures because
    these conditions are explicitly handled by the cleaning
    pipeline.

    Returns
    -------
    dict
        Validation results and dataset statistics.
    """

    # =========================================================
    # 1. Required columns
    # =========================================================

    required_columns = {
        "Invoice",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer ID",
        "Country",
        "Transaction Type",
        "Transaction Value",
        "Quantity Treatment",
        "Price Treatment",
        "Include In Sales Analysis",
        "Analytical Transaction Value",
        "Customer ID Missing Flag",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    validation_results = {}

    # =========================================================
    # 2. Basic dataset validation
    # =========================================================

    validation_results["row_count"] = int(
        len(df)
    )

    validation_results["duplicate_rows"] = int(
        df.duplicated().sum()
    )

    # =========================================================
    # 3. Customer ID validation
    # =========================================================

    # After cleaning, missing Customer IDs are represented
    # by the sentinel value 0.
    validation_results["missing_customer_id"] = int(
        df["Customer ID"].isna().sum()
    )

    # Original Customer ID was missing.
    validation_results["customer_id_flag_1"] = int(
        (df["Customer ID Missing Flag"] == 1).sum()
    )

    # Original Customer ID was available.
    validation_results["customer_id_flag_0"] = int(
        (df["Customer ID Missing Flag"] == 0).sum()
    )

    # Flag = 1 must correspond to Customer ID = 0.
    invalid_missing_customer_id_rows = (
        (df["Customer ID Missing Flag"] == 1)
        & (df["Customer ID"] != 0)
    ).sum()

    validation_results[
        "invalid_missing_customer_id_rows"
    ] = int(
        invalid_missing_customer_id_rows
    )

    # Flag = 0 must correspond to Customer ID != 0.
    invalid_available_customer_id_rows = (
        (df["Customer ID Missing Flag"] == 0)
        & (df["Customer ID"] == 0)
    ).sum()

    validation_results[
        "invalid_available_customer_id_rows"
    ] = int(
        invalid_available_customer_id_rows
    )

    # =========================================================
    # 4. Customer ID flag validation
    # =========================================================

    invalid_customer_id_flag_values = (
        ~df["Customer ID Missing Flag"].isin([0, 1])
    ).sum()

    validation_results[
        "invalid_customer_id_flag_values"
    ] = int(
        invalid_customer_id_flag_values
    )

    # =========================================================
    # 5. Description validation
    # =========================================================

    validation_results["missing_description"] = int(
        df["Description"].isna().sum()
    )

    # =========================================================
    # 6. Quantity validation
    # =========================================================

    # Negative quantities represent returns/cancellations
    # and are therefore reported, not treated as failures.
    validation_results["negative_quantity"] = int(
        (df["Quantity"] < 0).sum()
    )

    validation_results["zero_quantity"] = int(
        (df["Quantity"] == 0).sum()
    )

    # =========================================================
    # 7. Price validation
    # =========================================================

    # Zero and negative prices have already been classified
    # by handle_quantity_and_price().
    validation_results["zero_price"] = int(
        (df["Price"] == 0).sum()
    )

    validation_results["negative_price"] = int(
        (df["Price"] < 0).sum()
    )

    # =========================================================
    # 8. InvoiceDate validation
    # =========================================================

    validation_results["missing_invoice_date"] = int(
        df["InvoiceDate"].isna().sum()
    )

    # =========================================================
    # 9. Transaction treatment validation
    # =========================================================

    validation_results[
        "missing_transaction_type"
    ] = int(
        df["Transaction Type"].isna().sum()
    )

    validation_results[
        "missing_quantity_treatment"
    ] = int(
        df["Quantity Treatment"].isna().sum()
    )

    validation_results[
        "missing_price_treatment"
    ] = int(
        df["Price Treatment"].isna().sum()
    )

    # =========================================================
    # 10. Sales-analysis validation
    # =========================================================

    validation_results[
        "missing_sales_analysis_flag"
    ] = int(
        df["Include In Sales Analysis"].isna().sum()
    )

    validation_results[
        "missing_analytical_transaction_value"
    ] = int(
        df["Analytical Transaction Value"].isna().sum()
    )

    # =========================================================
    # 11. Overall validation status
    # =========================================================

    validation_results["passed"] = all(
        [
            validation_results["duplicate_rows"] == 0,

            validation_results["missing_customer_id"] == 0,

            validation_results[
                "invalid_missing_customer_id_rows"
            ] == 0,

            validation_results[
                "invalid_available_customer_id_rows"
            ] == 0,

            validation_results[
                "invalid_customer_id_flag_values"
            ] == 0,

            validation_results[
                "missing_invoice_date"
            ] == 0,

            validation_results[
                "missing_transaction_type"
            ] == 0,

            validation_results[
                "missing_quantity_treatment"
            ] == 0,

            validation_results[
                "missing_price_treatment"
            ] == 0,

            validation_results[
                "missing_sales_analysis_flag"
            ] == 0,

            validation_results[
                "missing_analytical_transaction_value"
            ] == 0,
        ]
    )

    return validation_results


# 7.-------------------  Validation ---------------------------------- 
def summarize_quantity_price_treatment(
    df: pd.DataFrame,
) -> dict:
    """
    Summarize Quantity and Price treatment results.
    """

    summary = {
        "quantity_treatment": (
            df["Quantity Treatment"]
            .value_counts()
        ),
        "price_treatment": (
            df["Price Treatment"]
            .value_counts()
        ),
        "transaction_type": (
            df["Transaction Type"]
            .value_counts()
        ),
        "quantity_iqr_outliers": int(
            df["Quantity IQR Outlier"].sum()
        ),
        "price_iqr_outliers": int(
            df["Price IQR Outlier"].sum()
        ),
        "non_positive_prices": int(
            (df["Price"] <= 0).sum()
        ),
        "excluded_from_sales_analysis": int(
            (~df["Include In Sales Analysis"]).sum()
        ),
    }

    return summary