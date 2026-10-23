import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from src.data.ingest import get_project_root


################################################################
#------ 1. Classification model dataset preparation
########################################################################

# 1. Creating churn dataset
def create_churn_dataset(
    customer_features: pd.DataFrame,
    cleaned_transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the complete customer-level churn dataset.

    Churn Definition
    ----------------
    A customer is classified as:

        Churn = 1
        if the customer makes no qualifying purchase
        during the 90-day period following the point-in-time cutoff.

        Churn = 0
        if the customer makes at least one qualifying purchase
        during the following 90-day period.

    The point-in-time cutoff is automatically calculated as:

        maximum InvoiceDate - 90 days

    Customer features are assumed to have already been calculated
    using information available on or before this cutoff.

    Parameters
    ----------
    customer_features : pd.DataFrame
        Frozen customer-level feature table.

    cleaned_transactions : pd.DataFrame
        Cleaned transaction-level dataset used to determine
        future customer purchase behavior.

    Returns
    -------
    pd.DataFrame
        Complete customer-level churn dataset containing all
        customer features plus the Churn target.
    """

    # ============================================================
    # 1. Validate customer feature dataframe
    # ============================================================

    if customer_features.empty:
        raise ValueError(
            "customer_features is empty."
        )

    if "Customer ID" not in customer_features.columns:
        raise ValueError(
            "customer_features must contain 'Customer ID'."
        )

    if customer_features["Customer ID"].isna().any():
        raise ValueError(
            "customer_features contains missing Customer IDs."
        )

    if customer_features["Customer ID"].duplicated().any():
        raise ValueError(
            "customer_features contains duplicate Customer IDs."
        )

    # ============================================================
    # 2. Validate cleaned transaction dataframe
    # ============================================================

    required_transaction_columns = {
        "Customer ID",
        "InvoiceDate",
        "Transaction Type",
        "Include In Sales Analysis",
        "Quantity",
    }

    missing_columns = (
        required_transaction_columns
        - set(cleaned_transactions.columns)
    )

    if missing_columns:
        raise ValueError(
            "cleaned_transactions is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if cleaned_transactions.empty:
        raise ValueError(
            "cleaned_transactions is empty."
        )

    # ============================================================
    # 3. Create working transaction dataframe
    # ============================================================

    working_transactions = cleaned_transactions.copy()

    working_transactions["InvoiceDate"] = pd.to_datetime(
        working_transactions["InvoiceDate"],
        errors="coerce",
    )

    if working_transactions["InvoiceDate"].isna().any():
        raise ValueError(
            "cleaned_transactions contains invalid InvoiceDate values."
        )

    # ============================================================
    # 4. Determine maximum transaction date
    # ============================================================

    max_invoice_date = (
        working_transactions["InvoiceDate"].max()
    )

    if pd.isna(max_invoice_date):
        raise ValueError(
            "Unable to determine maximum InvoiceDate."
        )

    # ============================================================
    # 5. Automatically determine point-in-time cutoff
    # ============================================================

    cutoff_date = (
        max_invoice_date
        - pd.Timedelta(days=90)
    )

    # ============================================================
    # 6. Define future 90-day prediction window
    # ============================================================

    future_transactions = working_transactions[
        (
            working_transactions["InvoiceDate"]
            > cutoff_date
        )
        &
        (
            working_transactions["InvoiceDate"]
            <= max_invoice_date
        )
    ].copy()

    # ============================================================
    # 7. Identify qualifying future purchases
    # ============================================================

    future_purchases = future_transactions[
        (
            future_transactions["Customer ID"].notna()
        )
        &
        (
            future_transactions["Customer ID"] != 0
        )
        &
        (
            future_transactions["Transaction Type"]
            == "Purchase"
        )
        &
        (
            future_transactions["Include In Sales Analysis"]
            == True
        )
        &
        (
            future_transactions["Quantity"] > 0
        )
    ].copy()

    # ============================================================
    # 8. Identify customers with at least one future purchase
    # ============================================================

    future_purchase_customer_ids = set(
        future_purchases["Customer ID"]
        .astype("int64")
        .unique()
    )

    # ============================================================
    # 9. Prepare customer feature dataframe
    # ============================================================

    churn_dataset = customer_features.copy()

    churn_dataset["Customer ID"] = (
        churn_dataset["Customer ID"]
        .astype("int64")
    )

    # ============================================================
    # 10. Create churn target
    # ============================================================

    churn_dataset["Churn"] = (
        ~churn_dataset["Customer ID"].isin(
            future_purchase_customer_ids
        )
    ).astype("int8")

    # ============================================================
    # 11. Validate churn target
    # ============================================================

    if churn_dataset["Churn"].isna().any():
        raise ValueError(
            "Churn contains missing values."
        )

    if not set(
        churn_dataset["Churn"].unique()
    ).issubset({0, 1}):

        raise ValueError(
            "Churn must contain only 0 and 1."
        )

    # ============================================================
    # 12. Validate customer population
    # ============================================================

    feature_customer_ids = set(
        customer_features["Customer ID"]
        .astype("int64")
    )

    output_customer_ids = set(
        churn_dataset["Customer ID"]
    )

    if feature_customer_ids != output_customer_ids:
        raise ValueError(
            "Customer population changed while creating "
            "the churn dataset."
        )

    # ============================================================
    # 13. Store metadata
    # ============================================================

    churn_dataset.attrs["max_invoice_date"] = (
        max_invoice_date
    )

    churn_dataset.attrs["cutoff_date"] = (
        cutoff_date
    )

    churn_dataset.attrs["prediction_window_days"] = 90

    churn_dataset.attrs["customer_count"] = (
        len(churn_dataset)
    )

    churn_dataset.attrs["future_purchase_customer_count"] = (
        len(future_purchase_customer_ids)
    )

    churn_dataset.attrs["churned_customer_count"] = int(
        (churn_dataset["Churn"] == 1).sum()
    )

    churn_dataset.attrs["non_churned_customer_count"] = int(
        (churn_dataset["Churn"] == 0).sum()
    )

    # ============================================================
    # 14. Return complete customer-level churn dataset
    # ============================================================

    return churn_dataset

# 2. Validating churn dataset
def validate_churn_dataset(
    churn_dataset: pd.DataFrame,
) -> dict:
    """
    Validate the customer-level churn dataset.

    The validation checks:
    - Dataset is not empty
    - Customer ID exists
    - Customer IDs are unique
    - Customer IDs are not missing
    - Churn target exists
    - Churn contains only 0 and 1
    - Churn target is not missing
    - Predictor features are not completely missing
    - Dataset contains both churn classes
    - Customer count is greater than zero

    Parameters
    ----------
    churn_dataset : pd.DataFrame
        Complete customer-level churn dataset containing
        customer features and the Churn target.

    Returns
    -------
    dict
        Validation results containing:
        - validation
        - summary
        - failed_checks
    """

    # ============================================================
    # 1. Initialize validation
    # ============================================================

    validation = {}
    failed_checks = []

    # ============================================================
    # 2. Dataset not empty
    # ============================================================

    validation["dataset_not_empty"] = (
        not churn_dataset.empty
    )

    # ============================================================
    # 3. Customer ID column
    # ============================================================

    validation["customer_id_column_present"] = (
        "Customer ID" in churn_dataset.columns
    )

    # ============================================================
    # 4. Churn column
    # ============================================================

    validation["churn_column_present"] = (
        "Churn" in churn_dataset.columns
    )

    # Stop structural validation if required columns are absent
    if (
        not validation["dataset_not_empty"]
        or not validation["customer_id_column_present"]
        or not validation["churn_column_present"]
    ):

        for check_name, result in validation.items():

            if not bool(result):
                failed_checks.append(check_name)

        validation["passed"] = False

        return {
            "validation": validation,
            "summary": {
                "customer_count": 0,
                "feature_count": 0,
            },
            "failed_checks": failed_checks,
        }

    # ============================================================
    # 5. Customer ID checks
    # ============================================================

    validation["no_missing_customer_id"] = (
        not churn_dataset["Customer ID"].isna().any()
    )

    validation["customer_id_is_unique"] = (
        not churn_dataset["Customer ID"].duplicated().any()
    )

    # ============================================================
    # 6. Churn missing-value check
    # ============================================================

    validation["no_missing_churn"] = (
        not churn_dataset["Churn"].isna().any()
    )

    # ============================================================
    # 7. Churn value check
    # ============================================================

    actual_churn_values = set(
        churn_dataset["Churn"]
        .dropna()
        .unique()
    )

    validation["churn_values_are_binary"] = (
        actual_churn_values.issubset({0, 1})
    )

    # ============================================================
    # 8. Both classes must be present
    # ============================================================

    churn_class_count = (
        churn_dataset["Churn"]
        .value_counts()
    )

    validation["both_churn_classes_present"] = (
        set(churn_class_count.index) == {0, 1}
    )

    # ============================================================
    # 9. Churn data type
    # ============================================================

    validation["churn_is_numeric"] = (
        pd.api.types.is_numeric_dtype(
            churn_dataset["Churn"]
        )
    )

    # ============================================================
    # 10. Predictor columns
    # ============================================================

    predictor_columns = [
        column
        for column in churn_dataset.columns
        if column not in ["Customer ID", "Churn"]
    ]

    validation["predictor_features_present"] = (
        len(predictor_columns) > 0
    )

    # ============================================================
    # 11. Check completely missing predictor columns
    # ============================================================

    completely_missing_features = [
        column
        for column in predictor_columns
        if churn_dataset[column].isna().all()
    ]

    validation["no_completely_missing_features"] = (
        len(completely_missing_features) == 0
    )

    # ============================================================
    # 12. Customer count
    # ============================================================

    customer_count = len(churn_dataset)

    validation["customer_count_greater_than_zero"] = (
        customer_count > 0
    )

    # ============================================================
    # 13. Build failed-check list
    # ============================================================

    for check_name, result in validation.items():

        if not bool(result):
            failed_checks.append(check_name)

    # ============================================================
    # 14. Summary
    # ============================================================

    summary = {
        "customer_count": customer_count,
        "feature_count": len(predictor_columns),
        "churned_customers": int(
            (churn_dataset["Churn"] == 1).sum()
        ),
        "non_churned_customers": int(
            (churn_dataset["Churn"] == 0).sum()
        ),
        "churn_rate": float(
            churn_dataset["Churn"].mean()
        ),
        "minimum_churn_value": (
            churn_dataset["Churn"].min()
        ),
        "maximum_churn_value": (
            churn_dataset["Churn"].max()
        ),
        "completely_missing_features": (
            completely_missing_features
        ),
    }

    # ============================================================
    # 15. Overall validation
    # ============================================================

    validation["passed"] = (
        len(failed_checks) == 0
    )

    # ============================================================
    # 16. Return validation report
    # ============================================================

    return {
        "validation": validation,
        "summary": summary,
        "failed_checks": failed_checks,
    }

# 3. analyze churn dataset
def analyze_churn_dataset(
    churn_dataset: pd.DataFrame,
) -> dict:
    """
    Analyze the statistical and structural quality of the
    customer-level churn dataset.

    Parameters
    ----------
    churn_dataset : pd.DataFrame
        Customer-level churn dataset containing customer
        features and the Churn target.

    Returns
    -------
    dict
        Statistical and data-quality summary.
    """

    if not isinstance(churn_dataset, pd.DataFrame):
        raise TypeError(
            "churn_dataset must be a pandas DataFrame."
        )

    if churn_dataset.empty:
        raise ValueError(
            "churn_dataset is empty."
        )

    # ------------------------------------------------------------
    # Basic structure
    # ------------------------------------------------------------

    shape = churn_dataset.shape

    row_count = shape[0]
    column_count = shape[1]

    # ------------------------------------------------------------
    # Duplicate analysis
    # ------------------------------------------------------------

    duplicate_rows = int(
        churn_dataset.duplicated().sum()
    )

    duplicate_customer_ids = 0

    if "Customer ID" in churn_dataset.columns:
        duplicate_customer_ids = int(
            churn_dataset["Customer ID"]
            .duplicated()
            .sum()
        )

    # ------------------------------------------------------------
    # Missing-value analysis
    # ------------------------------------------------------------

    missing_values = (
        churn_dataset.isna()
        .sum()
        .sort_values(ascending=False)
    )

    missing_columns = (
        missing_values[
            missing_values > 0
        ]
        .to_dict()
    )

    total_missing_values = int(
        missing_values.sum()
    )

    # ------------------------------------------------------------
    # Data types
    # ------------------------------------------------------------

    data_types = {
        column: str(dtype)
        for column, dtype
        in churn_dataset.dtypes.items()
    }

    # ------------------------------------------------------------
    # Churn statistics
    # ------------------------------------------------------------

    churn_statistics = {}

    if "Churn" in churn_dataset.columns:

        churn_counts = (
            churn_dataset["Churn"]
            .value_counts()
            .sort_index()
        )

        churn_percentages = (
            churn_dataset["Churn"]
            .value_counts(
                normalize=True
            )
            .sort_index()
            .mul(100)
        )

        churn_statistics = {
            "churn_counts": churn_counts.to_dict(),
            "churn_percentages": (
                churn_percentages.to_dict()
            ),
            "churn_rate": float(
                churn_dataset["Churn"].mean()
            ),
        }

    # ------------------------------------------------------------
    # Numerical descriptive statistics
    # ------------------------------------------------------------

    numerical_statistics = (
        churn_dataset
        .select_dtypes(include="number")
        .describe()
        .round(4)
    )

    # ------------------------------------------------------------
    # Basic quality status
    # ------------------------------------------------------------

    quality_checks = {
        "no_duplicate_rows": (
            duplicate_rows == 0
        ),
        "no_duplicate_customer_ids": (
            duplicate_customer_ids == 0
        ),
        "no_missing_values": (
            total_missing_values == 0
        ),
        "churn_column_present": (
            "Churn" in churn_dataset.columns
        ),
    }

    quality_status = all(
        quality_checks.values()
    )

    # ------------------------------------------------------------
    # Result
    # ------------------------------------------------------------

    return {
        "shape": shape,
        "row_count": row_count,
        "column_count": column_count,
        "duplicate_rows": duplicate_rows,
        "duplicate_customer_ids": (
            duplicate_customer_ids
        ),
        "total_missing_values": (
            total_missing_values
        ),
        "missing_columns": missing_columns,
        "data_types": data_types,
        "churn_statistics": churn_statistics,
        "numerical_statistics": numerical_statistics,
        "quality_checks": quality_checks,
        "quality_status": quality_status,
    }


# 4. preprcessing datasets for churn
def preprocess_churn_features(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    config: dict,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    StandardScaler,
]:
    """
    Preprocess churn-model features using StandardScaler.

    The scaler is fitted only on the training data and then
    used to transform the validation and test datasets.

    The fitted scaler is saved using the path configured
    in config.yaml.

    Returns
    -------
    tuple
        X_train_scaled,
        X_validation_scaled,
        X_test_scaled,
        scaler
    """

    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")

    if not isinstance(X_validation, pd.DataFrame):
        raise TypeError(
            "X_validation must be a pandas DataFrame."
        )

    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")

    if X_train.empty:
        raise ValueError("X_train is empty.")

    if X_validation.empty:
        raise ValueError(
            "X_validation is empty."
        )

    if X_test.empty:
        raise ValueError("X_test is empty.")

    if list(X_train.columns) != list(X_validation.columns):
        raise ValueError(
            "X_train and X_validation must have the same "
            "feature columns in the same order."
        )

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "X_train and X_test must have the same "
            "feature columns in the same order."
        )

    if X_train.isna().any().any():
        raise ValueError(
            "X_train contains missing values."
        )

    if X_validation.isna().any().any():
        raise ValueError(
            "X_validation contains missing values."
        )

    if X_test.isna().any().any():
        raise ValueError(
            "X_test contains missing values."
        )

    scaler = StandardScaler()

    # Fit ONLY on training data
    X_train_scaled = scaler.fit_transform(
        X_train
    )

    # Transform validation and test using
    # the scaler fitted on training data
    X_validation_scaled = scaler.transform(
        X_validation
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # Convert scaled arrays back to DataFrames
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

    # Get project root
    project_root = get_project_root()

    # Get artifact directory from config.yaml
    artifact_directory = (
        project_root
        / config["model_artifacts"]["preprocessing_dir"]
    )

    artifact_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Get scaler filename from config.yaml
    scaler_path = (
        artifact_directory
        / config["model_artifacts"]["churn"][
            "scaler_filename"
        ]
    )

    # Save fitted scaler
    joblib.dump(
        scaler,
        scaler_path,
    )

    print("=" * 75)
    print("CHURN FEATURE PREPROCESSING")
    print("=" * 75)

    print(
        f"Training features:    {X_train_scaled.shape}"
    )
    print(
        f"Validation features:  {X_validation_scaled.shape}"
    )
    print(
        f"Test features:        {X_test_scaled.shape}"
    )

    print(
        f"Scaler saved to:      {scaler_path}"
    )

    print("=" * 75)

    return (
        X_train_scaled,
        X_validation_scaled,
        X_test_scaled,
        scaler,
    )



################################################################
#-------- 2. Regression dataset builder ------------------------
################################################################


#------- 1. Create regression ClV dataset------------------------
def create_clv_regression_dataset(
    customer_features: pd.DataFrame,
    cleaned_transactions: pd.DataFrame,
    config: dict,
    cutoff_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Create a customer-level dataset for CLV regression.

    Predictor features come from customer_features and therefore
    represent customer behaviour available before the prediction
    cutoff.

    The CLV regression target is the customer's qualifying purchase
    value generated during the configured future prediction window.

    Returns/cancellations are not included directly in the future CLV
    target. They remain available in the transaction dataset and are
    used by other behavioural features such as Return_Rate.

    If cutoff_date is not supplied, it is automatically calculated
    according to the CLV cutoff rule defined in config.yaml.

    Parameters
    ----------
    customer_features : pd.DataFrame
        Customer-level historical features.

    cleaned_transactions : pd.DataFrame
        Cleaned transaction-level data.

    config : dict
        Project configuration loaded from config.yaml.

    cutoff_date : pd.Timestamp | None, default=None
        Optional prediction cutoff date. If None, the cutoff is
        calculated automatically using the configured cutoff rule.

    Returns
    -------
    pd.DataFrame
        Customer-level CLV regression dataset containing the
        historical predictor features and future CLV target.

    Raises
    ------
    TypeError
        If input objects have incorrect types.

    ValueError
        If required columns, configuration, dates, or validation
        conditions are invalid.
    """

    # =========================================================
    # 1. Validate input types
    # =========================================================

    if not isinstance(customer_features, pd.DataFrame):
        raise TypeError(
            "customer_features must be a pandas DataFrame."
        )

    if not isinstance(cleaned_transactions, pd.DataFrame):
        raise TypeError(
            "cleaned_transactions must be a pandas DataFrame."
        )

    if not isinstance(config, dict):
        raise TypeError(
            "config must be a dictionary loaded from config.yaml."
        )

    if customer_features.empty:
        raise ValueError(
            "customer_features cannot be empty."
        )

    if cleaned_transactions.empty:
        raise ValueError(
            "cleaned_transactions cannot be empty."
        )

    # =========================================================
    # 2. Validate customer feature columns
    # =========================================================

    required_customer_columns = {
        "Customer ID"
    }

    missing_customer_columns = (
        required_customer_columns
        - set(customer_features.columns)
    )

    if missing_customer_columns:
        raise ValueError(
            "customer_features is missing required columns: "
            f"{sorted(missing_customer_columns)}"
        )

    # =========================================================
    # 3. Validate transaction columns
    # =========================================================

    required_transaction_columns = {
        "Customer ID",
        "InvoiceDate",
        "Transaction Type",
        "Include In Sales Analysis",
        "Analytical Transaction Value",
    }

    missing_transaction_columns = (
        required_transaction_columns
        - set(cleaned_transactions.columns)
    )

    if missing_transaction_columns:
        raise ValueError(
            "cleaned_transactions is missing required columns: "
            f"{sorted(missing_transaction_columns)}"
        )

    # =========================================================
    # 4. Read CLV configuration
    # =========================================================

    try:
        clv_config = config["targets"]["clv"]

        prediction_window_days = clv_config[
            "prediction_window_days"
        ]

        cutoff_config = clv_config["cutoff"]

        cutoff_type = cutoff_config["type"]

    except KeyError as exc:
        raise ValueError(
            f"Missing required CLV configuration: {exc}"
        ) from exc

    # =========================================================
    # 5. Validate prediction window
    # =========================================================

    if not isinstance(prediction_window_days, int):
        raise TypeError(
            "targets.clv.prediction_window_days "
            "must be an integer."
        )

    if prediction_window_days <= 0:
        raise ValueError(
            "targets.clv.prediction_window_days "
            "must be greater than zero."
        )

    # =========================================================
    # 6. Work on a copy of transactions
    # =========================================================

    transactions = cleaned_transactions.copy()

    # =========================================================
    # 7. Convert InvoiceDate to datetime
    # =========================================================

    transactions["InvoiceDate"] = pd.to_datetime(
        transactions["InvoiceDate"],
        errors="coerce",
    )

    if transactions["InvoiceDate"].isna().any():
        raise ValueError(
            "cleaned_transactions contains invalid "
            "InvoiceDate values."
        )

    # =========================================================
    # 8. Determine cutoff date
    # =========================================================

    if cutoff_date is None:

        if cutoff_type != "max_date_minus_window":
            raise ValueError(
                "Unsupported CLV cutoff type: "
                f"{cutoff_type}"
            )

        max_transaction_date = transactions[
            "InvoiceDate"
        ].max()

        cutoff_date = (
            max_transaction_date
            - pd.Timedelta(
                days=prediction_window_days
            )
        )

    else:
        cutoff_date = pd.Timestamp(cutoff_date)

    # =========================================================
    # 9. Determine future CLV window
    # =========================================================

    future_end_date = (
        cutoff_date
        + pd.Timedelta(
            days=prediction_window_days
        )
    )

    # =========================================================
    # 10. Validate future window
    # =========================================================

    max_transaction_date = transactions[
        "InvoiceDate"
    ].max()

    if future_end_date > max_transaction_date:
        raise ValueError(
            "The future CLV prediction window extends beyond "
            "the available transaction data. "
            f"Cutoff: {cutoff_date}, "
            f"Future end: {future_end_date}, "
            f"Maximum transaction date: "
            f"{max_transaction_date}"
        )

    # =========================================================
    # 11. Select transactions after cutoff
    #     and within prediction window
    # =========================================================

    future_transactions = transactions[
        (transactions["InvoiceDate"] > cutoff_date)
        & (transactions["InvoiceDate"] <= future_end_date)
    ].copy()

    # =========================================================
    # 12. Keep only qualifying sales transactions
    #
    #     Important:
    #     - Purchase transactions contribute to future CLV.
    #     - Cancellations/returns do not create negative CLV.
    #     - Returns remain available in cleaned_transactions
    #       for behavioural features such as Return_Rate.
    # =========================================================

    future_transactions = future_transactions[
        (
            future_transactions[
                "Include In Sales Analysis"
            ].eq(1)
        )
        & (
            future_transactions[
                "Transaction Type"
            ].eq("Purchase")
        )
    ].copy()

    # =========================================================
    # 13. Define target column
    # =========================================================

    target_column = (
        f"Future_CLV_{prediction_window_days}D"
    )

    # =========================================================
    # 14. Aggregate future purchase value by customer
    # =========================================================

    future_clv = (
        future_transactions
        .groupby(
            "Customer ID",
            as_index=False,
        )[
            "Analytical Transaction Value"
        ]
        .sum()
        .rename(
            columns={
                "Analytical Transaction Value":
                    target_column
            }
        )
    )

    # =========================================================
    # 15. Start from customer_features
    #
    #     This ensures every customer in the modeling
    #     population remains in the CLV dataset.
    # =========================================================

    clv_dataset = customer_features.copy()

    clv_dataset = clv_dataset.merge(
        future_clv,
        on="Customer ID",
        how="left",
        validate="one_to_one",
    )

    # =========================================================
    # 16. Customers with no future qualifying purchase
    #     receive future CLV = 0
    # =========================================================

    clv_dataset[target_column] = (
        clv_dataset[target_column]
        .fillna(0.0)
    )

    # =========================================================
    # 17. Validate customer uniqueness
    # =========================================================

    if clv_dataset["Customer ID"].duplicated().any():
        raise ValueError(
            "CLV regression dataset contains duplicate "
            "Customer IDs."
        )

    # =========================================================
    # 18. Validate target values
    # =========================================================

    if clv_dataset[target_column].isna().any():
        raise ValueError(
            f"{target_column} contains missing values."
        )

    if (clv_dataset[target_column] < 0).any():
        raise ValueError(
            f"{target_column} contains negative values."
        )

    # =========================================================
    # 19. Validate target data type
    # =========================================================

    if not pd.api.types.is_numeric_dtype(
        clv_dataset[target_column]
    ):
        raise TypeError(
            f"{target_column} must be numeric."
        )

    # =========================================================
    # 20. Return final CLV regression dataset
    # =========================================================

    return clv_dataset


#------- 2.Validating CLV dataset --------------------------------
def validate_clv_regression_dataset(
    clv_dataset: pd.DataFrame,
    config: dict,
) -> dict:
    """
    Validate the customer-level CLV regression dataset.

    Checks:
    - Dataset is a DataFrame and not empty
    - Customer ID exists and is unique
    - Expected CLV target exists
    - Target contains no missing values
    - Target contains no negative values
    - Target is numeric
    - Predictor rows contain no missing values
    - Expected target window is correctly configured

    Returns
    -------
    dict
        Validation results and summary statistics.
    """

    # =========================================================
    # 1. Basic input validation
    # =========================================================

    if not isinstance(clv_dataset, pd.DataFrame):
        raise TypeError(
            "clv_dataset must be a pandas DataFrame."
        )

    if clv_dataset.empty:
        raise ValueError(
            "clv_dataset cannot be empty."
        )

    if not isinstance(config, dict):
        raise TypeError(
            "config must be a dictionary."
        )

    # =========================================================
    # 2. Read CLV configuration
    # =========================================================

    try:
        prediction_window_days = config[
            "targets"
        ][
            "clv"
        ][
            "prediction_window_days"
        ]

    except KeyError as exc:
        raise ValueError(
            f"Missing CLV configuration: {exc}"
        ) from exc

    target_column = (
        f"Future_CLV_{prediction_window_days}D"
    )

    # =========================================================
    # 3. Required columns
    # =========================================================

    required_columns = {
        "Customer ID",
        target_column,
    }

    missing_columns = (
        required_columns
        - set(clv_dataset.columns)
    )

    if missing_columns:
        raise ValueError(
            "CLV dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # =========================================================
    # 4. Customer ID validation
    # =========================================================

    duplicate_customer_ids = (
        clv_dataset["Customer ID"]
        .duplicated()
        .sum()
    )

    missing_customer_ids = (
        clv_dataset["Customer ID"]
        .isna()
        .sum()
    )

    # =========================================================
    # 5. Target validation
    # =========================================================

    target = clv_dataset[target_column]

    missing_target = target.isna().sum()

    negative_target = (
        (target < 0).sum()
    )

    non_numeric_target = (
        not pd.api.types.is_numeric_dtype(target)
    )

    # =========================================================
    # 6. Predictor missing-value validation
    # =========================================================

    predictor_columns = [
        column
        for column in clv_dataset.columns
        if column not in {
            "Customer ID",
            target_column,
        }
    ]

    missing_predictor_values = (
        clv_dataset[predictor_columns]
        .isna()
        .sum()
        .sum()
    )

    # =========================================================
    # 7. Zero and positive CLV statistics
    # =========================================================

    zero_clv_count = (
        target.eq(0).sum()
    )

    positive_clv_count = (
        target.gt(0).sum()
    )

    # =========================================================
    # 8. Target statistics
    # =========================================================

    target_mean = target.mean()
    target_median = target.median()
    target_std = target.std()
    target_min = target.min()
    target_max = target.max()

    # =========================================================
    # 9. Overall validation status
    # =========================================================

    passed = (
        duplicate_customer_ids == 0
        and missing_customer_ids == 0
        and missing_target == 0
        and negative_target == 0
        and non_numeric_target is False
        and missing_predictor_values == 0
    )

    # =========================================================
    # 10. Validation result
    # =========================================================

    validation_results = {
        "row_count": len(clv_dataset),
        "column_count": len(clv_dataset.columns),

        "duplicate_customer_id": (
            duplicate_customer_ids
        ),

        "missing_customer_id": (
            missing_customer_ids
        ),

        "missing_target": (
            missing_target
        ),

        "negative_target": (
            negative_target
        ),

        "non_numeric_target": (
            int(non_numeric_target)
        ),

        "missing_predictor_values": (
            missing_predictor_values
        ),

        "zero_clv_count": (
            zero_clv_count
        ),

        "positive_clv_count": (
            positive_clv_count
        ),

        "zero_clv_percentage": (
            zero_clv_count
            / len(clv_dataset)
            * 100
        ),

        "positive_clv_percentage": (
            positive_clv_count
            / len(clv_dataset)
            * 100
        ),

        "target_mean": target_mean,
        "target_median": target_median,
        "target_std": target_std,
        "target_min": target_min,
        "target_max": target_max,

        "target_column": target_column,
        "prediction_window_days": prediction_window_days,

        "passed": passed,
    }

    # =========================================================
    # 11. Fail loudly if validation fails
    # =========================================================

    if not passed:
        failed_checks = {
            key: value
            for key, value in validation_results.items()
            if key in {
                "duplicate_customer_id",
                "missing_customer_id",
                "missing_target",
                "negative_target",
                "non_numeric_target",
                "missing_predictor_values",
            }
            and value != 0
        }

        raise ValueError(
            "CLV regression dataset validation failed: "
            f"{failed_checks}"
        )

    return validation_results


#------- 3. Aanlyze_clv_dataset --------------------------------
def analyze_clv_dataset(
    clv_dataset: pd.DataFrame,
    config: dict,
) -> dict:
    """
    Analyze the statistical and structural quality of the
    customer-level CLV regression dataset.

    Parameters
    ----------
    clv_dataset : pd.DataFrame
        Customer-level CLV regression dataset containing
        customer features and the Future_CLV target.

    config : dict
        Project configuration containing the CLV prediction
        window.

    Returns
    -------
    dict
        Statistical and data-quality summary.
    """

    # ============================================================
    # 1. Input validation
    # ============================================================

    if not isinstance(clv_dataset, pd.DataFrame):
        raise TypeError(
            "clv_dataset must be a pandas DataFrame."
        )

    if clv_dataset.empty:
        raise ValueError(
            "clv_dataset is empty."
        )

    if not isinstance(config, dict):
        raise TypeError(
            "config must be a dictionary."
        )

    # ============================================================
    # 2. Read CLV configuration
    # ============================================================

    try:
        prediction_window_days = config[
            "targets"
        ][
            "clv"
        ][
            "prediction_window_days"
        ]

    except KeyError as exc:
        raise ValueError(
            f"Missing CLV configuration: {exc}"
        ) from exc

    target_column = (
        f"Future_CLV_{prediction_window_days}D"
    )

    # ============================================================
    # 3. Basic structure
    # ============================================================

    shape = clv_dataset.shape

    row_count = shape[0]
    column_count = shape[1]

    # ============================================================
    # 4. Duplicate analysis
    # ============================================================

    duplicate_rows = int(
        clv_dataset.duplicated().sum()
    )

    duplicate_customer_ids = 0

    if "Customer ID" in clv_dataset.columns:
        duplicate_customer_ids = int(
            clv_dataset["Customer ID"]
            .duplicated()
            .sum()
        )

    # ============================================================
    # 5. Missing-value analysis
    # ============================================================

    missing_values = (
        clv_dataset.isna()
        .sum()
        .sort_values(ascending=False)
    )

    missing_columns = (
        missing_values[
            missing_values > 0
        ]
        .to_dict()
    )

    total_missing_values = int(
        missing_values.sum()
    )

    # ============================================================
    # 6. Data types
    # ============================================================

    data_types = {
        column: str(dtype)
        for column, dtype
        in clv_dataset.dtypes.items()
    }

    # ============================================================
    # 7. CLV target statistics
    # ============================================================

    clv_statistics = {}

    if target_column in clv_dataset.columns:

        clv_target = clv_dataset[target_column]

        zero_clv_count = int(
            (clv_target == 0).sum()
        )

        positive_clv_count = int(
            (clv_target > 0).sum()
        )

        negative_clv_count = int(
            (clv_target < 0).sum()
        )

        clv_statistics = {
            "target_column": target_column,

            "prediction_window_days": (
                prediction_window_days
            ),

            "zero_clv_count": (
                zero_clv_count
            ),

            "positive_clv_count": (
                positive_clv_count
            ),

            "negative_clv_count": (
                negative_clv_count
            ),

            "zero_clv_percentage": float(
                (clv_target == 0).mean() * 100
            ),

            "positive_clv_percentage": float(
                (clv_target > 0).mean() * 100
            ),

            "negative_clv_percentage": float(
                (clv_target < 0).mean() * 100
            ),

            "mean_clv": float(
                clv_target.mean()
            ),

            "median_clv": float(
                clv_target.median()
            ),

            "standard_deviation": float(
                clv_target.std()
            ),

            "minimum_clv": float(
                clv_target.min()
            ),

            "maximum_clv": float(
                clv_target.max()
            ),

            "percentile_25": float(
                clv_target.quantile(0.25)
            ),

            "percentile_75": float(
                clv_target.quantile(0.75)
            ),

            "percentile_90": float(
                clv_target.quantile(0.90)
            ),

            "percentile_95": float(
                clv_target.quantile(0.95)
            ),

            "percentile_99": float(
                clv_target.quantile(0.99)
            ),
        }

    # ============================================================
    # 8. Numerical descriptive statistics
    # ============================================================

    numerical_statistics = (
        clv_dataset
        .select_dtypes(include="number")
        .describe()
        .round(4)
    )

    # ============================================================
    # 9. Predictor columns
    # ============================================================

    predictor_columns = [
        column
        for column in clv_dataset.columns
        if column not in [
            "Customer ID",
            target_column,
        ]
    ]

    # ============================================================
    # 10. Basic quality checks
    # ============================================================

    quality_checks = {
        "no_duplicate_rows": (
            duplicate_rows == 0
        ),

        "no_duplicate_customer_ids": (
            duplicate_customer_ids == 0
        ),

        "no_missing_values": (
            total_missing_values == 0
        ),

        "future_clv_column_present": (
            target_column in clv_dataset.columns
        ),

        "future_clv_is_numeric": (
            target_column in clv_dataset.columns
            and pd.api.types.is_numeric_dtype(
                clv_dataset[target_column]
            )
        ),

        "no_negative_future_clv": (
            target_column in clv_dataset.columns
            and not (
                clv_dataset[target_column] < 0
            ).any()
        ),

        "predictor_features_present": (
            len(predictor_columns) > 0
        ),
    }

    # ============================================================
    # 11. Overall quality status
    # ============================================================

    quality_status = all(
        quality_checks.values()
    )

    # ============================================================
    # 12. Result
    # ============================================================

    return {
        "shape": shape,

        "row_count": row_count,

        "column_count": column_count,

        "duplicate_rows": (
            duplicate_rows
        ),

        "duplicate_customer_ids": (
            duplicate_customer_ids
        ),

        "total_missing_values": (
            total_missing_values
        ),

        "missing_columns": (
            missing_columns
        ),

        "data_types": (
            data_types
        ),

        "clv_statistics": (
            clv_statistics
        ),

        "numerical_statistics": (
            numerical_statistics
        ),

        "quality_checks": (
            quality_checks
        ),

        "quality_status": (
            quality_status
        ),
    }


# 4. preprcessing datasets for churn
def preprocess_clv_features(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    config: dict,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    StandardScaler,
]:
    """
    Preprocess CLV-model features using StandardScaler.

    The scaler is fitted only on the training data and then
    used to transform the validation and test datasets.

    The fitted scaler is saved using the path configured
    in config.yaml.

    Returns
    -------
    tuple
        X_train_scaled,
        X_validation_scaled,
        X_test_scaled,
        scaler
    """

    # ============================================================
    # 1. Validate input types
    # ============================================================

    if not isinstance(X_train, pd.DataFrame):
        raise TypeError(
            "X_train must be a pandas DataFrame."
        )

    if not isinstance(X_validation, pd.DataFrame):
        raise TypeError(
            "X_validation must be a pandas DataFrame."
        )

    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(
            "X_test must be a pandas DataFrame."
        )

    # ============================================================
    # 2. Validate datasets are not empty
    # ============================================================

    if X_train.empty:
        raise ValueError(
            "X_train is empty."
        )

    if X_validation.empty:
        raise ValueError(
            "X_validation is empty."
        )

    if X_test.empty:
        raise ValueError(
            "X_test is empty."
        )

    # ============================================================
    # 3. Validate feature columns
    # ============================================================

    if list(X_train.columns) != list(
        X_validation.columns
    ):
        raise ValueError(
            "X_train and X_validation must have the same "
            "feature columns in the same order."
        )

    if list(X_train.columns) != list(
        X_test.columns
    ):
        raise ValueError(
            "X_train and X_test must have the same "
            "feature columns in the same order."
        )

    # ============================================================
    # 4. Validate missing values
    # ============================================================

    if X_train.isna().any().any():
        raise ValueError(
            "X_train contains missing values."
        )

    if X_validation.isna().any().any():
        raise ValueError(
            "X_validation contains missing values."
        )

    if X_test.isna().any().any():
        raise ValueError(
            "X_test contains missing values."
        )

    # ============================================================
    # 5. Create scaler
    # ============================================================

    scaler = StandardScaler()

    # ============================================================
    # 6. Fit ONLY on training data
    # ============================================================

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    # ============================================================
    # 7. Transform validation and test
    # ============================================================

    X_validation_scaled = scaler.transform(
        X_validation
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # ============================================================
    # 8. Convert scaled arrays back to DataFrames
    # ============================================================

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

    # ============================================================
    # 9. Get project root
    # ============================================================

    project_root = get_project_root()

    # ============================================================
    # 10. Get preprocessing artifact directory
    # ============================================================

    artifact_directory = (
        project_root
        / config[
            "model_artifacts"
        ][
            "preprocessing_dir"
        ]
    )

    artifact_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ============================================================
    # 11. Get CLV scaler filename
    # ============================================================

    scaler_path = (
        artifact_directory
        / config[
            "model_artifacts"
        ][
            "clv"
        ][
            "scaler_filename"
        ]
    )

    # ============================================================
    # 12. Save fitted scaler
    # ============================================================

    joblib.dump(
        scaler,
        scaler_path,
    )

    # ============================================================
    # 13. Display preprocessing information
    # ============================================================

    print("=" * 75)
    print("CLV FEATURE PREPROCESSING")
    print("=" * 75)

    print(
        f"Training features:    "
        f"{X_train_scaled.shape}"
    )

    print(
        f"Validation features:  "
        f"{X_validation_scaled.shape}"
    )

    print(
        f"Test features:        "
        f"{X_test_scaled.shape}"
    )

    print(
        f"Scaler saved to:      "
        f"{scaler_path}"
    )

    print("=" * 75)

    # ============================================================
    # 14. Return processed features
    # ============================================================

    return (
        X_train_scaled,
        X_validation_scaled,
        X_test_scaled,
        scaler,
    )
