import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

def analyze_structure(df: pd.DataFrame) -> dict:
    """
    Analyze the basic structure of the dataset.

    Includes:
    - Number of rows
    - Number of columns
    - Column names
    - Data types
    - Date range
    """

    result = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "data_types": df.dtypes.astype(str).to_dict(),
    }

    # Date analysis
    if "InvoiceDate" in df.columns:

        dates = pd.to_datetime(
            df["InvoiceDate"],
            errors="coerce"
        )

        result["start_date"] = dates.min()
        result["end_date"] = dates.max()
        result["invalid_dates"] = int(
            dates.isna().sum()
        )

    return result

def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze missing values across all columns.

    Returns:
    - Missing count
    - Missing percentage
    """

    result = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percentage": (
            df.isna().mean() * 100
        ).round(2)
    })

    return result.sort_values(
        by="missing_count",
        ascending=False
    )

def analyze_duplicates(df: pd.DataFrame) -> dict:
    """
    Analyze exact duplicate records across all columns.

    The analysis identifies:
    - Number of duplicate occurrences
    - Percentage of duplicate occurrences
    - Total rows belonging to duplicate groups
    - Whether duplicates are exact across all columns
    - Sample duplicate records
    """

    # ---------------------------------------------------------
    # Identify exact duplicates across ALL columns
    # ---------------------------------------------------------

    duplicate_mask = df.duplicated(
        subset=df.columns,
        keep=False
    )

    duplicate_rows = df[
        duplicate_mask
    ].copy()

    # Number of duplicate occurrences after
    # keeping the first occurrence
    duplicate_count = int(
        df.duplicated(
            subset=df.columns,
            keep="first"
        ).sum()
    )

    # ---------------------------------------------------------
    # Duplicate percentage
    # ---------------------------------------------------------

    duplicate_percentage = round(
        duplicate_count / len(df) * 100,
        2
    )

    # ---------------------------------------------------------
    # Number of unique duplicate groups
    # ---------------------------------------------------------

    if not duplicate_rows.empty:

        duplicate_groups = (
            duplicate_rows
            .groupby(
                list(df.columns),
                dropna=False
            )
            .size()
            .reset_index(
                name="duplicate_count"
            )
        )

        duplicate_group_count = len(
            duplicate_groups
        )

        # Check whether each duplicate group
        # contains truly identical rows
        exact_duplicates = (
            duplicate_groups["duplicate_count"] > 1
        ).all()

    else:

        duplicate_groups = pd.DataFrame()

        duplicate_group_count = 0

        exact_duplicates = True

    # ---------------------------------------------------------
    # Sample exact duplicate rows
    # ---------------------------------------------------------

    duplicate_sample = duplicate_rows.head(10)

    # ---------------------------------------------------------
    # Return analysis
    # ---------------------------------------------------------

    return {

        "duplicate_count": duplicate_count,

        "duplicate_percentage": (
            duplicate_percentage
        ),

        "duplicate_group_rows": len(
            duplicate_rows
        ),

        "duplicate_groups": (
            duplicate_group_count
        ),

        "exact_duplicates": (
            exact_duplicates
        ),

        "sample_duplicate_rows": (
            duplicate_sample
        ),
    }

def analyze_data_quality(df: pd.DataFrame) -> dict:
    """
    Analyze the major data-quality characteristics
    identified during the initial EDA.

    Includes:
    - Quantity
    - Price
    - Customer ID
    - Description
    - StockCode
    - Country
    """

    total_rows = len(df)

    # ---------------------------------------------------------
    # Quantity
    # ---------------------------------------------------------

    negative_quantity_mask = df["Quantity"] < 0

    negative_quantity = df[
             negative_quantity_mask
       ].copy()

    negative_quantity_count = len(
        negative_quantity
          )

    zero_quantity = int(
      (df["Quantity"] == 0).sum()
      )

    positive_quantity = int(
    (df["Quantity"] > 0).sum()
       )

    # ---------------------------------------------------------
    # Price
    # ---------------------------------------------------------

    negative_price = int(
        (df["Price"] < 0).sum()
    )

    zero_price = int(
        (df["Price"] == 0).sum()
    )

    positive_price = int(
        (df["Price"] > 0).sum()
    )

    # ---------------------------------------------------------
    # Customer ID
    # ---------------------------------------------------------

    missing_customer_id = int(
        df["Customer ID"].isna().sum()
    )

    unique_customers = int(
        df["Customer ID"].nunique(
            dropna=True
        )
    )

    # ---------------------------------------------------------
    # Description
    # ---------------------------------------------------------

    missing_description = int(
        df["Description"].isna().sum()
    )

    # ---------------------------------------------------------
    # StockCode
    # ---------------------------------------------------------

    unique_stock_codes = int(
        df["StockCode"].nunique()
    )

    missing_stock_codes = int(
        df["StockCode"].isna().sum()
    )

    alphabetic_stockcode_rows = int(
        df["StockCode"]
        .astype(str)
        .str.contains(
            r"[A-Za-z]",
            regex=True,
            na=False
        )
        .sum()
    )

    # ---------------------------------------------------------
    # Invoice / Country
    # ---------------------------------------------------------

    unique_invoices = int(
        df["Invoice"].nunique()
    )

    unique_countries = int(
        df["Country"].nunique()
    )

    # ---------------------------------------------------------
    # Return result
    # ---------------------------------------------------------

    return {

        "quantity": {
            "negative": negative_quantity_count,
            "negative_percentage": round(negative_quantity_count / total_rows * 100,2),
            "zero": zero_quantity,
            "positive": positive_quantity,
            "negative_sample": negative_quantity.head(20),
        },

        "price": {
            "negative": negative_price,
            "zero": zero_price,
            "positive": positive_price,
            "negative_percentage": round(
                negative_price / total_rows * 100,
                2
            ),
            "zero_percentage": round(
                zero_price / total_rows * 100,
                2
            ),
        },

        "customer": {
            "unique_customers": unique_customers,
            "missing_customer_id": missing_customer_id,
            "missing_customer_percentage": round(
                missing_customer_id / total_rows * 100,
                2
            ),
        },

        "description": {
            "missing": missing_description,
            "missing_percentage": round(
                missing_description / total_rows * 100,
                2
            ),
        },

        "stock_code": {
            "unique_stock_codes": unique_stock_codes,
            "missing": missing_stock_codes,
            "alphabetic_rows": alphabetic_stockcode_rows,
        },

        "invoice": {
            "unique_invoices": unique_invoices,
        },

        "country": {
            "unique_countries": unique_countries,
        },
    }

def stockcode_data_quality(df: pd.DataFrame) -> dict:
    """
    Analyze StockCode quality and identify non-standard StockCodes.

    Includes:
    - Unique StockCodes
    - Missing StockCodes
    - StockCodes containing alphabetic characters
    - StockCodes starting with A, B, C, or D
    - Specific investigation of StockCode == "B"
    """

    stock_code = df["StockCode"].astype(str)

    # ---------------------------------------------------------
    # Basic StockCode statistics
    # ---------------------------------------------------------

    unique_stock_codes = int(
        df["StockCode"].nunique()
    )

    missing_stock_codes = int(
        df["StockCode"].isna().sum()
    )

    # ---------------------------------------------------------
    # StockCodes containing alphabetic characters
    # ---------------------------------------------------------

    alphabetic_stockcode_mask = stock_code.str.contains(
        r"[A-Za-z]",
        regex=True,
        na=False
    )

    alphabetic_stockcode_count = int(
        alphabetic_stockcode_mask.sum()
    )

    alphabetic_stockcode_rows = df[
        alphabetic_stockcode_mask
    ].copy()

    # ---------------------------------------------------------
    # StockCodes starting with A, B, C, or D
    # ---------------------------------------------------------

    prefix_stockcode_mask = stock_code.str.startswith(
        ("A", "B", "C", "D")
    )

    prefix_stockcode_count = int(
        prefix_stockcode_mask.sum()
    )

    prefix_stockcode_rows = df[
        prefix_stockcode_mask
    ].copy()

    # ---------------------------------------------------------
    # Specific StockCode == "B"
    # ---------------------------------------------------------

    stock_code_b_mask = stock_code == "B"

    stock_code_b_rows = df[
        stock_code_b_mask
    ].copy()

    stock_code_b_count = len(
        stock_code_b_rows
    )

    # ---------------------------------------------------------
    # Return analysis results
    # ---------------------------------------------------------

    return {

        "basic": {

            "unique_stock_codes": unique_stock_codes,

            "missing_stock_codes": missing_stock_codes,
        },

        "alphabetic_stockcodes": {

            "count": alphabetic_stockcode_count,

            "sample": (
                alphabetic_stockcode_rows[
                    [
                        "Invoice",
                        "StockCode",
                        "Description",
                        "Quantity",
                        "Price"
                    ]
                ].head(50)
            ),
        },

        "prefix_abcd_stockcodes": {

            "count": prefix_stockcode_count,

            "sample": (
                prefix_stockcode_rows[
                    [
                        "Invoice",
                        "StockCode",
                        "Description",
                        "Quantity",
                        "Price"
                    ]
                ].head(30)
            ),
        },

        "stockcode_b": {

            "count": stock_code_b_count,

            "records": (
                stock_code_b_rows[
                    [
                        "Invoice",
                        "StockCode",
                        "Description",
                        "Quantity",
                        "Price",
                        "Customer ID",
                        "Country"
                    ]
                ].head(30)
            ),
        },
    }

def display_analysis(title: str, result) -> None:
    """
    Display analysis results in a clean, readable format.

    Parameters
    ----------
    title : str
        Title of the analysis section.

    result : dict or pandas.DataFrame
        Result returned by one of the analysis functions.
    """

    print("\n" + "=" * 70)
    print(f"{title:^70}")
    print("=" * 70)

    # ---------------------------------------------------------
    # Dictionary result
    # ---------------------------------------------------------

    if isinstance(result, dict):

        def print_dictionary(data, level=0):

            for key, value in data.items():

                # Convert key into readable text
                formatted_key = (
                    str(key)
                    .replace("_", " ")
                    .title()
                )

                indent = "    " * level

                # -------------------------------------------------
                # Nested dictionary
                # -------------------------------------------------

                if isinstance(value, dict):

                    print(
                        f"\n{indent}{formatted_key}"
                    )

                    print(
                        f"{indent}{'-' * len(formatted_key)}"
                    )

                    print_dictionary(
                        value,
                        level + 1
                    )

                # -------------------------------------------------
                # DataFrame
                # -------------------------------------------------

                elif isinstance(value, pd.DataFrame):

                    print(
                        f"\n{indent}{formatted_key}"
                    )

                    print(
                        f"{indent}{'-' * len(formatted_key)}"
                    )

                    if value.empty:

                        print(
                            f"{indent}No records found."
                        )

                    else:

                        print(
                            value.to_string(
                                index=False
                            )
                        )

                # -------------------------------------------------
                # Float
                # -------------------------------------------------

                elif isinstance(value, float):

                    print(
                        f"{indent}"
                        f"{formatted_key:<35}: "
                        f"{value:.2f}"
                    )

                # -------------------------------------------------
                # Integer
                # -------------------------------------------------

                elif isinstance(value, int):

                    print(
                        f"{indent}"
                        f"{formatted_key:<35}: "
                        f"{value:,}"
                    )

                # -------------------------------------------------
                # Other values
                # -------------------------------------------------

                else:

                    print(
                        f"{indent}"
                        f"{formatted_key:<35}: "
                        f"{value}"
                    )

        print_dictionary(result)

    # ---------------------------------------------------------
    # DataFrame result
    # ---------------------------------------------------------

    elif isinstance(result, pd.DataFrame):

        print()

        if result.empty:

            print("No records found.")

        else:

            print(
                result.to_string(
                    index=False
                )
            )

    # ---------------------------------------------------------
    # Unsupported result
    # ---------------------------------------------------------

    else:

        print(
            f"Unsupported result type: "
            f"{type(result).__name__}"
        )

    print("=" * 70)

#########################################################################
#--- Pre Cleaning Exploratory Data Analysis (EDA) Functions--------------
#########################################################################

############################ descriptive_statistics #####################
def analyze_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate descriptive statistics for important numerical variables.

    Variables:
    - Quantity
    - Price
    """

    numerical_columns = [
        "Quantity",
        "Price"
    ]

    result = df[numerical_columns].describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99
        ]
    ).T

    result["median"] = df[
        numerical_columns
    ].median()

    result["skewness"] = df[
        numerical_columns
    ].skew()

    result["IQR"] = (
        result["75%"] - result["25%"]
    )

    result = result[
        [
            "count",
            "mean",
            "std",
            "min",
            "1%",
            "5%",
            "25%",
            "median",
            "50%",
            "75%",
            "95%",
            "99%",
            "max",
            "IQR",
            "skewness"
        ]
    ]

    return result.round(2)

####################### transaction_value ################################
def analyze_transaction_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze transaction value calculated as Quantity × Price.

    This analysis is performed without modifying the original DataFrame.
    """

    transaction_value = (
        df["Quantity"] * df["Price"]
    )

    result = pd.DataFrame({
        "count": [
            transaction_value.count()
        ],

        "mean": [
            transaction_value.mean()
        ],

        "std": [
            transaction_value.std()
        ],

        "min": [
            transaction_value.min()
        ],

        "1%": [
            transaction_value.quantile(0.01)
        ],

        "5%": [
            transaction_value.quantile(0.05)
        ],

        "25%": [
            transaction_value.quantile(0.25)
        ],

        "median": [
            transaction_value.median()
        ],

        "75%": [
            transaction_value.quantile(0.75)
        ],

        "95%": [
            transaction_value.quantile(0.95)
        ],

        "99%": [
            transaction_value.quantile(0.99)
        ],

        "max": [
            transaction_value.max()
        ],

        "IQR": [
            transaction_value.quantile(0.75)
            - transaction_value.quantile(0.25)
        ],

        "skewness": [
            transaction_value.skew()
        ]
    })

    return result.round(2)

##########################categorical date ###############################
def analyze_categorical_data(df: pd.DataFrame) -> dict:

    """
    Analyze important categorical variables.

    Includes:
    - Country
    - StockCode
    - Invoice
    """

    result = {

        "country": {
            "unique_count": int(
                df["Country"].nunique()
            ),
            "top_values": (
                df["Country"]
                .value_counts()
                .head(20)
            )
        },

        "stock_code": {
            "unique_count": int(
                df["StockCode"].nunique()
            ),
            "top_values": (
                df["StockCode"]
                .value_counts()
                .head(20)
            )
        },

        "invoice": {
            "unique_count": int(
                df["Invoice"].nunique()
            ),
            "top_values": (
                df["Invoice"]
                .value_counts()
                .head(20)
            )
        }
    }

    return result

#######################plot Quantity Distribution#########################
def plot_quantity_distribution(
    df: pd.DataFrame,
    bins: int = 60
) -> None:
    """
    Plot the Quantity distribution using the 1st to 99th percentile
    range for better visualization.

    Extreme values are not removed from the dataset. They are excluded
    only from the displayed range.
    """

    quantity = df["Quantity"].dropna()

    lower_limit = quantity.quantile(0.01)
    upper_limit = quantity.quantile(0.99)

    quantity_plot = quantity[
        (quantity >= lower_limit)
        & (quantity <= upper_limit)
    ]

    excluded_count = len(quantity) - len(quantity_plot)

    plt.figure(figsize=(10, 6))

    sns.histplot(
        quantity_plot,
        bins=bins,
        kde=True
    )

    plt.title(
        "Quantity Distribution "
        "(1st–99th Percentile)"
    )

    plt.xlabel("Quantity")
    plt.ylabel("Frequency")

    plt.axvline(
        quantity_plot.median(),
        linestyle="--",
        label=f"Median: {quantity_plot.median():.0f}"
    )

    plt.legend()

    plt.text(
        0.98,
        0.95,
        f"Excluded from view: {excluded_count:,} extreme observations",
        transform=plt.gca().transAxes,
        ha="right",
        va="top"
    )

    plt.tight_layout()
    plt.show()

########################plot Price Distribution###########################
def plot_price_distribution(
    df: pd.DataFrame,
    bins: int = 60
) -> None:
    """
    Plot the Price distribution using the 1st to 99th percentile
    range for better visualization.

    Extreme values are not removed from the dataset. They are excluded
    only from the displayed range.
    """

    price = df["Price"].dropna()

    lower_limit = price.quantile(0.01)
    upper_limit = price.quantile(0.99)

    price_plot = price[
        (price >= lower_limit)
        & (price <= upper_limit)
    ]

    excluded_count = len(price) - len(price_plot)

    plt.figure(figsize=(10, 6))

    sns.histplot(
        price_plot,
        bins=bins,
        kde=True
    )

    plt.title(
        "Price Distribution "
        "(1st–99th Percentile)"
    )

    plt.xlabel("Price")
    plt.ylabel("Frequency")

    plt.axvline(
        price_plot.median(),
        linestyle="--",
        label=f"Median: {price_plot.median():.2f}"
    )

    plt.legend()

    plt.text(
        0.98,
        0.95,
        f"Excluded from view: {excluded_count:,} extreme observations",
        transform=plt.gca().transAxes,
        ha="right",
        va="top"
    )

    plt.tight_layout()
    plt.show()

################plot transactional value distribution######################
def plot_transaction_value_distribution(
    df: pd.DataFrame,
    bins: int = 60
) -> None:
    """
    Plot the Transaction Value distribution using the 1st to 99th
    percentile range.

    Transaction Value = Quantity × Price

    Extreme values are not removed from the dataset. They are excluded
    only from the displayed range.
    """

    transaction_value = (
        df["Quantity"] * df["Price"]
    ).dropna()

    lower_limit = transaction_value.quantile(0.01)
    upper_limit = transaction_value.quantile(0.99)

    transaction_plot = transaction_value[
        (transaction_value >= lower_limit)
        & (transaction_value <= upper_limit)
    ]

    excluded_count = (
        len(transaction_value)
        - len(transaction_plot)
    )

    plt.figure(figsize=(10, 6))

    sns.histplot(
        transaction_plot,
        bins=bins,
        kde=True
    )

    plt.title(
        "Transaction Value Distribution "
        "(1st–99th Percentile)"
    )

    plt.xlabel("Transaction Value")
    plt.ylabel("Frequency")

    plt.axvline(
        transaction_plot.median(),
        linestyle="--",
        label=f"Median: {transaction_plot.median():.2f}"
    )

    plt.legend()

    plt.text(
        0.98,
        0.95,
        f"Excluded from view: {excluded_count:,} extreme observations",
        transform=plt.gca().transAxes,
        ha="right",
        va="top"
    )

    plt.tight_layout()
    plt.show()



#############################################################################
#---- FEATURED ENGINEERED EDA CHURN DATADSET=----
#############################################################################

#------------ 1. Churndaset distribution ------------
def analyze_churn_distribution(
    churn_dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze and visualize the distribution of the Churn target.

    Parameters
    ----------
    churn_dataset : pd.DataFrame
        Customer-level churn dataset containing the 'Churn' column.

    Returns
    -------
    pd.DataFrame
        Summary table containing churn class, customer count,
        and percentage.
    """

    if not isinstance(churn_dataset, pd.DataFrame):
        raise TypeError(
            "churn_dataset must be a pandas DataFrame."
        )

    if churn_dataset.empty:
        raise ValueError(
            "churn_dataset is empty."
        )

    if "Churn" not in churn_dataset.columns:
        raise ValueError(
            "churn_dataset must contain a 'Churn' column."
        )

    if churn_dataset["Churn"].isna().any():
        raise ValueError(
            "Churn contains missing values."
        )

    if not set(churn_dataset["Churn"].unique()).issubset({0, 1}):
        raise ValueError(
            "Churn must contain only 0 and 1."
        )

    churn_counts = (
        churn_dataset["Churn"]
        .value_counts()
        .sort_index()
    )

    churn_percentages = (
        churn_dataset["Churn"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    churn_summary = pd.DataFrame({
        "Churn": churn_counts.index,
        "Customer_Count": churn_counts.values,
        "Percentage": churn_percentages.values,
    })

    churn_summary["Customer_Status"] = (
        churn_summary["Churn"]
        .map({
            0: "Non-Churn",
            1: "Churn",
        })
    )

    churn_summary = churn_summary[
        [
            "Churn",
            "Customer_Status",
            "Customer_Count",
            "Percentage",
        ]
    ]

    plt.figure(figsize=(7, 5))

    plt.bar(
        churn_summary["Customer_Status"],
        churn_summary["Customer_Count"],
    )

    plt.title("Customer Churn Distribution")
    plt.xlabel("Customer Status")
    plt.ylabel("Number of Customers")

    plt.tight_layout()
    plt.show()

    return churn_summary


#------------ 2. numerical feature analysis--------------------
def analyze_numerical_features(
    churn_dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze descriptive statistics of numerical predictor features
    and visualize their skewness.

    Customer ID and Churn are excluded because Customer ID is an
    identifier and Churn is the target variable.

    Parameters
    ----------
    churn_dataset : pd.DataFrame
        Customer-level churn dataset.

    Returns
    -------
    pd.DataFrame
        Descriptive statistics for numerical predictor features.
    """

    if not isinstance(churn_dataset, pd.DataFrame):
        raise TypeError(
            "churn_dataset must be a pandas DataFrame."
        )

    if churn_dataset.empty:
        raise ValueError(
            "churn_dataset is empty."
        )

    required_columns = {
        "Customer ID",
        "Churn",
    }

    missing_columns = (
        required_columns
        - set(churn_dataset.columns)
    )

    if missing_columns:
        raise ValueError(
            "churn_dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    predictor_columns = [
        column
        for column in churn_dataset.columns
        if column not in [
            "Customer ID",
            "Churn",
        ]
    ]

    numerical_features = (
        churn_dataset[predictor_columns]
        .select_dtypes(include="number")
    )

    if numerical_features.empty:
        raise ValueError(
            "No numerical predictor features found."
        )

    statistics = (
        numerical_features
        .describe()
        .T
    )

    statistics["skewness"] = (
        numerical_features
        .skew()
    )

    statistics = (
        statistics
        .round(4)
    )

    # --------------------------------------------------
    # Skewness Visualization
    # --------------------------------------------------

    skewness = (
        statistics["skewness"]
        .sort_values()
    )

    plt.figure(figsize=(10, 9))

    plt.barh(
        skewness.index,
        skewness.values,
    )

    plt.axvline(
        0,
        linewidth=1,
    )

    plt.title(
        "Numerical Feature Skewness"
    )

    plt.xlabel(
        "Skewness"
    )

    plt.ylabel(
        "Feature"
    )

    plt.tight_layout()
    plt.show()

    return statistics

#------------ 3A. feature distribution --------------------
def analyze_feature_distributions(
    churn_dataset: pd.DataFrame,
) -> None:
    """
    Visualize distributions and potential outliers of numerical
    predictor features.

    Customer ID and Churn are excluded.

    Parameters
    ----------
    churn_dataset : pd.DataFrame
        Customer-level churn dataset.

    Returns
    -------
    None
        Displays histogram and boxplot visualizations.
    """

    if not isinstance(churn_dataset, pd.DataFrame):
        raise TypeError(
            "churn_dataset must be a pandas DataFrame."
        )

    if churn_dataset.empty:
        raise ValueError(
            "churn_dataset is empty."
        )

    if "Churn" not in churn_dataset.columns:
        raise ValueError(
            "churn_dataset must contain a 'Churn' column."
        )

    predictor_columns = [
        column
        for column in churn_dataset.columns
        if column not in [
            "Customer ID",
            "Churn",
        ]
    ]

    numerical_features = (
        churn_dataset[predictor_columns]
        .select_dtypes(include="number")
    )

    if numerical_features.empty:
        raise ValueError(
            "No numerical predictor features found."
        )

    # --------------------------------------------------
    # Histogram
    # --------------------------------------------------

    numerical_features.hist(
        figsize=(18, 22),
        bins=30,
        layout=None,
    )

    plt.suptitle(
        "Numerical Feature Distributions",
        fontsize=16,
        y=1.02,
    )

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # Boxplot
    # --------------------------------------------------

    feature_count = len(numerical_features.columns)

    rows = (feature_count + 2) // 3

    fig, axes = plt.subplots(
        rows,
        3,
        figsize=(18, rows * 4),
    )

    axes = axes.flatten()

    for index, column in enumerate(
        numerical_features.columns
    ):
        axes[index].boxplot(
            numerical_features[column].dropna()
        )

        axes[index].set_title(column)
        axes[index].set_ylabel("Value")

    for index in range(
        feature_count,
        len(axes)
    ):
        axes[index].set_visible(False)

    fig.suptitle(
        "Numerical Feature Boxplots",
        fontsize=16,
    )

    plt.tight_layout()
    plt.show()


#------------- 3B analyze outliers --------------------------------
def analyze_feature_outliers(
    churn_dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze IQR-based outliers for numerical predictor features.

    Customer ID and Churn are excluded.

    Parameters
    ----------
    churn_dataset : pd.DataFrame
        Customer-level churn dataset.

    Returns
    -------
    pd.DataFrame
        Outlier statistics for each numerical feature.
    """

    if not isinstance(churn_dataset, pd.DataFrame):
        raise TypeError(
            "churn_dataset must be a pandas DataFrame."
        )

    if churn_dataset.empty:
        raise ValueError(
            "churn_dataset is empty."
        )

    predictor_data = (
        churn_dataset
        .drop(columns=["Customer ID", "Churn"])
        .select_dtypes(include="number")
    )

    if predictor_data.empty:
        raise ValueError(
            "No numerical predictor features found."
        )

    q1 = predictor_data.quantile(0.25)
    q3 = predictor_data.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    outlier_counts = (
        (predictor_data.lt(lower_bound)) |
        (predictor_data.gt(upper_bound))
    ).sum()

    outlier_percentages = (
        outlier_counts
        / len(predictor_data)
        * 100
    )

    outlier_summary = pd.DataFrame({
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Lower_Bound": lower_bound,
        "Upper_Bound": upper_bound,
        "Outlier_Count": outlier_counts,
        "Outlier_Percentage": outlier_percentages,
    })

    outlier_summary = (
        outlier_summary
        .sort_values(
            "Outlier_Percentage",
            ascending=False,
        )
        .round(4)
    )

    return outlier_summary


#------------- 4. analyze feature by churn----------------
def analyze_features_by_churn(
    churn_dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare numerical predictor features between churned
    and non-churned customers and visualize the largest
    mean differences.

    Customer ID and Churn are excluded from the predictors.

    Parameters
    ----------
    churn_dataset : pd.DataFrame
        Customer-level churn dataset.

    Returns
    -------
    pd.DataFrame
        Feature-level comparison between Churn = 0
        and Churn = 1.
    """

    if not isinstance(churn_dataset, pd.DataFrame):
        raise TypeError(
            "churn_dataset must be a pandas DataFrame."
        )

    if churn_dataset.empty:
        raise ValueError(
            "churn_dataset is empty."
        )

    if "Churn" not in churn_dataset.columns:
        raise ValueError(
            "churn_dataset must contain a 'Churn' column."
        )

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

    predictor_data = (
        churn_dataset
        .drop(
            columns=["Customer ID", "Churn"]
        )
        .select_dtypes(include="number")
    )

    if predictor_data.empty:
        raise ValueError(
            "No numerical predictor features found."
        )

    churn_0 = churn_dataset[
        churn_dataset["Churn"] == 0
    ]

    churn_1 = churn_dataset[
        churn_dataset["Churn"] == 1
    ]

    comparison = pd.DataFrame({
        "Mean_Non_Churn": (
            churn_0[predictor_data.columns]
            .mean()
        ),

        "Mean_Churn": (
            churn_1[predictor_data.columns]
            .mean()
        ),

        "Median_Non_Churn": (
            churn_0[predictor_data.columns]
            .median()
        ),

        "Median_Churn": (
            churn_1[predictor_data.columns]
            .median()
        ),
    })

    comparison["Mean_Difference"] = (
        comparison["Mean_Churn"]
        - comparison["Mean_Non_Churn"]
    )

    comparison["Median_Difference"] = (
        comparison["Median_Churn"]
        - comparison["Median_Non_Churn"]
    )

    comparison["Absolute_Mean_Difference"] = (
        comparison["Mean_Difference"]
        .abs()
    )

    comparison = (
        comparison
        .sort_values(
            "Absolute_Mean_Difference",
            ascending=False,
        )
        .round(4)
    )

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------

    top_features = (
        comparison
        .head(15)
        .sort_values(
            "Mean_Difference"
        )
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        top_features.index,
        top_features["Mean_Difference"],
    )

    plt.axvline(
        0,
        linewidth=1,
    )

    plt.title(
        "Top 15 Feature Mean Differences by Churn"
    )

    plt.xlabel(
        "Mean Difference (Churn − Non-Churn)"
    )

    plt.ylabel(
        "Feature"
    )

    plt.tight_layout()
    plt.show()

    return comparison


#------- 5. crrelation with churn -------
def analyze_feature_churn_correlation(
    churn_dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze the correlation between numerical predictor features
    and the Churn target.

    Customer ID is excluded because it is an identifier.

    Parameters
    ----------
    churn_dataset : pd.DataFrame
        Customer-level churn dataset.

    Returns
    -------
    pd.DataFrame
        Features ranked by absolute correlation with Churn.
    """

    if not isinstance(churn_dataset, pd.DataFrame):
        raise TypeError(
            "churn_dataset must be a pandas DataFrame."
        )

    if churn_dataset.empty:
        raise ValueError(
            "churn_dataset is empty."
        )

    required_columns = {
        "Customer ID",
        "Churn",
    }

    missing_columns = (
        required_columns
        - set(churn_dataset.columns)
    )

    if missing_columns:
        raise ValueError(
            "churn_dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

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

    predictor_data = (
        churn_dataset
        .drop(
            columns=["Customer ID", "Churn"]
        )
        .select_dtypes(include="number")
    )

    if predictor_data.empty:
        raise ValueError(
            "No numerical predictor features found."
        )

    correlation = (
        predictor_data
        .corrwith(
            churn_dataset["Churn"]
        )
        .rename("Correlation_With_Churn")
        .to_frame()
    )

    correlation["Absolute_Correlation"] = (
        correlation["Correlation_With_Churn"]
        .abs()
    )

    correlation["Relationship"] = (
        correlation["Correlation_With_Churn"]
        .apply(
            lambda value:
                "Positive"
                if value > 0
                else "Negative"
                if value < 0
                else "None"
        )
    )

    correlation = (
        correlation
        .sort_values(
            "Absolute_Correlation",
            ascending=False,
        )
        .round(4)
    )

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------

    top_features = (
        correlation
        .head(15)
        .sort_values(
            "Correlation_With_Churn"
        )
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        top_features.index,
        top_features[
            "Correlation_With_Churn"
        ],
    )

    plt.axvline(
        0,
        linewidth=1,
    )

    plt.title(
        "Top 15 Feature Correlations with Churn"
    )

    plt.xlabel(
        "Pearson Correlation with Churn"
    )

    plt.ylabel(
        "Feature"
    )

    plt.tight_layout()
    plt.show()

    return correlation


#------- 6. feature to feature correlation ------
def analyze_feature_correlations(
    churn_dataset: pd.DataFrame,
    threshold: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze correlations between numerical predictor features.

    Customer ID and Churn are excluded because Customer ID is an
    identifier and Churn is the target variable.

    Parameters
    ----------
    churn_dataset : pd.DataFrame
        Customer-level churn dataset.

    threshold : float, default=0.80
        Absolute correlation threshold used to identify highly
        correlated feature pairs.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        correlation_matrix:
            Full feature-to-feature Pearson correlation matrix.

        high_correlation_pairs:
            Ranked feature pairs whose absolute correlation is
            greater than or equal to the threshold.
    """

    if not isinstance(churn_dataset, pd.DataFrame):
        raise TypeError(
            "churn_dataset must be a pandas DataFrame."
        )

    if churn_dataset.empty:
        raise ValueError(
            "churn_dataset is empty."
        )

    required_columns = {
        "Customer ID",
        "Churn",
    }

    missing_columns = (
        required_columns
        - set(churn_dataset.columns)
    )

    if missing_columns:
        raise ValueError(
            "churn_dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if not 0 < threshold < 1:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    predictor_data = (
        churn_dataset
        .drop(
            columns=["Customer ID", "Churn"]
        )
        .select_dtypes(include="number")
    )

    if predictor_data.empty:
        raise ValueError(
            "No numerical predictor features found."
        )

    # --------------------------------------------------
    # Full correlation matrix
    # --------------------------------------------------

    correlation_matrix = (
        predictor_data
        .corr(method="pearson")
    )

    # --------------------------------------------------
    # Ranked highly correlated feature pairs
    # --------------------------------------------------

    upper_triangle = (
        correlation_matrix
        .where(
            np.triu(
                np.ones(
                    correlation_matrix.shape,
                    dtype=bool,
                ),
                k=1,
            )
        )
    )

    correlation_pairs = (
        upper_triangle
        .stack()
        .reset_index()
    )

    correlation_pairs.columns = [
        "Feature_1",
        "Feature_2",
        "Correlation",
    ]

    correlation_pairs[
        "Absolute_Correlation"
    ] = (
        correlation_pairs["Correlation"]
        .abs()
    )

    high_correlation_pairs = (
        correlation_pairs[
            correlation_pairs[
                "Absolute_Correlation"
            ] >= threshold
        ]
        .sort_values(
            "Absolute_Correlation",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    high_correlation_pairs = (
        high_correlation_pairs
        .round(4)
    )

    # --------------------------------------------------
    # Heatmap
    # --------------------------------------------------

    plt.figure(
        figsize=(18, 14)
    )

    sns.heatmap(
        correlation_matrix,
        cmap="coolwarm",
        center=0,
        linewidths=0.3,
        square=True,
    )

    plt.title(
        "Feature-to-Feature Correlation Matrix"
    )

    plt.tight_layout()
    plt.show()

    return (
        correlation_matrix.round(4),
        high_correlation_pairs,
    )


#-------Step 7 — VIF Analysis----------------------------------------------------------------
def analyze_vif(
    churn_dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze multicollinearity among numerical predictor features
    using Variance Inflation Factor (VIF).

    Customer ID and Churn are excluded.

    Parameters
    ----------
    churn_dataset : pd.DataFrame
        Customer-level churn dataset.

    Returns
    -------
    pd.DataFrame
        VIF values ranked from highest to lowest.
    """

    if not isinstance(churn_dataset, pd.DataFrame):
        raise TypeError(
            "churn_dataset must be a pandas DataFrame."
        )

    if churn_dataset.empty:
        raise ValueError(
            "churn_dataset is empty."
        )

    required_columns = {
        "Customer ID",
        "Churn",
    }

    missing_columns = (
        required_columns
        - set(churn_dataset.columns)
    )

    if missing_columns:
        raise ValueError(
            "churn_dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    predictor_data = (
        churn_dataset
        .drop(
            columns=["Customer ID", "Churn"]
        )
        .select_dtypes(include="number")
        .copy()
    )

    if predictor_data.empty:
        raise ValueError(
            "No numerical predictor features found."
        )

    if predictor_data.isna().any().any():
        raise ValueError(
            "Predictor data contains missing values. "
            "VIF requires complete predictor data."
        )

    # Remove constant features because their VIF is undefined.
    constant_features = [
        column
        for column in predictor_data.columns
        if predictor_data[column].nunique() <= 1
    ]

    if constant_features:
        predictor_data = predictor_data.drop(
            columns=constant_features
        )

    if predictor_data.empty:
        raise ValueError(
            "No variable numerical predictors remain "
            "after removing constant features."
        )

    vif_results = []

    for column in predictor_data.columns:

        other_features = predictor_data.drop(
            columns=[column]
        )

        if other_features.empty:
            vif = 1.0
        else:
            r_squared = (
                LinearRegression()
                .fit(
                    other_features,
                    predictor_data[column],
                )
                .score(
                    other_features,
                    predictor_data[column],
                )
            )

            if r_squared >= 1.0:
                vif = np.inf
            else:
                vif = 1 / (1 - r_squared)

        vif_results.append({
            "Feature": column,
            "VIF": vif,
        })

    vif_summary = pd.DataFrame(
        vif_results
    )

    vif_summary["VIF"] = (
        vif_summary["VIF"]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    vif_summary = (
        vif_summary
        .sort_values(
            "VIF",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # VIF interpretation
    # --------------------------------------------------

    def classify_vif(value):
        if pd.isna(value):
            return "Infinite / Undefined"
        elif value < 5:
            return "Low"
        elif value < 10:
            return "Moderate"
        else:
            return "High"

    vif_summary["VIF_Level"] = (
        vif_summary["VIF"]
        .apply(classify_vif)
    )

    vif_summary["VIF"] = (
        vif_summary["VIF"]
        .round(4)
    )

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------

    top_vif = (
        vif_summary
        .head(15)
        .sort_values(
            "VIF"
        )
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        top_vif["Feature"],
        top_vif["VIF"],
    )

    plt.axvline(
        5,
        linewidth=1,
        linestyle="--",
    )

    plt.axvline(
        10,
        linewidth=1,
        linestyle="--",
    )

    plt.title(
        "Top 15 Features by Variance Inflation Factor"
    )

    plt.xlabel(
        "Variance Inflation Factor (VIF)"
    )

    plt.ylabel(
        "Feature"
    )

    plt.tight_layout()
    plt.show()

    return vif_summary

#-------Step 8 — Important Features & Initial Churn Hypotheses
def analyze_churn_feature_importance(
    churn_dataset: pd.DataFrame,
    churn_correlation: pd.DataFrame,
    churn_feature_comparison: pd.DataFrame,
    vif_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Consolidate important churn features using correlation,
    mean difference, and VIF information.

    This function is intended for EDA interpretation and
    hypothesis generation. It does not perform model-based
    feature selection.
    """

    if not isinstance(churn_dataset, pd.DataFrame):
        raise TypeError(
            "churn_dataset must be a pandas DataFrame."
        )

    if churn_dataset.empty:
        raise ValueError(
            "churn_dataset is empty."
        )

    if not isinstance(
        churn_correlation,
        pd.DataFrame,
    ):
        raise TypeError(
            "churn_correlation must be a pandas DataFrame."
        )

    if not isinstance(
        churn_feature_comparison,
        pd.DataFrame,
    ):
        raise TypeError(
            "churn_feature_comparison must be a pandas DataFrame."
        )

    if not isinstance(
        vif_results,
        pd.DataFrame,
    ):
        raise TypeError(
            "vif_results must be a pandas DataFrame."
        )

    # --------------------------------------------------
    # Correlation data
    # --------------------------------------------------

    required_correlation_columns = {
        "Correlation_With_Churn",
        "Absolute_Correlation",
    }

    if not required_correlation_columns.issubset(
        churn_correlation.columns
    ):
        raise ValueError(
            "churn_correlation does not contain the "
            "required correlation columns."
        )

    correlation_data = (
        churn_correlation[
            [
                "Correlation_With_Churn",
                "Absolute_Correlation",
            ]
        ]
        .copy()
    )

    correlation_data.index.name = "Feature"

    correlation_data = (
        correlation_data
        .reset_index()
    )

    # --------------------------------------------------
    # Feature comparison data
    # --------------------------------------------------

    required_comparison_columns = {
        "Mean_Difference",
        "Absolute_Mean_Difference",
    }

    if not required_comparison_columns.issubset(
        churn_feature_comparison.columns
    ):
        raise ValueError(
            "churn_feature_comparison does not contain "
            "the required comparison columns."
        )

    comparison_data = (
        churn_feature_comparison[
            [
                "Mean_Difference",
                "Absolute_Mean_Difference",
            ]
        ]
        .copy()
    )

    comparison_data.index.name = "Feature"

    comparison_data = (
        comparison_data
        .reset_index()
    )

    # --------------------------------------------------
    # VIF data
    # --------------------------------------------------

    required_vif_columns = {
        "Feature",
        "VIF",
        "VIF_Level",
    }

    if not required_vif_columns.issubset(
        vif_results.columns
    ):
        raise ValueError(
            "vif_results does not contain the "
            "required VIF columns."
        )

    vif_data = (
        vif_results[
            [
                "Feature",
                "VIF",
                "VIF_Level",
            ]
        ]
        .copy()
    )

    # --------------------------------------------------
    # Combine analysis
    # --------------------------------------------------

    analysis = (
        correlation_data
        .merge(
            comparison_data,
            on="Feature",
            how="left",
        )
        .merge(
            vif_data,
            on="Feature",
            how="left",
        )
    )

    # --------------------------------------------------
    # Priority based on absolute correlation
    # --------------------------------------------------

    analysis["Priority"] = (
        analysis["Absolute_Correlation"]
        .rank(
            ascending=False,
            method="min",
        )
    )

    analysis = (
        analysis
        .sort_values(
            "Absolute_Correlation",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return analysis
