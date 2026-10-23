import numpy as np
import pandas as pd
from IPython.display import display

# ---- 1. Quality check and audit dataset  ----------------
def audit_segmentation_features(
    df,
    correlation_threshold=0.99,
    display_report=True
):
    """
    Audit feature quality and identify potentially redundant
    numerical features. Does not drop any columns.

    Returns
    -------
    dict
        Quality report, correlation matrix, highly correlated
        feature pairs, and exact duplicate feature pairs.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    numeric_df = df.select_dtypes(include="number")

    # 1. Feature quality report
    quality_report = pd.DataFrame(index=df.columns)

    quality_report["Data Type"] = df.dtypes.astype(str)
    quality_report["Missing Values"] = df.isna().sum()
    quality_report["Missing (%)"] = (
        df.isna().mean() * 100
    ).round(2)
    quality_report["Unique Values"] = df.nunique()

    quality_report["Infinite Values"] = 0
    quality_report["Standard Deviation"] = np.nan

    if not numeric_df.empty:
        quality_report.loc[
            numeric_df.columns, "Infinite Values"
        ] = np.isinf(numeric_df).sum()

        quality_report.loc[
            numeric_df.columns, "Standard Deviation"
        ] = numeric_df.std()

    quality_report["Zero Variance"] = (
        quality_report["Standard Deviation"].eq(0)
        & quality_report.index.isin(numeric_df.columns)
    )

    quality_report.index.name = "Feature"

    # 2. Correlation matrix
    correlation_matrix = numeric_df.corr()

    # 3. Highly correlated feature pairs
    correlated_pairs = []

    columns = numeric_df.columns.tolist()

    for i, col1 in enumerate(columns):
        for col2 in columns[i + 1:]:
            correlation = correlation_matrix.loc[col1, col2]

            if (
                pd.notna(correlation)
                and abs(correlation) >= correlation_threshold
            ):
                difference = (
                    numeric_df[col1] - numeric_df[col2]
                ).abs()

                correlated_pairs.append({
                    "Feature 1": col1,
                    "Feature 2": col2,
                    "Correlation": correlation,
                    "Exact Match": numeric_df[col1].equals(
                        numeric_df[col2]
                    ),
                    "Rows Different": int(
                        (difference > 1e-9).sum()
                    ),
                    "Maximum Absolute Difference": difference.max()
                })

    correlation_report = pd.DataFrame(
        correlated_pairs,
        columns=[
            "Feature 1",
            "Feature 2",
            "Correlation",
            "Exact Match",
            "Rows Different",
            "Maximum Absolute Difference"
        ]
    )

    if not correlation_report.empty:
        correlation_report["Correlation"] = (
            correlation_report["Correlation"].round(6)
        )
        correlation_report[
            "Maximum Absolute Difference"
        ] = correlation_report[
            "Maximum Absolute Difference"
        ].round(6)

        correlation_report = correlation_report.sort_values(
            "Correlation",
            key=lambda values: values.abs(),
            ascending=False
        ).reset_index(drop=True)

    # 4. Summary
    summary = pd.DataFrame({
        "Metric": [
            "Rows",
            "Columns",
            "Numeric Features",
            "Non-Numeric Features",
            "Missing Cells",
            "Infinite Values",
            "Zero-Variance Numeric Features",
            "Highly Correlated Feature Pairs"
        ],
        "Value": [
            len(df),
            len(df.columns),
            len(numeric_df.columns),
            len(df.columns) - len(numeric_df.columns),
            int(df.isna().sum().sum()),
            int(np.isinf(numeric_df).sum().sum()),
            int(quality_report["Zero Variance"].sum()),
            len(correlation_report)
        ]
    })

    # 5. Notebook-friendly display
    if display_report:
        print("=" * 65)
        print("CUSTOMER SEGMENTATION — FEATURE AUDIT")
        print("=" * 65)

        display(summary)

        print("\nFEATURE QUALITY REPORT")
        display(
            quality_report.sort_values(
                ["Missing Values", "Infinite Values"],
                ascending=False
            )
        )

        print("\nCORRELATION MATRIX")
        display(correlation_matrix.round(3))

        print("\nHIGHLY CORRELATED FEATURE PAIRS")
        if correlation_report.empty:
            print("No pairs found at the selected threshold.")
        else:
            display(correlation_report)

    # Nothing is removed from df
    return {
        "summary": summary,
        "quality_report": quality_report,
        "correlation_matrix": correlation_matrix,
        "correlation_report": correlation_report
    }


#------ 2. Feature selection ----------------------------------------------------
def select_segmentation_features(
    df,
    exclude_columns=None,
    display_report=True
):
    """
    Exclude explicitly specified columns from a DataFrame.

    Does not automatically remove correlated, constant,
    or otherwise questionable features.

    Returns
    -------
    pandas.DataFrame
        A separate feature matrix.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    exclude_columns = list(exclude_columns or [])

    missing_columns = [
        col for col in exclude_columns
        if col not in df.columns
    ]

    valid_exclusions = [
        col for col in exclude_columns
        if col in df.columns
    ]

    X = df.drop(columns=valid_exclusions).copy()

    if display_report:
        selection_report = pd.DataFrame({
            "Metric": [
                "Original Rows",
                "Original Columns",
                "Columns Requested for Exclusion",
                "Columns Actually Excluded",
                "Columns Not Found",
                "Selected Features"
            ],
            "Value": [
                len(df),
                len(df.columns),
                len(exclude_columns),
                len(valid_exclusions),
                len(missing_columns),
                X.shape[1]
            ]
        })

        print("=" * 65)
        print("CUSTOMER SEGMENTATION — FEATURE SELECTION")
        print("=" * 65)

        display(selection_report)

        print("\nExcluded columns:")
        display(pd.DataFrame({
            "Column": valid_exclusions
        }))

        if missing_columns:
            print("Warning — requested columns not found:")
            print(missing_columns)

        print("\nSelected feature matrix shape:", X.shape)

    return X



#--------3. Feature scaling function -----------------
from sklearn.preprocessing import StandardScaler
import pandas as pd


def scale_segmentation_features(X, scaler=None):
    """
    Apply Z-score scaling to customer segmentation features.

    Parameters
    ----------
    X : pandas.DataFrame
        Selected numerical features.
    scaler : StandardScaler, optional
        Existing fitted scaler. If omitted, a new scaler is fitted.

    Returns
    -------
    dict
        Scaled feature DataFrame and fitted scaler.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if X.empty:
        raise ValueError("X must not be empty.")

    if not X.select_dtypes(include="number").columns.equals(X.columns):
        raise ValueError("All input features must be numerical.")

    if X.isna().any().any():
        raise ValueError("X contains missing values.")

    import numpy as np

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("X contains infinite or non-finite values.")

    if scaler is None:
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(X)
    else:
        if not hasattr(scaler, "feature_names_in_"):
            raise ValueError(
                "The supplied scaler must be fitted using named features."
            )

        if list(X.columns) != list(scaler.feature_names_in_):
            raise ValueError(
                "Input feature names or order do not match the fitted scaler."
            )

        scaled_values = scaler.transform(X)

    X_scaled = pd.DataFrame(
        scaled_values,
        columns=X.columns,
        index=X.index
    )

    return {
        "X_scaled": X_scaled,
        "scaler": scaler
    }
