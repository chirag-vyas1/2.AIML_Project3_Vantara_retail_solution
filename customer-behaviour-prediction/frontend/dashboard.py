import os

API_BASE_URL = os.getenv("API_BASE_URL","http://127.0.0.1:8000",)

import requests
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt


from src.models.model_persistance import get_project_root
from src.models.model_persistance import load_model_artifacts
from src.utils.config_loader import load_config
from src.explainability.shap_explainer import explain_customer_for_dashboard
from src.explainability.shap_explainer import explain_global_for_dashboard


# ---------------------------------------------------------
# Vantara Retail Customer Behaviour Prediction
# Day 17: Simple Streamlit Dashboard
#
# Run from the project root:
# python -m streamlit run front-end/dashboard.py
# ---------------------------------------------------------

st.set_page_config(page_title="Vantara Customer Analytics",page_icon="📊",layout="wide",)

def show_home():
    """Display the dashboard introduction."""

    st.title("Vantara Customer Analytics")
    st.write(
        "Understand customer behaviour, identify churn risk, "
        "and analyse retail sales."
    )

    st.divider()

    st.header("Project Overview")

    st.write(
        """
        This project uses the Online Retail II dataset to understand
        customer purchasing behaviour.

        The objective is to convert transaction data into useful
        business insights using data analysis and machine learning.
        """
    )

    st.header("What can we analyse?")

    st.subheader("1. Customer Segmentation")
    st.write(
        """
        Group customers with similar purchasing behaviour using
        Gaussian Mixture Model (GMM).

        Business question: Which types of customers do we have,
        and how do their purchasing patterns differ?
        """
    )

    st.subheader("2. Customer Churn Prediction")
    st.write(
        """
        Estimate the probability that a customer belongs to the
        churn class defined by our trained model.

        Business question: Which customers may need retention
        attention?
        """
    )

    st.subheader("3. Revenue Analysis")
    st.write(
        """
        Analyse historical transaction values over time.

        Business question: How have sales changed, and are there
        patterns that may help us plan ahead?
        """
    )

    st.divider()

    st.header("Project Workflow")

    st.write("1. Clean and validate transaction data.")
    st.write("2. Create customer-level features.")
    st.write("3. Segment customers and predict churn.")
    st.write("4. Analyse results and support business decisions.")

    st.info(
        "Select a page from the sidebar to explore the dashboard. "
        "The analysis pages will be implemented incrementally "
        "using verified project data and model artifacts."
    )


from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


def show_customer_segmentation():
    from pathlib import Path

    import joblib
    import numpy as np
    import pandas as pd
    import streamlit as st

    # 1. Define project paths
    project_root = Path(__file__).resolve().parents[1]

    customer_features_path = (
        project_root
        / "data"
        / "processed"
        / "customer_features.csv"
    )

    gmm_artifact_path = (
        project_root
        / "model_artifacts"
        / "models"
        / "clustering"
        / "gmm_customer_segmentation"
        / "gmm_customer_segmentation_artifact.joblib"
    )

    st.title("Customer Segmentation")
    st.write(
        "Explore customer groups identified by the "
        "trained Gaussian Mixture Model (GMM)."
    )

    # 2. Validate required files
    if not customer_features_path.is_file():
        st.error("Customer features dataset was not found.")
        return

    if not gmm_artifact_path.is_file():
        st.error("Saved GMM model artifact was not found.")
        return

    # 3. Load customer data and GMM artifact
    try:
        customer_features = pd.read_csv(customer_features_path)
        artifact = joblib.load(gmm_artifact_path)

        gmm_model = artifact["model"]
        preprocessing = artifact["preprocessing"]

        scaler = preprocessing["scaler"]
        feature_names = preprocessing["feature_names"]

    except Exception as exc:
        st.error(f"Unable to load data or GMM artifact: {exc}")
        return

    # 4. Validate required columns
    if "Customer ID" not in customer_features.columns:
        st.error("The dataset is missing the Customer ID column.")
        return

    missing_columns = [
        column
        for column in feature_names
        if column not in customer_features.columns
    ]

    if missing_columns:
        st.error(f"Missing required GMM features: {missing_columns}")
        return

    # 5. Select features in the original training order
    X_customer = customer_features.loc[:, feature_names].copy()

    try:
        X_customer = X_customer.apply(
            pd.to_numeric,
            errors="raise"
        )

        if not np.isfinite(
            X_customer.to_numpy(dtype=float)
        ).all():
            st.error(
                "GMM features contain missing or infinite values. "
                "Apply the same preprocessing used during training."
            )
            return

        # 6. Scale and predict in memory
        X_scaled = scaler.transform(X_customer)

        customer_features["GMM_Cluster"] = (
            gmm_model.predict(X_scaled)
        )

    except Exception as exc:
        st.error(f"Unable to predict customer segments: {exc}")
        return

    # 7. Segment filter
    available_segments = sorted(
        customer_features["GMM_Cluster"].unique().tolist()
    )

    selected_segments = st.sidebar.multiselect(
        "Select GMM Segments",
        options=available_segments,
        default=available_segments,
        format_func=lambda segment: f"Segment {segment}"
    )

    filtered = customer_features[
        customer_features["GMM_Cluster"].isin(selected_segments)
    ].copy()

    if filtered.empty:
        st.info("Select at least one segment to display customers.")
        return

    # 8. Overview metrics
    st.subheader("Customer Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Customers",
        f"{filtered['Customer ID'].nunique():,}"
    )

    if "Historical_CLV" in filtered.columns:
        median_clv = filtered["Historical_CLV"].median()
        col2.metric(
            "Median Historical CLV",
            f"{median_clv:,.2f}" if pd.notna(median_clv) else "N/A"
        )
    else:
        col2.metric("Median Historical CLV", "N/A")

    if "Frequency" in filtered.columns:
        median_frequency = filtered["Frequency"].median()
        col3.metric(
            "Median Purchase Frequency",
            (
                f"{median_frequency:,.1f}"
                if pd.notna(median_frequency)
                else "N/A"
            )
        )
    else:
        col3.metric("Median Purchase Frequency", "N/A")

    # 9. Segment distribution
    st.subheader("Segment Distribution")

    segment_counts = (
        filtered.groupby("GMM_Cluster")["Customer ID"]
        .nunique()
        .sort_index()
    )

    segment_percentages = (
        segment_counts / segment_counts.sum() * 100
    ).round(2)

    col1, col2 = st.columns(2)

    with col1:
        st.write("Customers per Segment")
        st.bar_chart(segment_counts)

    with col2:
        st.write("Customer Percentage by Segment")
        st.bar_chart(segment_percentages)

    st.dataframe(
        pd.DataFrame({
            "Customer Count": segment_counts,
            "Percentage (%)": segment_percentages,
        }),
        use_container_width=True
    )

    # 10. Segment comparison
    st.subheader("Segment Comparison")

    comparison_candidates = [
        "Historical_CLV",
        "Frequency",
        "Recency",
        "Average_Order_Spend",
        "Engagement_Score",
        "Return_Rate",
    ]

    comparison_metrics = [
        column
        for column in comparison_candidates
        if column in filtered.columns
    ]

    if comparison_metrics:
        selected_metric = st.selectbox(
            "Choose a metric to compare",
            options=comparison_metrics
        )

        metric_comparison = (
            filtered.groupby("GMM_Cluster")[selected_metric]
            .median()
            .sort_index()
        )

        st.write(f"Median {selected_metric} by Segment")
        st.bar_chart(metric_comparison)
    else:
        st.info("No supported comparison metrics are available.")

    # 11. Customer details
    st.subheader("Customer Details")

    display_columns = [
        "Customer ID",
        "GMM_Cluster",
        "Historical_CLV",
        "Frequency",
        "Recency",
        "Average_Order_Spend",
        "Engagement_Score",
        "Return_Rate",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in filtered.columns
    ]

    st.dataframe(
        filtered[display_columns],
        use_container_width=True,
        hide_index=True
    )

def show_churn_leaderboard():
    import numpy as np
    import pandas as pd
    import requests
    import streamlit as st
    from pathlib import Path

    from src.models.model_persistance import (
        get_project_root,
        load_model_artifacts,
    )
    from src.utils.config_loader import load_config

    # --------------------------------------------------
    # 1. Define paths and API settings
    # --------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]

    dataset_path = (
        project_root
        / "data"
        / "processed"
        / "customer_churn_dataset.csv"
    )

    #api_url = "http://127.0.0.1:8000/predict/batch"
    api_url = f"{API_BASE_URL}/predict/batch"
    batch_size = 250

    st.title("Churn Analysis")
    st.write(
        "Identify customers with the highest predicted "
        "churn probability using the trained XGBoost model."
    )

    # --------------------------------------------------
    # 2. Load customer dataset
    # --------------------------------------------------
    if not dataset_path.is_file():
        st.error("Customer churn dataset was not found.")
        return

    try:
        customers = pd.read_csv(dataset_path)
    except Exception as exc:
        st.error(f"Unable to load customer dataset: {exc}")
        return

    if "Customer ID" not in customers.columns:
        st.error("Customer ID column is missing.")
        return

    if customers["Customer ID"].isna().any():
        st.error("Customer IDs contain missing values.")
        return

    if customers["Customer ID"].duplicated().any():
        st.error(
            "Customer IDs are duplicated. "
            "Please verify the customer dataset."
        )
        return

    # --------------------------------------------------
    # 3. Load feature names from the saved XGBoost artifact
    # --------------------------------------------------
    try:
        config = load_config()

        model_bundle = load_model_artifacts(
            config=config,
            get_project_root=get_project_root,
            model_key="xgboost",
            model_type="classification",
        )

        feature_names = model_bundle["feature_names"]

        if not feature_names:
            st.error(
                "No feature names were found in the "
                "saved XGBoost artifact."
            )
            return

    except Exception as exc:
        st.error(
            f"Unable to load XGBoost artifact metadata: {exc}"
        )
        return

    # --------------------------------------------------
    # 4. Validate dataset against model features
    # --------------------------------------------------
    missing_columns = [
        column
        for column in feature_names
        if column not in customers.columns
    ]

    if missing_columns:
        st.error(
            "The customer dataset is missing model features: "
            f"{missing_columns}"
        )
        return

    # Preserve the exact feature order expected by the model
    X = customers.loc[:, feature_names].copy()

    try:
        X = X.apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        st.error(f"Model features must be numeric: {exc}")
        return

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        st.error(
            "Model inputs contain missing or infinite values. "
            "Please check the processed dataset."
        )
        return

    # Match the API's StrictInt requirement
    if "Frequency" in X.columns:
        frequency = X["Frequency"].to_numpy(dtype=float)

        if not np.equal(
            frequency,
            np.floor(frequency)
        ).all():
            st.error(
                "Frequency contains non-integer values, "
                "but the API requires integers."
            )
            return

        X["Frequency"] = X["Frequency"].astype(int)

    # --------------------------------------------------
    # 5. Initialize session state
    # --------------------------------------------------
    if "churn_leaderboard" not in st.session_state:
        st.session_state["churn_leaderboard"] = None

    if "churn_dataset_signature" not in st.session_state:
        st.session_state["churn_dataset_signature"] = None

    # Invalidate old results if the dataset has changed
    dataset_signature = (
        str(dataset_path.resolve()),
        dataset_path.stat().st_mtime_ns,
        len(customers),
    )

    if (
        st.session_state["churn_dataset_signature"]
        != dataset_signature
    ):
        st.session_state["churn_leaderboard"] = None
        st.session_state["churn_dataset_signature"] = (
            dataset_signature
        )

    # --------------------------------------------------
    # 6. Generate predictions only when requested
    # --------------------------------------------------
    if st.button(
        "Run Churn Analysis",
        key="run_churn_analysis",
    ):
        predictions = []
        progress = st.progress(0)
        status = st.empty()

        try:
            with requests.Session() as session:
                total_batches = (
                    len(X) + batch_size - 1
                ) // batch_size

                for batch_number, start in enumerate(
                    range(0, len(X), batch_size),
                    start=1,
                ):
                    batch = X.iloc[
                        start:start + batch_size
                    ]

                    response = session.post(
                        api_url,
                        json=batch.to_dict(orient="records"),
                        timeout=120,
                    )

                    response.raise_for_status()
                    result = response.json()

                    batch_predictions = result["predictions"]

                    if len(batch_predictions) != len(batch):
                        raise ValueError(
                            "Prediction count does not match "
                            "the submitted batch size."
                        )

                    predictions.extend(batch_predictions)

                    progress.progress(
                        batch_number / total_batches
                    )
                    status.write(
                        f"Processing batch {batch_number} "
                        f"of {total_batches}..."
                    )

            if len(predictions) != len(customers):
                raise ValueError(
                    "Total prediction count does not match "
                    "the number of customers."
                )

            # Validate API response
            prediction_df = pd.DataFrame(predictions)

            required_response_columns = [
                "predicted_class",
                "churn_probability",
            ]

            if not all(
                column in prediction_df.columns
                for column in required_response_columns
            ):
                raise ValueError(
                    "The API returned an unexpected response."
                )

            probabilities = prediction_df[
                "churn_probability"
            ].to_numpy(dtype=float)

            if (
                not np.isfinite(probabilities).all()
                or not np.logical_and(
                    probabilities >= 0,
                    probabilities <= 1,
                ).all()
            ):
                raise ValueError(
                    "The API returned invalid churn probabilities."
                )

            predicted_classes = prediction_df[
                "predicted_class"
            ]

            if not predicted_classes.isin([0, 1]).all():
                raise ValueError(
                    "The API returned an unexpected prediction class."
                )

            # Attach predictions to original customer records
            leaderboard = customers.copy()

            leaderboard["Predicted_Churn"] = (
                predicted_classes.to_numpy()
            )
            leaderboard["Churn_Probability"] = probabilities

            # Highest predicted churn probability first
            leaderboard = leaderboard.sort_values(
                "Churn_Probability",
                ascending=False,
            ).reset_index(drop=True)

            # Preserve predictions across Streamlit reruns
            st.session_state["churn_leaderboard"] = leaderboard

        except requests.RequestException as exc:
            st.error(
                f"Could not connect to FastAPI at {API_BASE_URL}."
                f"Make sure the API is running. Details: {exc}"
            )
            return

        except (KeyError, ValueError, TypeError) as exc:
            st.error(
                f"Unable to process churn predictions: {exc}"
            )
            return

        finally:
            progress.empty()
            status.empty()

    # --------------------------------------------------
    # 7. Retrieve saved predictions
    # --------------------------------------------------
    leaderboard = st.session_state["churn_leaderboard"]

    if leaderboard is None:
        st.info(
            "Click 'Run Churn Analysis' to generate predictions."
        )
        return

    # --------------------------------------------------
    # 8. Summary metrics
    # --------------------------------------------------
    st.success(
        f"Predictions generated for {len(leaderboard):,} customers."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Customers Analysed",
        f"{len(leaderboard):,}",
    )

    col2.metric(
        "Predicted Churn = 1",
        f"{(leaderboard['Predicted_Churn'] == 1).sum():,}",
    )

    col3.metric(
        "High Predicted Risk (>= 70%)",
        f"{(leaderboard['Churn_Probability'] >= 0.70).sum():,}",
    )


    # --------------------------------------------------
    # 9. Churn-risk leaderboard and filters
    # --------------------------------------------------
    st.subheader("Customer Churn-Risk Leaderboard")

    st.caption(
        "Filter customers by predicted churn class and "
        "sort them by predicted churn probability."
    )

    # Filter by predicted churn class
    selected_classes = st.multiselect(
        "Filter by predicted churn class",
        options=[0, 1],
        default=[1],
        format_func=lambda value: (
            "Predicted Churn = 1"
            if value == 1
            else "Predicted Churn = 0"
        ),
        key="churn_class_filter",
    )

    filtered_leaderboard = leaderboard[
        leaderboard["Predicted_Churn"].isin(selected_classes)
    ].copy()

    if filtered_leaderboard.empty:
        st.info("No customers match the selected churn classes.")
        return

    # Allow the slider to display every matching customer
    max_rows = len(filtered_leaderboard)

    number_of_customers = st.slider(
        "Number of customers to display",
        min_value=1,
        max_value=max_rows,
        value=min(100, max_rows),
        key="churn_leaderboard_row_count",
    )

    display_columns = [
        "Customer ID",
        "Predicted_Churn",
        "Churn_Probability",
        "Recency",
        "Frequency",
        "Historical_CLV",
        "Engagement_Score",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in filtered_leaderboard.columns
    ]

    display_df = filtered_leaderboard[
        display_columns
    ].head(number_of_customers).copy()

    display_df["Churn_Probability"] = (
        display_df["Churn_Probability"] * 100
    ).round(2)

    display_df = display_df.rename(
        columns={
            "Predicted_Churn": "Predicted Churn Class",
            "Churn_Probability": "Churn Probability (%)",
        }
    )

    st.caption(
        f"Displaying {len(display_df):,} of "
        f"{len(filtered_leaderboard):,} matching customers."
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=600,
    )

    # Download all customers matching the selected class filter
    download_df = filtered_leaderboard[
        display_columns
    ].copy()

    download_df["Churn_Probability"] = (
        download_df["Churn_Probability"] * 100
    ).round(2)

    st.download_button(
        label="Download all matching customers as CSV",
        data=download_df.to_csv(index=False).encode("utf-8"),
        file_name="churn_risk_leaderboard.csv",
        mime="text/csv",
    )

def show_revenue_trends():
    from pathlib import Path

    import pandas as pd
    import streamlit as st

    # --------------------------------------------------
    # 1. Define project path
    # --------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]

    transactions_path = (
        project_root
        / "data"
        / "interim"
        / "cleaned_transactions.csv"
    )

    st.title("Revenue Analysis")
    st.write(
        "Explore historical revenue trends using the "
        "cleaned transaction dataset."
    )

    # --------------------------------------------------
    # 2. Validate and load data
    # --------------------------------------------------
    if not transactions_path.is_file():
        st.error("Cleaned transactions dataset was not found.")
        return

    try:
        transactions = pd.read_csv(transactions_path)
    except Exception as exc:
        st.error(f"Unable to load transactions: {exc}")
        return

    required_columns = [
        "InvoiceDate",
        "Analytical Transaction Value",
        "Include In Sales Analysis",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in transactions.columns
    ]

    if missing_columns:
        st.error(
            f"Required revenue columns are missing: "
            f"{missing_columns}"
        )
        return

    # --------------------------------------------------
    # 3. Prepare dates, inclusion flag and revenue
    # --------------------------------------------------
    transactions["InvoiceDate"] = pd.to_datetime(
        transactions["InvoiceDate"],
        errors="coerce",
    )

    transactions["Analytical Transaction Value"] = pd.to_numeric(
        transactions["Analytical Transaction Value"],
        errors="coerce",
    )

    include_flag = (
        transactions["Include In Sales Analysis"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    included_values = {"true", "1", "yes", "y"}

    sales = transactions.loc[
        include_flag.isin(included_values)
        & transactions["InvoiceDate"].notna()
        & transactions["Analytical Transaction Value"].notna()
    ].copy()

    if sales.empty:
        st.warning(
            "No eligible transactions were found for revenue analysis."
        )
        return

    # --------------------------------------------------
    # 4. Create monthly revenue series
    # --------------------------------------------------
    sales["Month"] = (
        sales["InvoiceDate"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    monthly_revenue = (
        sales.groupby("Month")["Analytical Transaction Value"]
        .sum()
        .sort_index()
    )

    total_revenue = sales["Analytical Transaction Value"].sum()
    transaction_count = sales["Invoice"].nunique()

    average_invoice_value = (
        sales.groupby("Invoice")["Analytical Transaction Value"]
        .sum()
        .mean()
    )

    # --------------------------------------------------
    # 5. Summary metrics
    # --------------------------------------------------
    st.subheader("Revenue Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Analytical Revenue",
        f"{total_revenue:,.2f}",
    )

    col2.metric(
        "Invoices Analysed",
        f"{transaction_count:,}",
    )

    col3.metric(
        "Average Revenue per Invoice",
        (
            f"{average_invoice_value:,.2f}"
            if pd.notna(average_invoice_value)
            else "N/A"
        ),
    )

    # --------------------------------------------------
    # 6. Monthly revenue trend
    # --------------------------------------------------
    st.subheader("Monthly Revenue Trend")

    st.line_chart(
        monthly_revenue,
        x_label="Month",
        y_label="Revenue",
    )

    # --------------------------------------------------
    # 7. Monthly revenue table
    # --------------------------------------------------
    st.subheader("Monthly Revenue Details")

    monthly_summary = monthly_revenue.rename(
        "Revenue"
    ).reset_index()

    monthly_summary["Month"] = (
        monthly_summary["Month"].dt.strftime("%Y-%m")
    )

    monthly_summary["Revenue"] = (
        monthly_summary["Revenue"].round(2)
    )

    st.dataframe(
        monthly_summary,
        use_container_width=True,
        hide_index=True,
    )


#================================================================
#--------- Shap and lime explainibility tables--------------
#================================================================

@st.cache_resource
def load_shap_dashboard_resources():
    """Load the saved XGBoost model and SHAP explainer once."""

    config = load_config()

    model_bundle = load_model_artifacts(
        config=config,
        get_project_root=get_project_root,
        model_key="xgboost",
        model_type="classification",
    )

    model = model_bundle["model"]
    feature_names = model_bundle["feature_names"]

    explainer = shap.TreeExplainer(model)

    return model, feature_names, explainer

@st.cache_data
def load_customer_churn_data():
    """Load the existing processed customer dataset."""

    project_root = Path(get_project_root()).resolve()

    data_path = (
        project_root
        / "data"
        / "processed"
        / "customer_churn_dataset.csv"
    )

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Customer churn dataset not found: {data_path}"
        )

    return pd.read_csv(data_path)

#---function with fetching model through API --------------------------------
def show_shap_explainability():
    """Display global and individual SHAP insights via FastAPI."""

    st.title("SHAP Explainability")
    st.caption(
        "Understand churn predictions through global feature "
        "importance and individual customer explanations."
    )

    project_root = Path(get_project_root()).resolve()
    data_path = (
        project_root
        / "data"
        / "processed"
        / "customer_churn_dataset.csv"
    )

    # ---------------------------------------------------------
    # LOAD CUSTOMER DATA
    # ---------------------------------------------------------
    try:
        customers = pd.read_csv(data_path)
    except Exception as exc:
        st.error(f"Unable to load customer data: {exc}")
        return

    if customers.empty:
        st.warning("The customer dataset is empty.")
        return

    # ---------------------------------------------------------
    # GET MODEL FEATURE NAMES THROUGH FASTAPI
    # ---------------------------------------------------------
    try:
        response = requests.get(
            f"{API_BASE_URL}/metadata",
            timeout=10,
        )
        response.raise_for_status()
        metadata = response.json()

        feature_names = metadata.get("feature_names")

        if not feature_names:
            st.error(
                "The API metadata response does not contain "
                "the expected feature_names field."
            )
            return

    except requests.RequestException as exc:
        st.error(
            "Cannot reach the FastAPI service. Make sure it is "
            f"running. Details: {exc}"
        )
        return

    missing_features = [
        feature
        for feature in feature_names
        if feature not in customers.columns
    ]

    if missing_features:
        st.error(
            "The customer dataset is missing required features: "
            + ", ".join(missing_features)
        )
        return

    # ---------------------------------------------------------
    # TABS
    # ---------------------------------------------------------
    global_tab, individual_tab = st.tabs([
        "Global customer insights",
        "Individual customer",
    ])

    # =========================================================
    # GLOBAL SHAP
    # =========================================================
    with global_tab:
        st.subheader("What influences churn across customers?")

        max_customers = min(500, len(customers))

        sample_size = st.slider(
            "Number of customers to analyse",
            min_value=1,
            max_value=max_customers,
            value=min(200, max_customers),
            key="global_shap_sample_size",
        )

        if st.button(
            "Generate global SHAP insights",
            key="generate_global_shap",
        ):
            X = customers[feature_names].sample(
                n=sample_size,
                random_state=42,
            )

            X = X.apply(pd.to_numeric, errors="coerce")

            if (
                X.isna().any().any()
                or not np.isfinite(
                    X.to_numpy(dtype=float)
                ).all()
            ):
                st.error(
                    "Selected customer features contain "
                    "missing or invalid values."
                )
            else:
                payload = {
                    "customers": X.to_dict(orient="records"),
                    "max_features": min(15, len(feature_names)),
                }

                try:
                    with st.spinner(
                        "Calculating global SHAP values through FastAPI..."
                    ):
                        response = requests.post(
                            f"{API_BASE_URL}/explain/shap/global",
                            json=payload,
                            timeout=120,
                        )
                        response.raise_for_status()
                        result = response.json()

                    required_fields = {
                        "importance",
                        "customers_analysed",
                        "feature_names",
                        "feature_values",
                        "shap_values",
                    }

                    missing_fields = (
                        required_fields - result.keys()
                    )

                    if missing_fields:
                        st.error(
                            "The global SHAP API response is missing: "
                            + ", ".join(sorted(missing_fields))
                        )
                    else:
                        # Store the complete API response so the
                        # beeswarm can be rendered from returned values.
                        st.session_state[
                            "global_shap_api_raw_result"
                        ] = result

                except requests.RequestException as exc:
                    st.error(
                        f"Global SHAP request failed: {exc}"
                    )

                    if exc.response is not None:
                        st.code(
                            exc.response.text,
                            language="json",
                        )

                except (ValueError, KeyError, TypeError) as exc:
                    st.error(
                        f"Invalid global SHAP response: {exc}"
                    )

        global_result = st.session_state.get(
            "global_shap_api_raw_result"
        )

        if global_result is not None:
            st.caption(
                "Customers analysed: "
                f"{global_result['customers_analysed']}"
            )

            st.markdown("#### Global feature importance")

            importance = pd.DataFrame(
                global_result["importance"]
            )

            st.dataframe(
                importance,
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### SHAP summary beeswarm")

            try:
                import shap
                import matplotlib.pyplot as plt

                shap_values = np.asarray(
                    global_result["shap_values"],
                    dtype=float,
                )
                feature_values = np.asarray(
                    global_result["feature_values"],
                    dtype=float,
                )
                returned_features = global_result[
                    "feature_names"
                ]

                if shap_values.shape != feature_values.shape:
                    st.error(
                        "SHAP values and feature values have "
                        "different shapes."
                    )
                elif shap_values.shape[1] != len(
                    returned_features
                ):
                    st.error(
                        "The returned SHAP feature count does "
                        "not match the feature names."
                    )
                else:
                    explanation = shap.Explanation(
                        values=shap_values,
                        data=feature_values,
                        feature_names=returned_features,
                    )

                    shap.plots.beeswarm(
                        explanation,
                        max_display=min(
                            15,
                            len(returned_features),
                        ),
                        show=False,
                    )

                    figure = plt.gcf()
                    st.pyplot(figure)
                    plt.close(figure)

            except Exception as exc:
                st.error(
                    f"Unable to render the SHAP beeswarm plot: {exc}"
                )

    # =========================================================
    # INDIVIDUAL CUSTOMER SHAP
    # =========================================================
    with individual_tab:
        st.subheader("Why might this customer churn?")

        if "Customer ID" not in customers.columns:
            st.error(
                "Customer ID is missing from the dataset."
            )
            return

        customer_ids = (
            customers["Customer ID"]
            .dropna()
            .drop_duplicates()
            .tolist()
        )

        if not customer_ids:
            st.warning(
                "No customer IDs are available."
            )
            return

        selected_id = st.selectbox(
            "Select a customer",
            customer_ids,
            key="individual_shap_customer_id",
        )

        if st.button(
            "Explain selected customer",
            key="generate_individual_shap",
        ):
            selected_rows = customers.loc[
                customers["Customer ID"] == selected_id
            ]

            if selected_rows.empty:
                st.error("Selected customer was not found.")
            else:
                # Preserve the API's exact model feature order.
                X_customer = selected_rows[
                    feature_names
                ].iloc[[0]].apply(
                    pd.to_numeric,
                    errors="coerce",
                )

                if (
                    X_customer.isna().any().any()
                    or not np.isfinite(
                        X_customer.to_numpy(dtype=float)
                    ).all()
                ):
                    st.error(
                        "This customer's features contain "
                        "missing or invalid values."
                    )
                else:
                    payload = X_customer.iloc[0].to_dict()
                    # Frequency is defined as a strict integer in the API schema.

                    if "Frequency" in payload:
                         frequency = payload["Frequency"]

                         if not float(frequency).is_integer():
                              st.error("Frequency must be a whole number for the API request.")
                              st.stop()

                         payload["Frequency"] = int(frequency)


                    try:
                        with st.spinner(
                            "Generating individual SHAP explanation "
                            "through FastAPI..."
                        ):
                            response = requests.post(
                                f"{API_BASE_URL}/explain/shap/customer",
                                json=payload,
                                timeout=120,
                            )
                            response.raise_for_status()
                            result = response.json()

                        st.session_state[
                            "individual_shap_api_result"
                        ] = result

                        st.session_state[
                            "individual_shap_api_customer"
                        ] = selected_id

                    except requests.RequestException as exc:
                        st.error(
                            f"Customer SHAP request failed: {exc}"
                        )

                        if exc.response is not None:
                            st.code(
                                exc.response.text,
                                language="json",
                            )

                    except (ValueError, KeyError, TypeError) as exc:
                        st.error(
                            f"Invalid individual SHAP response: {exc}"
                        )

        individual_result = st.session_state.get(
            "individual_shap_api_result"
        )

        result_customer_id = st.session_state.get(
            "individual_shap_api_customer"
        )

        if (
            individual_result is not None
            and result_customer_id == selected_id
        ):
            prediction_col, probability_col = st.columns(2)

            with prediction_col:
                st.metric(
                    "Predicted class",
                    individual_result["prediction_label"],
                )

            with probability_col:
                st.metric(
                    "Churn probability",
                    f"{individual_result['churn_probability']:.1%}",
                )

            contributions = pd.DataFrame(
                individual_result["contributions"]
            )

            st.markdown("#### Individual feature contributions")

            st.dataframe(
                contributions,
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### SHAP waterfall plot")

            required_fields = {"base_value", "feature_names", "feature_values", "shap_values",}

            if required_fields.issubset(individual_result):
                try:
                    import shap
                    import matplotlib.pyplot as plt

                    waterfall_explanation = shap.Explanation(
                         values=np.asarray(individual_result["shap_values"],dtype=float,),
                         base_values=float(individual_result["base_value"]),

                         data=np.asarray(individual_result["feature_values"], dtype=float,),
                         feature_names=individual_result["feature_names"],)

                    shap.plots.waterfall(
                            waterfall_explanation,
                            max_display=15,
                            show=False, )

                    figure = plt.gcf()
                    st.pyplot(figure)
                    plt.close(figure)


                except Exception as exc:
                  st.error(
                        f"Unable to render the SHAP waterfall plot: {exc}")


            st.markdown("#### SHAP contribution chart")

            if {
                "feature",
                "shap_value",
            }.issubset(contributions.columns):
                chart_data = contributions.set_index(
                    "feature"
                )[["shap_value"]]

                st.bar_chart(chart_data)





            st.info(
                "This chart shows individual feature contributions. "
                "A true SHAP waterfall plot also requires the SHAP "
                "base value and the customer's SHAP values from the API."
            )


def show_csv_batch_scoring():
    """Upload customer features, call FastAPI, and download predictions."""

    st.title("CSV Batch Scoring")
    st.write(
        "Upload a customer-features CSV to generate churn predictions "
        "and download a report."
    )

    uploaded_file = st.file_uploader(
        "Upload customer features CSV",
        type=["csv"],
        key="batch_scoring_csv",
    )

    if uploaded_file is None:
        st.info("Choose a CSV file to get started.")
        return

    try:
        # Read the CSV to preview and validate that it contains records.
        input_df = pd.read_csv(uploaded_file)

        if input_df.empty:
            st.error("The uploaded CSV contains no customer records.")
            return

        st.success(f"Uploaded {len(input_df):,} customer records.")
        st.subheader("Input preview")
        st.dataframe(input_df.head(10), use_container_width=True)

        if not st.button("Run CSV Batch Scoring", type="primary"):
            return

        # Reset the upload pointer before sending the original file.
        uploaded_file.seek(0)

        with st.spinner("Generating predictions through FastAPI..."):
            response = requests.post(
                f"{API_BASE_URL}/predict/batch/csv",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "text/csv",
                    )
                },
                timeout=300,
            )
            response.raise_for_status()

        # Parse the CSV returned by the API.
        from io import BytesIO

        result_df = pd.read_csv(BytesIO(response.content))

        required_columns = {
            "Predicted_Class",
            "Churn_Probability",
            "Model_Name",
        }
        missing_columns = required_columns - set(result_df.columns)

        if missing_columns:
            st.error(
                "The API report is missing columns: "
                + ", ".join(sorted(missing_columns))
            )
            return

        if len(result_df) != len(input_df):
            st.error(
                "The number of predictions does not match "
                "the number of uploaded customers."
            )
            return

        st.session_state["csv_batch_prediction_result"] = result_df

    except requests.RequestException as exc:
        st.error(f"FastAPI request failed: {exc}")
        if exc.response is not None:
            st.code(exc.response.text)

    except (ValueError, pd.errors.ParserError) as exc:
        st.error(f"Unable to process the CSV report: {exc}")

    # Display the last successful result after Streamlit reruns.
    result_df = st.session_state.get("csv_batch_prediction_result")

    if result_df is not None:
        st.divider()
        st.subheader("Prediction Results")

        total_customers = len(result_df)
        predicted_churners = int(
            (result_df["Predicted_Class"] == 1).sum()
        )
        high_risk = int(
            (result_df["Churn_Probability"] >= 0.70).sum()
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Customers Scored", f"{total_customers:,}")
        col2.metric("Predicted Churners", f"{predicted_churners:,}")
        col3.metric("High Risk (≥70%)", f"{high_risk:,}")

        st.caption(
            "High risk means a model-estimated churn probability of "
            "70% or higher; it is a dashboard threshold, not a guarantee."
        )

        st.dataframe(result_df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download Churn Predictions CSV",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="churn_predictions.csv",
            mime="text/csv",
        )




def main():
    """Dashboard entry point."""

    st.sidebar.title("Vantara")
    st.sidebar.write("Retail Customer Analytics")

    selected_page = st.sidebar.radio(
        "Select Analysis",
        [
            "Home",
            "Customer Segmentation",
            "Churn Analysis",
            "Revenue Analysis",
            "SHAP Explainability",
            "CSV Batch Scoring",
        ],
    )

    if selected_page == "Home":
        show_home()

    elif selected_page == "Customer Segmentation":
        show_customer_segmentation()

    elif selected_page == "Churn Analysis":
        show_churn_leaderboard()

    elif selected_page == "Revenue Analysis":
        show_revenue_trends()

    elif selected_page == "SHAP Explainability":
        show_shap_explainability()

    elif selected_page == "CSV Batch Scoring":
        show_csv_batch_scoring()


if __name__ == "__main__":
    main()
