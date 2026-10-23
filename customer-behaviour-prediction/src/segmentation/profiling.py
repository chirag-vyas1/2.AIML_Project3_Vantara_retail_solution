
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

################################################################
#---- K-means clustering Experiments ------
################################################################

#----- 1. to run experiment on clustering ----------
def run_kmeans_experiments(
    X,
    k_values=range(2, 11),
    random_state=42,
    n_init=10,
    silhouette_sample_size=2000,
):
    """
    Run K-Means experiments for different numbers of clusters.

    Parameters
    ----------
    X : pandas.DataFrame
        Scaled customer feature data.

    k_values : iterable
        Candidate numbers of clusters.

    random_state : int
        Random seed for reproducibility.

    n_init : int
        Number of centroid initializations.

    silhouette_sample_size : int or None
        Number of rows used for silhouette scoring.
        None means use all rows.

    Returns
    -------
    dict
        results: Experiment metrics as a DataFrame.
        models: Fitted K-Means models keyed by k.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if X.empty:
        raise ValueError("X cannot be empty.")

    if not X.columns.is_unique:
        raise ValueError("Feature names must be unique.")

    if not all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in X.dtypes
    ):
        raise ValueError("All features must be numeric.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("X contains missing or infinite values.")

    k_values = list(k_values)

    if not k_values:
        raise ValueError("Provide at least one candidate k.")

    if any(
        not isinstance(k, (int, np.integer))
        or isinstance(k, (bool, np.bool_))
        or not 2 <= k < len(X)
        for k in k_values
    ):
        raise ValueError(
            "Each k must be an integer between 2 and len(X) - 1."
        )

    if len(set(k_values)) != len(k_values):
        raise ValueError("Candidate k values must be unique.")

    if n_init < 1:
        raise ValueError("n_init must be at least 1.")

    if (
        silhouette_sample_size is not None
        and silhouette_sample_size < 2
    ):
        raise ValueError(
            "silhouette_sample_size must be at least 2 or None."
        )

    results = []
    models = {}

    sample_size = (
        None
        if silhouette_sample_size is None
        else min(silhouette_sample_size, len(X))
    )

    for k in k_values:
        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=n_init,
        )

        labels = model.fit_predict(X)

        score = silhouette_score(
            X,
            labels,
            sample_size=sample_size,
            random_state=random_state,
        )

        results.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette_score": score,
        })

        models[k] = model

    return {
        "results": pd.DataFrame(results),
        "models": models,
    }


#------2. visualize k-means experiement --------------------------------
def visualize_kmeans_experiments(results_df):
    """
    Visualize K-Means experiment results using:
    1. Elbow curve (inertia versus k)
    2. Silhouette score versus k

    Parameters
    ----------
    results_df : pandas.DataFrame
        Results returned by run_kmeans_experiments().
        Required columns: k, inertia, silhouette_score.
    """

    if not isinstance(results_df, pd.DataFrame):
        raise TypeError("results_df must be a pandas DataFrame.")

    required_columns = {
        "k",
        "inertia",
        "silhouette_score",
    }

    missing_columns = required_columns - set(results_df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if results_df.empty:
        raise ValueError("Experiment results cannot be empty.")

    if not np.isfinite(
        results_df[
            ["k", "inertia", "silhouette_score"]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("Experiment results contain invalid values.")

    results_df = results_df.sort_values("k")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Elbow curve
    axes[0].plot(
        results_df["k"],
        results_df["inertia"],
        marker="o",
    )
    axes[0].set_title("K-Means Elbow Method")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].set_xticks(results_df["k"])
    axes[0].grid(True)

    # Silhouette curve
    axes[1].plot(
        results_df["k"],
        results_df["silhouette_score"],
        marker="o",
    )
    axes[1].set_title("K-Means Silhouette Analysis")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_xticks(results_df["k"])
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


#------ 3. fit_kmeans_clustering ----------------------------------------------------
def fit_kmeans_cluster(
    X,
    n_clusters,
    random_state=42,
    n_init=10,
):
    """
    Fit a K-Means model using the selected number of clusters.

    Parameters
    ----------
    X : pandas.DataFrame
        Scaled customer feature data.

    n_clusters : int
        Selected number of clusters.

    random_state : int
        Random seed for reproducibility.

    n_init : int
        Number of centroid initializations.

    Returns
    -------
    dict
        model: Fitted K-Means model.
        labels: Cluster assignment for each customer.
        cluster_sizes: Number of customers per cluster.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if X.empty:
        raise ValueError("X cannot be empty.")

    if not X.columns.is_unique:
        raise ValueError("Feature names must be unique.")

    if not all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in X.dtypes
    ):
        raise ValueError("All features must be numeric.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("X contains missing or infinite values.")

    if (
        not isinstance(n_clusters, (int, np.integer))
        or isinstance(n_clusters, (bool, np.bool_))
        or not 2 <= n_clusters < len(X)
    ):
        raise ValueError(
            "n_clusters must be between 2 and len(X) - 1."
        )

    if n_init < 1:
        raise ValueError("n_init must be at least 1.")

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )

    labels = model.fit_predict(X)

    cluster_sizes = (
        pd.Series(labels, name="Cluster")
        .value_counts()
        .sort_index()
        .rename_axis("Cluster")
        .reset_index(name="Customer_Count")
    )

    return {
        "model": model,
        "labels": labels,
        "cluster_sizes": cluster_sizes,
    }

#------ 4. Evaluate kmeans model ----------------------------------------------------
def evaluate_kmeans_cluster(X, model, labels):
    """
    Evaluate the final fitted K-Means clustering model.

    Metrics:
    - Inertia
    - Silhouette score
    - Davies-Bouldin index
    - Cluster-size distribution

    Parameters
    ----------
    X : pandas.DataFrame
        Scaled customer features used to fit the model.

    model : fitted sklearn KMeans model
        Final selected K-Means model.

    labels : array-like
        Cluster labels produced by the fitted model.

    Returns
    -------
    dict
        Evaluation metrics and cluster-size summary.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if X.empty:
        raise ValueError("X cannot be empty.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("X contains missing or infinite values.")

    labels = np.asarray(labels)

    if labels.ndim != 1 or len(labels) != len(X):
        raise ValueError(
            "labels must contain one cluster assignment per row in X."
        )

    if pd.isna(labels).any():
        raise ValueError("Cluster labels cannot contain missing values.")

    n_clusters = len(np.unique(labels))

    if n_clusters < 2 or n_clusters >= len(X):
        raise ValueError(
            "Evaluation requires between 2 and len(X) - 1 clusters."
        )

    if n_clusters != model.n_clusters:
        raise ValueError(
            "The number of unique labels does not match model.n_clusters."
        )

    # Calculate evaluation metrics
    inertia = model.inertia_

    silhouette = silhouette_score(X, labels)

    davies_bouldin = davies_bouldin_score(X, labels)

    # Count customers in each cluster
    cluster_sizes = (
        pd.Series(labels, name="Cluster")
        .value_counts()
        .sort_index()
        .rename_axis("Cluster")
        .reset_index(name="Customer_Count")
    )

    cluster_sizes["Customer_Percentage"] = (
        cluster_sizes["Customer_Count"] / len(X) * 100
    )

    evaluation_metrics = {
        "n_clusters": n_clusters,
        "inertia": float(inertia),
        "silhouette_score": float(silhouette),
        "davies_bouldin_index": float(davies_bouldin),
    }

    return {
        "metrics": evaluation_metrics,
        "cluster_sizes": cluster_sizes,
    }
