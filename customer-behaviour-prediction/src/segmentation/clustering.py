
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

from sklearn.cluster import KMeans

from sklearn.mixture import GaussianMixture


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

################################################################
# 1. ---- K-means clustering Experiments ------
################################################################

#----- 1.1. to run experiment on clustering ----------
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


#------1.2. visualize k-means experiement --------------------------------
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


#------ 1.3. fit_kmeans_clustering ----------------------------------------------------
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

#------ 1.4. Evaluate kmeans model ----------------------------------------------------
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




####################################################################
#----- 2. GMM clustering Experiments ----------------------------------
####################################################################

#------ 1.1 GMM run experiment methods -----------------------------
def run_gmm_experiments(
    X,
    n_components_values=range(2, 11),
    covariance_type="full",
    random_state=42,
    n_init=3,
    max_iter=200,
    silhouette_sample_size=2000,
):
    """
    Experiment with different numbers of GMM components.

    Returns
    -------
    dict
        results: Experiment metrics DataFrame.
        models: Fitted GMM models keyed by component count.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if X.empty:
        raise ValueError("X cannot be empty.")
    if not X.columns.is_unique:
        raise ValueError("Feature names must be unique.")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("X contains missing or infinite values.")
    if not all(pd.api.types.is_numeric_dtype(d) for d in X.dtypes):
        raise ValueError("All features must be numeric.")

    component_values = list(n_components_values)

    if not component_values:
        raise ValueError("Provide candidate component counts.")

    if any(
        not isinstance(n, (int, np.integer))
        or isinstance(n, (bool, np.bool_))
        or not 2 <= n < len(X)
        for n in component_values
    ):
        raise ValueError(
            "Each component count must be between 2 and len(X) - 1."
        )

    if len(set(component_values)) != len(component_values):
        raise ValueError("Component counts must be unique.")

    if n_init < 1 or max_iter < 1:
        raise ValueError("n_init and max_iter must be positive.")

    if (
        silhouette_sample_size is not None
        and silhouette_sample_size < 2
    ):
        raise ValueError(
            "silhouette_sample_size must be at least 2 or None."
        )

    sample_size = (
        None
        if silhouette_sample_size is None
        else min(silhouette_sample_size, len(X))
    )

    results = []
    models = {}

    for n_components in component_values:
        model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
        )

        labels = model.fit_predict(X)

        unique_labels = np.unique(labels)
        score = (
            silhouette_score(
                X,
                labels,
                sample_size=sample_size,
                random_state=random_state,
            )
            if 2 <= len(unique_labels) < len(X)
            else np.nan
        )

        results.append({
            "n_components": n_components,
            "bic": model.bic(X),
            "aic": model.aic(X),
            "silhouette_score": score,
            "converged": model.converged_,
            "iterations": model.n_iter_,
            "smallest_cluster_size": int(
                pd.Series(labels).value_counts().min()
            ),
        })

        models[n_components] = model

    return {
        "results": pd.DataFrame(results),
        "models": models,
    }


#-------- 1.2 visualize gmm --------------------------------
def visualize_gmm_experiments(results_df):
    """
    Visualize GMM experiment results:
    1. BIC by number of components
    2. AIC by number of components
    3. Silhouette score by number of components
    """

    if not isinstance(results_df, pd.DataFrame):
        raise TypeError("results_df must be a pandas DataFrame.")

    required = {
        "n_components", "bic", "aic", "silhouette_score"
    }
    missing = required - set(results_df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )
    if results_df.empty:
        raise ValueError("Experiment results cannot be empty.")

    results_df = results_df.sort_values("n_components")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plots = [
        ("bic", "GMM Bayesian Information Criterion (BIC)"),
        ("aic", "GMM Akaike Information Criterion (AIC)"),
        ("silhouette_score", "GMM Silhouette Score"),
    ]

    for ax, (column, title) in zip(axes, plots):
        ax.plot(
            results_df["n_components"],
            results_df[column],
            marker="o",
        )
        ax.set_title(title)
        ax.set_xlabel("Number of Components")
        ax.set_ylabel(column.replace("_", " ").title())
        ax.set_xticks(results_df["n_components"])
        ax.grid(True)

    plt.tight_layout()
    plt.show()


#--------1.3 fit GMM cluster --------------------------------
def fit_gmm_cluster(
    X,
    n_components,
    covariance_type="full",
    random_state=42,
    n_init=3,
    max_iter=200,
):
    """
    Fit the selected GMM model.

    Returns
    -------
    dict
        model: Fitted GaussianMixture model.
        labels: Most likely component for each customer.
        probabilities: Probability of membership in each component.
        cluster_sizes: Customer counts and percentages by cluster.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if X.empty:
        raise ValueError("X cannot be empty.")
    if not X.columns.is_unique:
        raise ValueError("Feature names must be unique.")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("X contains missing or infinite values.")
    if not all(pd.api.types.is_numeric_dtype(d) for d in X.dtypes):
        raise ValueError("All features must be numeric.")

    if (
        not isinstance(n_components, (int, np.integer))
        or isinstance(n_components, (bool, np.bool_))
        or not 2 <= n_components < len(X)
    ):
        raise ValueError(
            "n_components must be between 2 and len(X) - 1."
        )

    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
    )

    labels = model.fit_predict(X)
    probabilities = model.predict_proba(X)

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

    return {
        "model": model,
        "labels": labels,
        "probabilities": probabilities,
        "cluster_sizes": cluster_sizes,
    }


#------- 1.4 evaluate GMM clustering -----------------------
def evaluate_gmm_cluster(X, model, labels):
    """
    Evaluate a fitted GMM clustering model.

    Returns evaluation metrics and cluster-size distribution.
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
            "Provide one cluster label for each row in X."
        )
    if pd.isna(labels).any():
        raise ValueError("Cluster labels cannot contain missing values.")

    n_clusters = len(np.unique(labels))

    if not 2 <= n_clusters < len(X):
        raise ValueError(
            "Evaluation requires between 2 and len(X) - 1 clusters."
        )

    if n_clusters != model.n_components:
        raise ValueError(
            "The number of observed clusters differs from "
            "model.n_components. Inspect the fitted GMM."
        )

    silhouette = silhouette_score(X, labels)
    db_index = davies_bouldin_score(X, labels)

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

    metrics = {
        "n_components": int(model.n_components),
        "bic": float(model.bic(X)),
        "aic": float(model.aic(X)),
        "silhouette_score": float(silhouette),
        "davies_bouldin_index": float(db_index),
        "converged": bool(model.converged_),
        "iterations": int(model.n_iter_),
    }

    return {
        "metrics": metrics,
        "cluster_sizes": cluster_sizes,
    }


####################################################################
#----- 3. DBSCAN clustering Experiments ----------------------------------
####################################################################


#--- 3.1 DBSCAN clustering Experiment --------------------------------
def run_dbscan_experiments(
    X,
    eps_values=(2.0, 3.0, 4.0, 5.0, 6.0),
    min_samples_values=(5, 10, 20),
):
    """
    Experiment with DBSCAN parameter combinations.

    Silhouette and Davies-Bouldin scores are calculated on
    non-noise customers only, when at least two clusters exist.

    Returns a dictionary containing the experiment results
    and fitted candidate models.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if X.empty:
        raise ValueError("X cannot be empty.")
    if not X.columns.is_unique:
        raise ValueError("Feature names must be unique.")
    if not all(pd.api.types.is_numeric_dtype(d) for d in X.dtypes):
        raise ValueError("All features must be numeric.")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("X contains missing or infinite values.")

    eps_values = list(eps_values)
    min_samples_values = list(min_samples_values)

    if not eps_values or not min_samples_values:
        raise ValueError("Provide candidate eps and min_samples values.")

    if any(not np.isfinite(eps) or eps <= 0 for eps in eps_values):
        raise ValueError("Every eps value must be positive and finite.")

    if any(
        not isinstance(m, (int, np.integer))
        or isinstance(m, (bool, np.bool_))
        or m < 1
        for m in min_samples_values
    ):
        raise ValueError("min_samples values must be positive integers.")

    results = []
    models = {}

    for eps in eps_values:
        for min_samples in min_samples_values:
            model = DBSCAN(
                eps=eps,
                min_samples=min_samples,
            )
            labels = model.fit_predict(X)

            noise_mask = labels == -1
            non_noise_mask = ~noise_mask
            non_noise_labels = labels[non_noise_mask]
            unique_clusters = np.unique(non_noise_labels)

            n_clusters = len(unique_clusters)
            n_noise = int(noise_mask.sum())
            n_assigned = int(non_noise_mask.sum())

            silhouette = np.nan
            db_index = np.nan

            # Evaluate only if at least two clusters remain
            # among non-noise customers.
            if 2 <= n_clusters < n_assigned:
                X_assigned = X.iloc[np.flatnonzero(non_noise_mask)]

                silhouette = silhouette_score(
                    X_assigned,
                    non_noise_labels,
                )
                db_index = davies_bouldin_score(
                    X_assigned,
                    non_noise_labels,
                )

            results.append({
                "eps": eps,
                "min_samples": min_samples,
                "n_clusters": n_clusters,
                "noise_count": n_noise,
                "noise_percentage": n_noise / len(X) * 100,
                "assigned_count": n_assigned,
                "silhouette_score": silhouette,
                "davies_bouldin_index": db_index,
            })

            models[(eps, min_samples)] = model

    return {
        "results": pd.DataFrame(results),
        "models": models,
    }

#--- 3.2 DBBSCAN visualaizing ----------------------------------------------------------------
def visualize_dbscan_experiments(results_df):
    """
    Visualize DBSCAN silhouette scores and noise percentages.
    """

    required = {
        "eps",
        "min_samples",
        "silhouette_score",
        "noise_percentage",
        "n_clusters",
    }

    if not isinstance(results_df, pd.DataFrame):
        raise TypeError("results_df must be a pandas DataFrame.")

    missing = required - set(results_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if results_df.empty:
        raise ValueError("Experiment results cannot be empty.")

    plot_df = results_df.copy()
    plot_df["configuration"] = plot_df.apply(
        lambda row: (
            f"eps={row['eps']}, min_samples={row['min_samples']}"
        ),
        axis=1,
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].bar(
        plot_df["configuration"],
        plot_df["silhouette_score"],
    )
    axes[0].set_title("DBSCAN Silhouette Scores")
    axes[0].set_ylabel("Silhouette Score")
    axes[0].tick_params(axis="x", rotation=90)
    axes[0].grid(axis="y")

    axes[1].bar(
        plot_df["configuration"],
        plot_df["noise_percentage"],
    )
    axes[1].set_title("DBSCAN Noise Percentage")
    axes[1].set_ylabel("Noise (%)")
    axes[1].tick_params(axis="x", rotation=90)
    axes[1].grid(axis="y")

    plt.tight_layout()
    plt.show()

#---- 3.3 fit DBSCAN --------------------------------
def fit_dbscan_cluster(X, eps, min_samples=5):
    """
    Fit DBSCAN using the selected parameters.

    Cluster label -1 represents noise.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if X.empty:
        raise ValueError("X cannot be empty.")
    if not X.columns.is_unique:
        raise ValueError("Feature names must be unique.")
    if not all(pd.api.types.is_numeric_dtype(d) for d in X.dtypes):
        raise ValueError("All features must be numeric.")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("X contains missing or infinite values.")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite.")
    if (
        not isinstance(min_samples, (int, np.integer))
        or isinstance(min_samples, (bool, np.bool_))
        or min_samples < 1
    ):
        raise ValueError("min_samples must be a positive integer.")

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
    )
    labels = model.fit_predict(X)

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

    return {
        "model": model,
        "labels": labels,
        "cluster_sizes": cluster_sizes,
    }


#------ 3.4 Evaluate dbScan --------------------------------
def evaluate_dbscan_cluster(X, labels):
    """
    Evaluate DBSCAN cluster assignments.

    Noise points (label -1) are excluded from silhouette and
    Davies-Bouldin calculations but counted in the noise metrics.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if X.empty:
        raise ValueError("X cannot be empty.")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("X contains missing or infinite values.")

    labels = np.asarray(labels)

    if labels.ndim != 1 or len(labels) != len(X):
        raise ValueError("Provide one label for every row in X.")
    if pd.isna(labels).any():
        raise ValueError("Labels cannot contain missing values.")

    noise_mask = labels == -1
    assigned_mask = ~noise_mask
    assigned_labels = labels[assigned_mask]
    unique_clusters = np.unique(assigned_labels)

    n_clusters = len(unique_clusters)
    n_noise = int(noise_mask.sum())
    n_assigned = int(assigned_mask.sum())

    silhouette = np.nan
    db_index = np.nan

    if 2 <= n_clusters < n_assigned:
        X_assigned = X.iloc[np.flatnonzero(assigned_mask)]

        silhouette = silhouette_score(
            X_assigned,
            assigned_labels,
        )
        db_index = davies_bouldin_score(
            X_assigned,
            assigned_labels,
        )

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

    metrics = {
        "n_clusters_excluding_noise": n_clusters,
        "noise_count": n_noise,
        "noise_percentage": n_noise / len(X) * 100,
        "assigned_count": n_assigned,
        "silhouette_score_excluding_noise": silhouette,
        "davies_bouldin_index_excluding_noise": db_index,
    }

    return {
        "metrics": metrics,
        "cluster_sizes": cluster_sizes,
    }
