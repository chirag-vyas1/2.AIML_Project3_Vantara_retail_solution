
import pandas as pd
from IPython.display import display

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


# ------- 1. Model Registry for Classification Models -------
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


#------ 3. library for cross validation ------
from sklearn.model_selection import RandomizedSearchCV




# ------- 1. Model Registry for -------
def get_model_registry(config: dict) -> dict:
    """
    Build the classification model registry from config.yaml.

    Only models marked as enabled in the configuration
    are included in the registry.
    """

    classification_config = (
        config["model"]["classification"]
    )

    model_registry = {}

    if classification_config["logistic_regression"]["enabled"]:
        model_registry["logistic_regression"] = {
            "model_class": LogisticRegression,
        }

    if classification_config["decision_tree"]["enabled"]:
        model_registry["decision_tree"] = {
            "model_class": DecisionTreeClassifier,
        }

    if classification_config["random_forest"]["enabled"]:
        model_registry["random_forest"] = {
            "model_class": RandomForestClassifier,
        }

    if classification_config["xgboost"]["enabled"]:
        model_registry["xgboost"] = {
            "model_class": XGBClassifier,
        }

    if classification_config["lightgbm"]["enabled"]:
        model_registry["lightgbm"] = {
            "model_class": LGBMClassifier,
        }

    if classification_config["svm"]["enabled"]:
        model_registry["svm"] = {
            "model_class": SVC,
        }

    return model_registry

#-------- 2. building baseline models ----
def build_classification_model(
    model_name: str,
    config: dict,
):
    """
    Build and return a fresh classification model
    using the base parameters defined in config.yaml.

    Parameters
    ----------
    model_name : str
        Name of the classification model.

    config : dict
        Project configuration loaded from config.yaml.

    Returns
    -------
    model
        Configured classification model.
    """

    classification_config = (
        config["model"]["classification"]
    )

    if model_name not in classification_config:
        raise ValueError(
            f"Unknown classification model: {model_name}"
        )

    model_config = classification_config[model_name]

    if not model_config.get("enabled", False):
        raise ValueError(
            f"Classification model '{model_name}' is disabled."
        )

    model_registry = get_model_registry(config)

    if model_name not in model_registry:
        raise ValueError(
            f"Model '{model_name}' is not available in the registry."
        )

    model_class = model_registry[model_name]["model_class"]

    base_parameters = model_config.get(
        "base_parameters",
        {},
    )

    model = model_class(
        **base_parameters
    )

    return model


#-------- 3. training individual basemodel  function ----------
def train_classification_model(
    model,
    X_train,
    y_train,
):
    """
    Train a classification model and record training time.

    Parameters
    ----------
    model : estimator
        Unfitted classification model.

    X_train : pandas.DataFrame
        Training feature matrix.

    y_train : pandas.Series
        Training target.

    Returns
    -------
    tuple
        fitted_model, training_time_seconds
    """

    if X_train.empty:
        raise ValueError("X_train is empty.")

    if y_train.empty:
        raise ValueError("y_train is empty.")

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train and y_train must contain "
            "the same number of observations."
        )

    start_time = time.perf_counter()

    model.fit(
        X_train,
        y_train,
    )

    end_time = time.perf_counter()

    training_time_seconds = (
        end_time - start_time
    )

    return (
        model,
        training_time_seconds,
    )


#-------- 4. train baseline for all six classification model-------
def train_baseline_classification_models(
    config: dict,
    X_train,
    X_train_scaled,
    y_train,
):
    """
    Train all enabled classification models using their
    configured preprocessing requirements.

    Returns
    -------
    dict
        Dictionary containing fitted models and training times.
    """

    classification_config = (
        config["model"]["classification"]
    )

    model_names = [
        model_name
        for model_name, model_config
        in classification_config.items()
        if model_config.get("enabled", False)
    ]

    trained_models = {}

    for model_name in model_names:

        model = build_classification_model(
            model_name=model_name,
            config=config,
        )

        use_scaled_features = (
            classification_config[model_name]
            .get("feature_scaling", False)
        )

        if use_scaled_features:
            X_train_model = X_train_scaled
        else:
            X_train_model = X_train

        fitted_model, training_time = (
            train_classification_model(
                model=model,
                X_train=X_train_model,
                y_train=y_train,
            )
        )

        trained_models[model_name] = {
            "model": fitted_model,
            "training_time": training_time,
            "feature_scaling": use_scaled_features,
        }

    return trained_models


#------- 5. Evaluate the classification model ----------
def evaluate_classification_model(
    model,
    X_data,
    y_data,
):
    """
    Evaluate a fitted classification model.

    Parameters
    ----------
    model : estimator
        Fitted classification model.

    X_data : pandas.DataFrame
        Feature matrix for evaluation.

    y_data : pandas.Series
        Target values for evaluation.

    Returns
    -------
    dict
        Classification evaluation metrics.
    """

    if X_data.empty:
        raise ValueError("X_data is empty.")

    if y_data.empty:
        raise ValueError("y_data is empty.")

    if len(X_data) != len(y_data):
        raise ValueError(
            "X_data and y_data must contain "
            "the same number of observations."
        )

    y_pred = model.predict(X_data)

    accuracy = accuracy_score(
        y_data,
        y_pred,
    )

    precision = precision_score(
        y_data,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_data,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_data,
        y_pred,
        zero_division=0,
    )

    if hasattr(model, "predict_proba"):

        y_score = model.predict_proba(
            X_data
        )[:, 1]

    else:

        y_score = model.decision_function(
            X_data
        )

    roc_auc = roc_auc_score(
        y_data,
        y_score,
    )

    confusion = confusion_matrix(
        y_data,
        y_pred,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": confusion,
    }


#--------6. Evalaution for all six-----
def evaluate_baseline_classification_models(
    baseline_models: dict,
    X_validation,
    X_validation_scaled,
    y_validation,
):
    """
    Evaluate all trained baseline classification models
    on the validation dataset.

    The function automatically selects scaled or unscaled
    validation features according to each model's configuration.

    Returns
    -------
    dict
        Dictionary containing evaluation results for each model.
    """

    if X_validation.empty:
        raise ValueError("X_validation is empty.")

    if X_validation_scaled.empty:
        raise ValueError("X_validation_scaled is empty.")

    if y_validation.empty:
        raise ValueError("y_validation is empty.")

    if len(X_validation) != len(y_validation):
        raise ValueError(
            "X_validation and y_validation must contain "
            "the same number of observations."
        )

    if len(X_validation_scaled) != len(y_validation):
        raise ValueError(
            "X_validation_scaled and y_validation must contain "
            "the same number of observations."
        )

    evaluation_results = {}

    for model_name, result in baseline_models.items():

        model = result["model"]

        use_scaled_features = result["feature_scaling"]

        if use_scaled_features:
            X_validation_model = X_validation_scaled
        else:
            X_validation_model = X_validation

        metrics = evaluate_classification_model(
            model=model,
            X_data=X_validation_model,
            y_data=y_validation,
        )

        evaluation_results[model_name] = metrics

    return evaluation_results



#---------7.hyperparameter tuning random frorest ------
def tune_classification_model(
    model,
    hyperparameter_space,
    X_train,
    y_train,
    cv_folds=5,
    scoring="roc_auc",
    n_iter=20,
    random_state=42,
):
    """
    Tune a classification model using randomized search
    with cross-validation.

    Parameters
    ----------
    model : estimator
        Unfitted classification model.

    hyperparameter_space : dict
        Hyperparameter search space from config.yaml.

    X_train : pandas.DataFrame
        Training feature matrix.

    y_train : pandas.Series
        Training target.

    cv_folds : int, default=5
        Number of cross-validation folds.

    scoring : str, default="roc_auc"
        Metric used to select the best model.

    n_iter : int, default=20
        Number of parameter combinations to evaluate.

    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    tuple
        best_model, best_parameters, best_cv_score, tuning_time
    """

    if X_train.empty:
        raise ValueError("X_train is empty.")

    if y_train.empty:
        raise ValueError("y_train is empty.")

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train and y_train must contain "
            "the same number of observations."
        )

    if not hyperparameter_space:
        raise ValueError(
            "hyperparameter_space cannot be empty."
        )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=hyperparameter_space,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv_folds,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )

    start_time = time.perf_counter()

    search.fit(
        X_train,
        y_train,
    )

    end_time = time.perf_counter()

    tuning_time = end_time - start_time

    return (
        search.best_estimator_,
        search.best_params_,
        search.best_score_,
        tuning_time,
    )
