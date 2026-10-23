from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import time
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
from sklearn.model_selection import RandomizedSearchCV




# ------- 1. Model Registry for Regression -------
def get_regression_model_registry(
    config: dict,
) -> dict:
    """
    Build the regression model registry from config.yaml.

    Only models marked as enabled in the configuration
    are included in the registry.
    """

    regression_config = (
        config["model"]["regression"]
    )

    model_registry = {}

    if regression_config["linear_regression"]["enabled"]:
        model_registry["linear_regression"] = {
            "model_class": LinearRegression,
        }

    if regression_config["ridge"]["enabled"]:
        model_registry["ridge"] = {
            "model_class": Ridge,
        }

    if regression_config["decision_tree"]["enabled"]:
        model_registry["decision_tree"] = {
            "model_class": DecisionTreeRegressor,
        }

    if regression_config["random_forest"]["enabled"]:
        model_registry["random_forest"] = {
            "model_class": RandomForestRegressor,
        }

    if regression_config["xgboost"]["enabled"]:
        model_registry["xgboost"] = {
            "model_class": XGBRegressor,
        }

    if regression_config["lightgbm"]["enabled"]:
        model_registry["lightgbm"] = {
            "model_class": LGBMRegressor,
        }

    if regression_config["svr"]["enabled"]:
        model_registry["svr"] = {
            "model_class": SVR,
        }

    return model_registry

# ------- 2. Model Building for Regression -------
def build_regression_model(
    model_name: str,
    config: dict,
):
    """
    Build and return a fresh regression model
    using the base parameters defined in config.yaml.

    Parameters
    ----------
    model_name : str
        Name of the regression model.

    config : dict
        Project configuration loaded from config.yaml.

    Returns
    -------
    model
        Configured regression model.
    """

    regression_config = (
        config["model"]["regression"]
    )

    if model_name not in regression_config:
        raise ValueError(
            f"Unknown regression model: {model_name}"
        )

    model_config = regression_config[model_name]

    if not model_config.get("enabled", False):
        raise ValueError(
            f"Regression model '{model_name}' is disabled."
        )

    model_registry = get_regression_model_registry(config)

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

# ------- 3. Training Individual Regression Model -------
def train_regression_model(
    model,
    X_train,
    y_train,
):
    """
    Train a regression model and record training time.

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

    return model, training_time_seconds

# ------- 4. Train Baseline Regression Models -------
def train_baseline_regression_models(
    config: dict,
    X_train,
    X_train_scaled,
    y_train,
):
    """
    Train all enabled regression models using their
    configured feature-scaling requirements.

    Returns
    -------
    dict
        Dictionary containing fitted models and
        training times.
    """

    regression_config = (
        config["model"]["regression"]
    )

    model_names = [
        model_name
        for model_name, model_config
        in regression_config.items()
        if model_config.get("enabled", False)
    ]

    trained_models = {}

    for model_name in model_names:

        model = build_regression_model(
            model_name=model_name,
            config=config,
        )

        use_scaled_features = (
            regression_config[model_name]
            .get("feature_scaling", False)
        )

        if use_scaled_features:
            X_train_model = X_train_scaled
        else:
            X_train_model = X_train

        fitted_model, training_time = (
            train_regression_model(
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


# ------- 5. Evaluate the Regression Model -------
def evaluate_regression_model(
    model,
    X_data,
    y_data,
):
    """
    Evaluate a fitted regression model.

    Parameters
    ----------
    model : estimator
        Fitted regression model.

    X_data : pandas.DataFrame
        Feature matrix for evaluation.

    y_data : pandas.Series
        Actual target values.

    Returns
    -------
    dict
        Regression evaluation metrics.
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

    mae = mean_absolute_error(
        y_data,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_data,
            y_pred,
        )
    )

    r2 = r2_score(
        y_data,
        y_pred,
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }

# ------- 6. Evaluate All Baseline Regression Models -------
def evaluate_baseline_regression_models(
    baseline_models: dict,
    X_validation,
    X_validation_scaled,
    y_validation,
):
    """
    Evaluate all trained baseline regression models
    on the validation dataset.

    The function selects scaled or unscaled validation
    features according to each model's configuration.

    Returns
    -------
    dict
        Evaluation metrics for each regression model.
    """

    if X_validation.empty:
        raise ValueError("X_validation is empty.")

    if X_validation_scaled.empty:
        raise ValueError(
            "X_validation_scaled is empty."
        )

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

        metrics = evaluate_regression_model(
            model=model,
            X_data=X_validation_model,
            y_data=y_validation,
        )

        evaluation_results[model_name] = metrics

    return evaluation_results


# --------- 7. Hyperparameter tuning: regression ---------
def tune_regression_model(
    model,
    hyperparameter_space,
    X_train,
    y_train,
    cv_folds=5,
    scoring="neg_mean_absolute_error",
    n_iter=20,
    random_state=42,
):
    """
    Tune a regression model using randomized search
    with cross-validation.

    Returns
    -------
    tuple
        best_model, best_parameters, best_cv_score, tuning_time
    """

    if X_train is None or len(X_train) == 0:
        raise ValueError("X_train is empty.")

    if y_train is None or len(y_train) == 0:
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
        error_score="raise",
    )

    start_time = time.perf_counter()

    search.fit(X_train, y_train)

    tuning_time = time.perf_counter() - start_time

    return (
        search.best_estimator_,
        search.best_params_,
        search.best_score_,
        tuning_time,
    )



########################################################################################
#================================================================
#--- lstm MODEL ----------------------------
#=================================================================
########################################################################################################################
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from pathlib import Path
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def build_lstm_model(
    sequence_length,
    n_features,
    lstm_units=128,
    dropout_rate=0.20,
    dense_units=32,
    learning_rate=0.0001,
    huber_delta=10.0,
):
    """Build and compile a masked LSTM regression model."""

    sequence_input = Input(
        shape=(sequence_length, n_features),
        name="sequence_input",
    )

    event_mask = Input(
        shape=(sequence_length,),
        dtype="bool",
        name="event_mask",
    )

    x = LSTM(
        units=lstm_units,
        name="lstm_layer",
    )(sequence_input, mask=event_mask)

    x = Dropout(
        rate=dropout_rate,
        name="dropout_layer",
    )(x)

    x = Dense(
        units=dense_units,
        activation="relu",
        name="dense_layer",
    )(x)

    output = Dense(
        units=1,
        activation="linear",
        name="predicted_days",
    )(x)

    model = Model(
        inputs=[sequence_input, event_mask],
        outputs=output,
        name="lstm_regressor",
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.Huber(delta=huber_delta),
        metrics=["mae"],
    )

    return model

def train_lstm_model(
    model,
    X_train,
    mask_train,
    y_train,
    X_validation,
    mask_validation,
    y_validation,
    batch_size=64,
    max_epochs=60,
    patience=7,
    verbose=0,
):
    """Train an LSTM with validation-based early stopping."""

    early_stopping = EarlyStopping(
        monitor="val_mae",
        patience=patience,
        restore_best_weights=True,
        mode="min",
        verbose=verbose,
    )

    history = model.fit(
        x={
            "sequence_input": X_train,
            "event_mask": mask_train,
        },
        y=y_train,
        validation_data=(
            {
                "sequence_input": X_validation,
                "event_mask": mask_validation,
            },
            y_validation,
        ),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=verbose,
    )

    best_epoch = int(np.argmin(history.history["val_mae"]) + 1)

    return model, history, best_epoch

def tune_lstm_model(
    split_result,
    regression_config,
):
    """
    Tune LSTM hyperparameters using the existing scaled,
    chronologically split sequences.

    Parameters
    ----------
    split_result : dict
        Output from split_lstm_sequences_chronologically().
    regression_config : dict
        Already-extracted 'regression' section from the
        project's existing configuration loader.

    Returns
    -------
    dict
        Tuning results, best model, history, parameters,
        and validation metrics.
    """

    lstm_config = regression_config["lstm"]
    tuning_config = regression_config["tuning"]

    if not lstm_config.get("enabled", True):
        raise ValueError("LSTM regression is disabled in configuration.")

    if lstm_config.get("feature_scaling", False):
        raise ValueError(
            "LSTM feature_scaling must be false because the "
            "sequence pipeline already provides scaled inputs."
        )

    # Use the existing scaled chronological partitions.
    X_train = split_result["X_train"]
    X_validation = split_result["X_validation"]

    mask_train = split_result["mask_train"]
    mask_validation = split_result["mask_validation"]

    y_train = np.asarray(
        split_result["y_train"]
    ).reshape(-1)

    y_validation = np.asarray(
        split_result["y_validation"]
    ).reshape(-1)

    # Validate dimensions.
    if X_train.ndim != 3 or X_validation.ndim != 3:
        raise ValueError("LSTM sequence inputs must be 3D arrays.")

    if X_train.shape[1:] != X_validation.shape[1:]:
        raise ValueError("Training and validation sequence shapes differ.")

    if mask_train.shape != X_train.shape[:2]:
        raise ValueError("Training mask shape does not match sequences.")

    if mask_validation.shape != X_validation.shape[:2]:
        raise ValueError(
            "Validation mask shape does not match sequences."
        )

    if len(y_train) != len(X_train):
        raise ValueError("Training target length mismatch.")

    if len(y_validation) != len(X_validation):
        raise ValueError("Validation target length mismatch.")

    if not all(
        np.isfinite(array).all()
        for array in (
            X_train, X_validation, y_train, y_validation
        )
    ):
        raise ValueError("Non-finite values found in model data.")

    base_parameters = lstm_config.get("base_parameters", {})
    hyperparameter_space = lstm_config["hyperparameter_space"]

    n_iter = int(tuning_config.get("n_iter", 8))
    random_state = int(base_parameters.get("random_state", 42))

    if n_iter < 1:
        raise ValueError("n_iter must be at least 1.")

    candidates = list(
        ParameterSampler(
            hyperparameter_space,
            n_iter=n_iter,
            random_state=random_state,
        )
    )

    max_epochs = int(base_parameters.get("max_epochs", 60))
    patience = int(base_parameters.get("patience", 7))
    verbose = int(base_parameters.get("verbose", 0))

    results = []
    best_model = None
    best_history = None
    best_parameters = None
    best_mae = float("inf")

    for trial, sampled_parameters in enumerate(candidates, start=1):
        print(f"Trial {trial}/{len(candidates)}")
        print(f"Parameters: {sampled_parameters}")

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(random_state + trial)

        # Reuse your existing model-building function unchanged.
        model = build_lstm_model(
            sequence_length=X_train.shape[1],
            n_features=X_train.shape[2],
            lstm_units=int(sampled_parameters["lstm_units"]),
            dropout_rate=float(sampled_parameters["dropout_rate"]),
            dense_units=int(sampled_parameters["dense_units"]),
            learning_rate=float(sampled_parameters["learning_rate"]),
            huber_delta=float(sampled_parameters["huber_delta"]),
        )

        # Reuse your existing training function unchanged.
        model, history, best_epoch = train_lstm_model(
            model=model,
            X_train=X_train,
            mask_train=mask_train,
            y_train=y_train,
            X_validation=X_validation,
            mask_validation=mask_validation,
            y_validation=y_validation,
            batch_size=int(sampled_parameters["batch_size"]),
            max_epochs=max_epochs,
            patience=patience,
            verbose=verbose,
        )

        predictions = model.predict(
            {
                "sequence_input": X_validation,
                "event_mask": mask_validation,
            },
            verbose=0,
        ).reshape(-1)

        val_mae = float(
            mean_absolute_error(y_validation, predictions)
        )

        val_rmse = float(
            np.sqrt(
                mean_squared_error(y_validation, predictions)
            )
        )

        results.append({
            "trial": trial,
            **sampled_parameters,
            "best_epoch": best_epoch,
            "validation_mae": val_mae,
            "validation_rmse": val_rmse,
        })

        print(
            f"Best epoch: {best_epoch} | "
            f"Validation MAE: {val_mae:.3f} | "
            f"Validation RMSE: {val_rmse:.3f}"
        )

        if val_mae < best_mae:
            best_mae = val_mae
            best_model = model
            best_history = history
            best_parameters = {
                **sampled_parameters,
                "max_epochs": max_epochs,
                "patience": patience,
                "random_state": random_state + trial,
                "best_epoch": best_epoch,
            }

    results_df = (
        pd.DataFrame(results)
        .sort_values("validation_mae")
        .reset_index(drop=True)
    )

    median_baseline_mae = mean_absolute_error(
        y_validation,
        np.full(len(y_validation), np.median(y_train)),
    )

    return {
        "results": results_df,
        "best_model": best_model,
        "best_history": best_history,
        "best_params": best_parameters,
        "best_validation_mae": best_mae,
        "median_baseline_mae": float(median_baseline_mae),
    }


def run_lstm_tuning_pipeline(
    split_result,
    config_path="config.yaml",
):
    """Load configuration and run the complete LSTM tuning workflow."""

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path.resolve()}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict) or "regression" not in config:
        raise ValueError(
            "config.yaml must contain a top-level 'regression' section."
        )

    result = tune_lstm_model(
        split_result=split_result,
        regression_config=config["regression"],
    )

    print("\n" + "=" * 55)
    print("LSTM HYPERPARAMETER TUNING COMPLETE")
    print("=" * 55)

    print("\nBest hyperparameters:")
    for key, value in result["best_params"].items():
        print(f"  {key}: {value}")

    print(
        f"\nBest validation MAE: "
        f"{result['best_validation_mae']:.3f} days"
    )
    print(
        f"Median baseline MAE: "
        f"{result['median_baseline_mae']:.3f} days"
    )

    if result["best_validation_mae"] < result["median_baseline_mae"]:
        print("Result: LSTM beats the median baseline.")
    else:
        print("Result: LSTM does not beat the median baseline.")

    print("\nTop tuning results:")
    print(result["results"].head(5).to_string(index=False))

    return result

def evaluate_lstm_model(
    model,
    X,
    mask,
    y_true,
    baseline_value=None,
):
    """
    Evaluate a trained LSTM regression model on one partition.

    Parameters
    ----------
    model : trained Keras model
    X : scaled sequence array, shape (samples, timesteps, features)
    mask : Boolean mask, shape (samples, timesteps)
    y_true : actual target values in days
    baseline_value : optional constant baseline prediction

    Returns
    -------
    dict containing metrics, predictions, and residuals.
    """

    y_true = np.asarray(y_true).reshape(-1)

    if len(X) != len(y_true) or len(mask) != len(y_true):
        raise ValueError("Input sequences, masks, and targets must align.")

    if not np.isfinite(X).all() or not np.isfinite(y_true).all():
        raise ValueError("Inputs contain non-finite values.")

    y_pred = model.predict(
        {
            "sequence_input": X,
            "event_mask": mask,
        },
        verbose=0,
    ).reshape(-1)

    if len(y_pred) != len(y_true):
        raise ValueError("Prediction count does not match target count.")

    residuals = y_true - y_pred

    metrics = {
        "MAE_days": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_days": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }

    if baseline_value is not None:
        baseline_predictions = np.full(
            len(y_true), float(baseline_value)
        )
        metrics["Baseline_MAE_days"] = float(
            mean_absolute_error(y_true, baseline_predictions)
        )
        metrics["MAE_improvement_days"] = (
            metrics["Baseline_MAE_days"] - metrics["MAE_days"]
        )

    predictions_df = pd.DataFrame({
        "actual_days": y_true,
        "predicted_days": y_pred,
        "residual_days": residuals,
        "absolute_error_days": np.abs(residuals),
    })

    return {
        "metrics": metrics,
        "predictions": predictions_df,
    }

def visualize_lstm_evaluation(evaluation_result):
    """
    Visualize predictions and errors from evaluate_lstm_model().
    Returns the three Matplotlib figure objects.
    """

    predictions = evaluation_result["predictions"]

    actual = predictions["actual_days"].to_numpy()
    predicted = predictions["predicted_days"].to_numpy()
    residuals = predictions["residual_days"].to_numpy()
    absolute_errors = predictions["absolute_error_days"].to_numpy()

    # 1. Actual versus predicted.
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    ax1.scatter(actual, predicted, alpha=0.35)

    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())
    ax1.plot([lower, upper], [lower, upper], linestyle="--")

    ax1.set(
        title="LSTM: Actual vs Predicted Purchase Timing",
        xlabel="Actual days",
        ylabel="Predicted days",
    )
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()

    # 2. Residual distribution.
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.hist(residuals, bins=40, edgecolor="black")
    ax2.axvline(0, linestyle="--")

    ax2.set(
        title="LSTM: Residual Distribution",
        xlabel="Residual (actual - predicted), days",
        ylabel="Frequency",
    )
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()

    # 3. Absolute prediction error distribution.
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    ax3.hist(absolute_errors, bins=40, edgecolor="black")

    ax3.set(
        title="LSTM: Absolute Prediction Error",
        xlabel="Absolute error, days",
        ylabel="Frequency",
    )
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()

    plt.show()

    return {
        "actual_vs_predicted": fig1,
        "residual_distribution": fig2,
        "absolute_error_distribution": fig3,
    }

def predict_lstm_hybrid(
    model,
    X,
    mask,
    history_length,
    training_median,
    min_history_for_lstm=4,
    batch_size=64,
):
    """
    Predict days until the next purchase using a hybrid strategy.

    Customers with insufficient history receive the training-median
    prediction. Customers with sufficient history receive LSTM predictions.

    Parameters
    ----------
    model : trained TensorFlow/Keras model
    X : scaled sequence features, shape (n_samples, sequence_length, n_features)
    mask : Boolean sequence mask, shape (n_samples, sequence_length)
    history_length : number of real historical events for each sample
    training_median : median calculated from y_train only
    min_history_for_lstm : minimum history length required to use the LSTM
    batch_size : prediction batch size

    Returns
    -------
    dict containing predictions and the selected prediction method.
    """
    import numpy as np

    X = np.asarray(X)
    mask = np.asarray(mask, dtype=bool)
    history_length = np.asarray(history_length).reshape(-1)

    n_samples = len(X)

    if X.ndim != 3:
        raise ValueError("X must have shape (samples, timesteps, features).")

    if mask.shape != X.shape[:2]:
        raise ValueError("Mask shape must match the first two dimensions of X.")

    if len(history_length) != n_samples:
        raise ValueError("history_length must contain one value per sample.")

    if not np.isfinite(X).all():
        raise ValueError("X contains non-finite values.")

    if not np.isfinite(training_median):
        raise ValueError("training_median must be finite.")

    if np.any(history_length < 1):
        raise ValueError("Every sample must have at least one real event.")

    # Use the existing trained model; do not retrain it.
    lstm_predictions = model.predict(
        {
            "sequence_input": X,
            "event_mask": mask,
        },
        batch_size=batch_size,
        verbose=0,
    ).reshape(-1)

    use_lstm = history_length >= min_history_for_lstm

    # Select LSTM predictions for sufficient history;
    # otherwise use the training-set median.
    hybrid_predictions = np.where(
        use_lstm,
        lstm_predictions,
        float(training_median),
    )

    return {
        "predictions": hybrid_predictions,
        "lstm_predictions": lstm_predictions,
        "use_lstm": use_lstm,
        "prediction_method": np.where(use_lstm, "LSTM", "Median baseline"),
    }
