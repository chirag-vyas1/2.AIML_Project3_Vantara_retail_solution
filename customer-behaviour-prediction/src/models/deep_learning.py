
from pathlib import Path

import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam

from src.data.ingest import load_config
from src.data.ingest import get_project_root


#-----  1. transform churn features with saved scaler  --------------
def transform_churn_features_with_saved_scaler(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    config: dict | None = None,
) -> dict:
    """
    Transform churn train, validation, and test features using the
    existing fitted scaler configured in config.yaml.

    The scaler path is obtained from:
        model_artifacts.preprocessing_dir
        model_artifacts.churn.scaler_filename

    This function only loads and applies the existing scaler. It does
    not fit, refit, or overwrite the scaler.

    Requirements:
        - All feature DataFrames must contain the same columns
          in the same order used when the scaler was fitted.
        - The saved scaler must have been fitted on the corresponding
          training features only.
        - Input features must be numeric, finite, and contain no NaNs.

    Returns:
        dict containing:
            X_train_scaled: scaled training features
            X_val_scaled: scaled validation features
            X_test_scaled: scaled test features
            scaler: loaded fitted scaler
            scaler_path: path to the existing scaler artifact
            feature_columns: ordered list of input feature names
    """

    if config is None:
        config = load_config()

    # --------------------------------------------------
    # 1. Resolve the existing scaler path from YAML
    # --------------------------------------------------
    project_root = Path(get_project_root())

    scaler_path = (
        project_root
        / config["model_artifacts"]["preprocessing_dir"]
        / config["model_artifacts"]["churn"]["scaler_filename"]
    )

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Existing churn scaler not found: {scaler_path}. "
            "Fit and save the appropriate scaler through your "
            "training preprocessing pipeline first."
        )

    # --------------------------------------------------
    # 2. Validate feature columns and their order
    # --------------------------------------------------
    feature_columns = list(X_train.columns)

    if not X_train.columns.is_unique:
        raise ValueError("X_train contains duplicate feature names.")

    for name, X in [
        ("X_val", X_val),
        ("X_test", X_test),
    ]:
        if not X.columns.is_unique:
            raise ValueError(f"{name} contains duplicate feature names.")

        if list(X.columns) != feature_columns:
            raise ValueError(
                f"{name} feature columns/order do not match X_train."
            )

    # --------------------------------------------------
    # 3. Validate numeric, missing, and infinite values
    # --------------------------------------------------
    for name, X in [
        ("X_train", X_train),
        ("X_val", X_val),
        ("X_test", X_test),
    ]:
        if not all(pd.api.types.is_numeric_dtype(dtype)
                   for dtype in X.dtypes):
            raise ValueError(f"{name} contains non-numeric features.")

        if X.isna().any().any():
            raise ValueError(
                f"{name} contains missing values. "
                "Apply the established imputation pipeline first."
            )

        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError(f"{name} contains infinite values.")

    # --------------------------------------------------
    # 4. Load the existing fitted scaler
    # --------------------------------------------------
    scaler = joblib.load(scaler_path)

    if not hasattr(scaler, "transform"):
        raise TypeError("The saved artifact does not support transform().")

    if not hasattr(scaler, "n_features_in_"):
        raise ValueError("The saved scaler is not fitted.")

    if scaler.n_features_in_ != len(feature_columns):
        raise ValueError(
            f"Scaler expects {scaler.n_features_in_} features, "
            f"but X_train contains {len(feature_columns)}."
        )

    if hasattr(scaler, "feature_names_in_"):
        if list(scaler.feature_names_in_) != feature_columns:
            raise ValueError(
                "Feature names/order do not match the saved scaler."
            )

    # --------------------------------------------------
    # 5. Apply the same scaler to every partition
    # --------------------------------------------------
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    for name, X_scaled in [
        ("X_train_scaled", X_train_scaled),
        ("X_val_scaled", X_val_scaled),
        ("X_test_scaled", X_test_scaled),
    ]:
        if not np.isfinite(X_scaled).all():
            raise ValueError(
                f"{name} contains non-finite values after scaling."
            )

    return {
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "scaler_path": str(scaler_path),
        "feature_columns": feature_columns,
    }

#-----  2. validating scaled data --------------------------------
def validate_scaled_churn_features(
    X_train_scaled,
    X_val_scaled,
    X_test_scaled,
    X_train,
    X_val,
    X_test,
    y_train=None,
    y_val=None,
    y_test=None,
    expected_feature_count=27,
    mean_tolerance=0.01,
    std_tolerance=0.05,
    display_report=True,
):
    """
    Validate scaled churn features before ANN training.

    Checks:
        1. Non-empty, two-dimensional feature arrays.
        2. Expected number of features and matching dimensions.
        3. Matching original feature columns and column order.
        4. Finite values and absence of missing values.
        5. Approximate zero mean and unit standard deviation
           on the scaled training features.
        6. Alignment between feature rows and optional targets.
        7. Binary target labels when targets are supplied.
        8. Validation/test feature partitions are not empty.

    Note:
        Mean and standard-deviation checks apply to X_train_scaled.
        Validation and test sets are transformed using the same scaler,
        so their means and standard deviations need not be zero and one.

    Returns:
        dict containing overall status, individual checks and metrics.

    Raises:
        ValueError if a critical validation check fails.
    """

    checks = {}
    metrics = {}

    # --------------------------------------------------
    # 1. Validate original feature DataFrames
    # --------------------------------------------------
    original_sets = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
    }

    for name, X in original_sets.items():
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame.")

        if X.empty:
            raise ValueError(f"{name} is empty.")

        if not X.columns.is_unique:
            raise ValueError(f"{name} contains duplicate column names.")

    feature_columns = list(X_train.columns)

    for name, X in original_sets.items():
        if list(X.columns) != feature_columns:
            raise ValueError(
                f"{name} feature names/order do not match X_train."
            )

    checks["original_feature_columns_match"] = True

    # --------------------------------------------------
    # 2. Validate scaled arrays
    # --------------------------------------------------
    scaled_sets = {
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
    }

    for name, X_scaled in scaled_sets.items():
        array = np.asarray(X_scaled)

        if array.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional.")

        if array.shape[0] == 0:
            raise ValueError(f"{name} has no rows.")

        if array.shape[1] != expected_feature_count:
            raise ValueError(
                f"{name} has {array.shape[1]} features; "
                f"expected {expected_feature_count}."
            )

        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"{name} contains non-numeric values.")

        if not np.isfinite(array).all():
            raise ValueError(
                f"{name} contains NaN or infinite values."
            )

        metrics[f"{name}_shape"] = array.shape

    checks["scaled_arrays_are_numeric_finite_and_nonempty"] = True

    # --------------------------------------------------
    # 3. Check row counts against original partitions
    # --------------------------------------------------
    for name, X, X_scaled in [
        ("train", X_train, X_train_scaled),
        ("validation", X_val, X_val_scaled),
        ("test", X_test, X_test_scaled),
    ]:
        if len(X) != len(X_scaled):
            raise ValueError(
                f"{name}: original and scaled row counts differ."
            )

    checks["scaled_row_counts_match"] = True

    for name, X in original_sets.items():
        if X.shape[1] != expected_feature_count:
            raise ValueError(
                f"{name} has {X.shape[1]} features; "
                f"expected {expected_feature_count}."
            )

    checks["expected_feature_count_matches"] = True

    # --------------------------------------------------
    # 4. Check training standardization
    # --------------------------------------------------
    train_array = np.asarray(X_train_scaled, dtype=float)

    feature_means = train_array.mean(axis=0)
    feature_stds = train_array.std(axis=0, ddof=0)

    max_abs_mean = float(np.abs(feature_means).max())
    max_std_deviation = float(np.abs(feature_stds - 1.0).max())
    min_feature_std = float(feature_stds.min())

    metrics["maximum_absolute_training_mean"] = max_abs_mean
    metrics["maximum_absolute_training_std_deviation_from_one"] = (
        max_std_deviation
    )
    metrics["minimum_training_feature_std"] = min_feature_std

    checks["training_mean_approximately_zero"] = (
        max_abs_mean <= mean_tolerance
    )

    # Constant features cannot have unit standard deviation.
    constant_feature_indices = np.flatnonzero(feature_stds == 0).tolist()
    metrics["constant_scaled_feature_indices"] = constant_feature_indices

    checks["training_std_approximately_one"] = (
        max_std_deviation <= std_tolerance
    )

    # --------------------------------------------------
    # 5. Validate optional targets
    # --------------------------------------------------
    target_sets = {
        "train": (y_train, X_train_scaled),
        "validation": (y_val, X_val_scaled),
        "test": (y_test, X_test_scaled),
    }

    for name, (y, X_scaled) in target_sets.items():
        if y is None:
            continue

        target = np.asarray(y).reshape(-1)

        if len(target) != len(X_scaled):
            raise ValueError(
                f"y_{name} has {len(target)} labels, but "
                f"the corresponding feature set has {len(X_scaled)} rows."
            )

        if pd.isna(target).any():
            raise ValueError(f"y_{name} contains missing labels.")

        unique_values = np.unique(target)

        metrics[f"y_{name}_unique_values"] = unique_values.tolist()

        if not np.isin(unique_values, [0, 1]).all():
            raise ValueError(
                f"y_{name} is not binary encoded as 0 and 1. "
                f"Observed values: {unique_values.tolist()}"
            )

    checks["provided_targets_are_aligned_and_binary"] = True

    # --------------------------------------------------
    # 6. Overall result and report
    # --------------------------------------------------
    overall_passed = all(checks.values())

    report = {
        "passed": overall_passed,
        "checks": checks,
        "metrics": metrics,
        "feature_count": expected_feature_count,
        "feature_columns": feature_columns,
    }

    if display_report:
        print("=" * 60)
        print("CHURN ANN PREPROCESSING VALIDATION")
        print("=" * 60)

        for check_name, passed in checks.items():
            status = "PASS" if passed else "FAIL"
            print(f"{status}: {check_name}")

        print("\nNumerical metrics:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value}")

        print("\nOverall result:", "PASS" if overall_passed else "FAIL")

    if not overall_passed:
        failed_checks = [
            name for name, passed in checks.items() if not passed
        ]
        raise ValueError(
            "Scaled churn feature validation failed: "
            + ", ".join(failed_checks)
        )

    return report


#-----  3. Build DL model ----------------------------------------------------
def build_deep_learning_model(
    model_type: str,
    input_shape: int,
    task_type: str,
    hidden_units=(64, 32, 16),
    dropout_rate: float = 0.30,
    learning_rate: float = 0.001,
    activation: str = "relu",
):
    """
    Build and compile a configurable deep-learning model.

    Supported models:
        - ann: ANN for tabular binary classification.
        - dnn: Dense neural network for tabular regression.

    Parameters:
        model_type: Architecture identifier ("ann" or "dnn").
        input_shape: Number of input features.
        task_type: "classification" or "regression".
        hidden_units: Neuron counts for successive Dense layers.
        dropout_rate: Dropout probability applied after hidden layers.
        learning_rate: Adam optimizer learning rate.
        activation: Hidden-layer activation function.

    Returns:
        A compiled TensorFlow/Keras model.
    """

    model_type = model_type.lower().strip()
    task_type = task_type.lower().strip()
    hidden_units = tuple(hidden_units)

    if model_type not in {"ann", "dnn"}:
        raise ValueError(
            f"Unsupported model_type: {model_type}. "
            "Currently supported: 'ann', 'dnn'."
        )

    if task_type not in {"classification", "regression"}:
        raise ValueError(
            "task_type must be 'classification' or 'regression'."
        )

    if model_type == "ann" and task_type != "classification":
        raise ValueError("The current ANN configuration is for classification.")

    if model_type == "dnn" and task_type != "regression":
        raise ValueError("The current DNN configuration is for regression.")

    if input_shape <= 0:
        raise ValueError("input_shape must be positive.")

    if not hidden_units or any(
        not isinstance(units, int) or units <= 0
        for units in hidden_units
    ):
        raise ValueError(
            "hidden_units must contain positive integers, "
            "for example (64, 32, 16)."
        )

    if not 0 <= dropout_rate < 1:
        raise ValueError("dropout_rate must be in [0, 1).")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    model = Sequential()
    model.add(Input(shape=(input_shape,)))

    for units in hidden_units:
        model.add(Dense(units, activation=activation))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    if task_type == "classification":
        model.add(Dense(1, activation="sigmoid"))
        loss = "binary_crossentropy"

        metrics = [
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]

    else:
        model.add(Dense(1, activation="linear"))
        loss = "mse"

        metrics = [
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ]

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics,
    )

    return model


#----- 4. Train Deep learning model --------------------------------
def train_deep_learning_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=100,
    batch_size=32,
    patience=10,
    min_delta=0.001,
    random_seed=42,
    verbose=1,
):
    """
    Train a compiled Keras model for binary classification or regression.

    Automatically infers the task from the model's compiled loss.

    Returns
    -------
    dict
        Trained model, task type, training history, completed epochs,
        best epoch, best validation loss, and early-stopping callback.
    """
    import random
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping

    # 1. Infer task from the compiled model
    loss_name = getattr(model.loss, "name", model.loss)
    if not isinstance(loss_name, str):
        loss_name = getattr(loss_name, "__name__", str(loss_name))

    loss_name = loss_name.lower().replace("_", "")

    classification_losses = {
        "binarycrossentropy",
        "binaryfocalcrossentropy",
    }
    regression_losses = {
        "mse",
        "meansquarederror",
        "mae",
        "meanabsoluteerror",
        "huber",
        "logcosh",
    }

    if loss_name in classification_losses:
        task_type = "classification"
    elif loss_name in regression_losses:
        task_type = "regression"
    else:
        raise ValueError(
            f"Cannot infer task from loss '{loss_name}'. "
            "Supported losses are binary classification losses "
            "and common regression losses."
        )

    # 2. Validate training parameters
    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not isinstance(patience, int) or patience < 0:
        raise ValueError("patience must be a non-negative integer.")
    if min_delta < 0:
        raise ValueError("min_delta must be non-negative.")
    if verbose not in (0, 1, 2):
        raise ValueError("verbose must be 0, 1, or 2.")

    # 3. Convert data to NumPy arrays
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
    y_val = np.asarray(y_val, dtype=np.float32).reshape(-1)

    # 4. Validate feature matrices
    for name, X in [("X_train", X_train), ("X_val", X_val)]:
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(
                f"{name} must be a non-empty 2D feature matrix."
            )
        if not np.isfinite(X).all():
            raise ValueError(
                f"{name} contains NaN or infinite values."
            )

    # 5. Validate targets and row alignment
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train row counts do not match.")
    if len(X_val) != len(y_val):
        raise ValueError("X_val and y_val row counts do not match.")
    if len(y_train) == 0 or len(y_val) == 0:
        raise ValueError("Training and validation targets cannot be empty.")
    if not np.isfinite(y_train).all() or not np.isfinite(y_val).all():
        raise ValueError("Targets contain NaN or infinite values.")

    if task_type == "classification":
        if not np.isin(y_train, [0, 1]).all():
            raise ValueError("Classification targets must contain only 0 and 1.")
        if not np.isin(y_val, [0, 1]).all():
            raise ValueError("Validation targets must contain only 0 and 1.")

        output_activation = getattr(
            model.layers[-1], "activation", None
        )
        if (
            output_activation is None
            or output_activation.__name__ != "sigmoid"
        ):
            raise ValueError(
                "Binary classification requires a sigmoid output layer."
            )

    # 6. Validate model dimensions
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError(
            "Training and validation feature counts do not match."
        )
    if model.input_shape[-1] != X_train.shape[1]:
        raise ValueError(
            "Feature count does not match the model input shape."
        )
    if model.output_shape[-1] != 1:
        raise ValueError(
            "This function expects a model with one output neuron."
        )

    # 7. Set random seeds
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    # 8. Configure early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=True,
        verbose=1 if verbose else 0,
    )

    # 9. Train the model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        shuffle=True,
        verbose=verbose,
    )

    # 10. Summarize training results
    history_dict = history.history
    best_epoch = int(np.argmin(history_dict["val_loss"]) + 1)
    best_val_loss = float(np.min(history_dict["val_loss"]))

    return {
        "model": model,
        "task_type": task_type,
        "history": history,
        "history_dict": history_dict,
        "epochs_completed": len(history.epoch),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "early_stopping": early_stopping,
    }


#---- 5. Tunning th Deep Learning model --------------------------------

def tune_deep_learning_model(
    model_type,
    X_train,
    y_train,
    X_val,
    y_val,
    parameter_grid=None,
    scoring=None,
    epochs=100,
    batch_size=32,
    patience=10,
    min_delta=0.001,
    random_seed=42,
    verbose=0,
):
    """
    Tune ANN classification or DNN regression models.

    Uses the existing build_deep_learning_model(),
    train_deep_learning_model(), and
    evaluate_deep_learning_model() functions.

    Returns the best model, parameters, score, training result,
    validation evaluation, and a table of all trials.
    """
    import itertools
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    # 1. Infer task from the requested architecture
    model_type = model_type.lower().strip()

    model_task_mapping = {
        "ann": "classification",
        "dnn": "regression",
    }

    if model_type not in model_task_mapping:
        raise ValueError("model_type must be 'ann' or 'dnn'.")

    task_type = model_task_mapping[model_type]

    # 2. Choose the default selection metric
    if scoring is None:
        scoring = "auc" if task_type == "classification" else "mae"

    allowed_scores = {
        "classification": {
            "auc", "accuracy", "precision", "recall", "f1"
        },
        "regression": {"mae", "rmse", "r2"},
    }

    if scoring not in allowed_scores[task_type]:
        raise ValueError(
            f"Invalid scoring '{scoring}' for {task_type}. "
            f"Choose from {sorted(allowed_scores[task_type])}."
        )

    # 3. Define a default search space
    if parameter_grid is None:
        parameter_grid = {
            "hidden_units": [
                (64, 32, 16),
                (128, 64, 32),
            ],
            "dropout_rate": [0.20, 0.30],
            "learning_rate": [0.001, 0.0005],
            "activation": ["relu"],
        }

    allowed_parameters = {
        "hidden_units",
        "dropout_rate",
        "learning_rate",
        "activation",
        "batch_size",
    }

    unknown_parameters = set(parameter_grid) - allowed_parameters

    if unknown_parameters:
        raise ValueError(
            "Unsupported parameter(s): "
            f"{sorted(unknown_parameters)}. "
            "Do not include model_type or task_type in parameter_grid."
        )

    if not parameter_grid or any(
        not isinstance(values, (list, tuple)) or len(values) == 0
        for values in parameter_grid.values()
    ):
        raise ValueError(
            "parameter_grid must contain non-empty lists or tuples."
        )

    # 4. Validate basic dataset dimensions
    X_train_array = np.asarray(X_train)
    X_val_array = np.asarray(X_val)

    if X_train_array.ndim != 2 or X_val_array.ndim != 2:
        raise ValueError("Training and validation features must be 2D.")

    if X_train_array.shape[1] != X_val_array.shape[1]:
        raise ValueError(
            "Training and validation feature counts do not match."
        )

    # 5. Generate combinations
    parameter_names = list(parameter_grid.keys())

    combinations = list(
        itertools.product(
            *(parameter_grid[name] for name in parameter_names)
        )
    )

    maximize_score = scoring in {
        "auc", "accuracy", "precision", "recall", "f1", "r2"
    }

    best_score = -np.inf if maximize_score else np.inf
    best_model = None
    best_params = None
    best_training_result = None
    best_evaluation = None
    trial_results = []

    # 6. Train each candidate independently
    for trial_number, combination in enumerate(combinations, start=1):

        params = dict(zip(parameter_names, combination))

        if verbose:
            print(
                f"\nTrial {trial_number}/{len(combinations)}"
                f" | model_type={model_type}"
                f" | task_type={task_type}"
            )
            print("Parameters:", params)

        # Clear previous Keras session and set trial seed
        tf.keras.backend.clear_session()
        trial_seed = random_seed + trial_number
        tf.keras.utils.set_random_seed(trial_seed)

        # IMPORTANT: Pass both required arguments from your builder
        model = build_deep_learning_model(
            model_type=model_type,
            input_shape=X_train_array.shape[1],
            task_type=task_type,
            **params,
        )

        # Train using the existing universal training function
        training_result = train_deep_learning_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            min_delta=min_delta,
            random_seed=trial_seed,
            verbose=verbose,
        )

        # Evaluate on validation data only
        evaluation = evaluate_deep_learning_model(
            model=training_result["model"],
            X=X_val,
            y=y_val,
            dataset_name="Tuning Validation",
            display_report=False,
        )

        metrics = evaluation["metrics"]

        if scoring == "f1":
            precision = metrics["precision"]
            recall = metrics["recall"]

            score = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0.0
            )
        else:
            score = metrics.get(scoring)

        if score is None or not np.isfinite(score):
            raise ValueError(
                f"Trial {trial_number} produced an invalid "
                f"{scoring} score."
            )

        trial_results.append({
            "trial": trial_number,
            **params,
            "scoring": scoring,
            "score": float(score),
            "best_epoch": training_result["best_epoch"],
            "epochs_completed": training_result["epochs_completed"],
            **metrics,
        })

        is_better = (
            score > best_score if maximize_score else score < best_score
        )

        if is_better:
            best_score = float(score)
            best_params = params.copy()
            best_model = training_result["model"]
            best_training_result = training_result
            best_evaluation = evaluation

        if verbose:
            print(f"Validation {scoring}: {score:.5f}")

    # 7. Rank and return all trial results
    results_df = pd.DataFrame(trial_results)

    results_df = results_df.sort_values(
        "score",
        ascending=not maximize_score,
    ).reset_index(drop=True)

    return {
        "model_type": model_type,
        "task_type": task_type,
        "scoring": scoring,
        "best_model": best_model,
        "best_params": best_params,
        "best_score": best_score,
        "best_training_result": best_training_result,
        "best_evaluation": best_evaluation,
        "results": results_df,
        "trials_completed": len(trial_results),
    }

#----- 6. Evaluating Deel learning function-----
def evaluate_deep_learning_model(
    model,
    X,
    y,
    dataset_name="Validation",
    threshold=0.5,
    display_report=True,
):
    """
    Evaluate a compiled Keras model for binary classification
    or regression.

    The task is inferred from the model's compiled loss.

    Returns
    -------
    dict
        Task type, evaluation metrics, actual values,
        and predictions.
    """
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
        classification_report,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    # 1. Infer task from the compiled loss
    loss_name = getattr(model.loss, "name", model.loss)
    if not isinstance(loss_name, str):
        loss_name = getattr(loss_name, "__name__", str(loss_name))
    loss_name = loss_name.lower().replace("_", "")

    classification_losses = {
        "binarycrossentropy",
        "binaryfocalcrossentropy",
    }
    regression_losses = {
        "mse",
        "meansquarederror",
        "mae",
        "meanabsoluteerror",
        "huber",
        "logcosh",
    }

    if loss_name in classification_losses:
        task_type = "classification"
    elif loss_name in regression_losses:
        task_type = "regression"
    else:
        raise ValueError(
            f"Cannot infer task from loss '{loss_name}'."
        )

    # 2. Validate inputs
    X = np.asarray(X, dtype=np.float32)
    y_true = np.asarray(y).reshape(-1)

    if X.ndim != 2 or len(X) == 0:
        raise ValueError("X must be a non-empty 2D feature matrix.")

    if len(X) != len(y_true):
        raise ValueError("X and y row counts do not match.")

    if not np.isfinite(X).all() or not np.isfinite(y_true).all():
        raise ValueError("X or y contains NaN or infinite values.")

    if X.shape[1] != model.input_shape[-1]:
        raise ValueError("Feature count does not match model input.")

    if model.output_shape[-1] != 1:
        raise ValueError("This function expects one output neuron.")

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    # 3. Generate predictions
    predictions = np.asarray(
        model.predict(X, verbose=0)
    ).reshape(-1)

    if len(predictions) != len(y_true):
        raise ValueError("Prediction and target lengths do not match.")

    if not np.isfinite(predictions).all():
        raise ValueError("Model generated non-finite predictions.")

    # 4. Calculate task-specific metrics
    if task_type == "classification":
        if not np.isin(y_true, [0, 1]).all():
            raise ValueError("Classification targets must be binary: 0 or 1.")

        if not np.all((predictions >= 0) & (predictions <= 1)):
            raise ValueError(
                "Classification predictions must be probabilities "
                "between 0 and 1. Check the sigmoid output."
            )

        y_true = y_true.astype(int)
        y_pred = (predictions >= threshold).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(y_true, y_pred, zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, zero_division=0)
            ),
            "f1": float(
                f1_score(y_true, y_pred, zero_division=0)
            ),
            "confusion_matrix": confusion_matrix(
                y_true, y_pred, labels=[0, 1]
            ).tolist(),
            "threshold": float(threshold),
        }

        # AUC requires both classes to be present.
        if len(np.unique(y_true)) == 2:
            metrics["auc"] = float(
                roc_auc_score(y_true, predictions)
            )
        else:
            metrics["auc"] = None

        if display_report:
            print(f"\n{dataset_name} Classification Results")
            print("-" * 45)
            for name, value in metrics.items():
                print(f"{name}: {value}")

            print("\nClassification Report:")
            print(
                classification_report(
                    y_true,
                    y_pred,
                    labels=[0, 1],
                    target_names=["Not Churned", "Churned"],
                    zero_division=0,
                )
            )

    else:
        y_true = y_true.astype(float)

        metrics = {
            "mae": float(mean_absolute_error(y_true, predictions)),
            "rmse": float(
                np.sqrt(mean_squared_error(y_true, predictions))
            ),
            "r2": float(r2_score(y_true, predictions)),
        }

        if display_report:
            print(f"\n{dataset_name} Regression Results")
            print("-" * 45)
            for name, value in metrics.items():
                print(f"{name}: {value:.6f}")

    return {
        "task_type": task_type,
        "dataset_name": dataset_name,
        "metrics": metrics,
        "y_true": y_true,
        "predictions": predictions,
    }


#---- 7. visualization --------------------------------
def visualize_deep_learning_results(
    evaluation_result,
    history=None,
    show_training_curves=True,
):
    """
    Visualize training history and evaluation results.

    Parameters
    ----------
    evaluation_result : dict
        Output of evaluate_deep_learning_model().
    history : dict, History, or None
        Training history from train_deep_learning_model().
        Can be omitted when evaluating an already-trained model.
    show_training_curves : bool
        Whether to plot available training and validation curves.

    Returns
    -------
    None
        Displays Matplotlib figures.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, ConfusionMatrixDisplay

    task_type = evaluation_result["task_type"]
    metrics = evaluation_result["metrics"]
    y_true = np.asarray(evaluation_result["y_true"])
    predictions = np.asarray(evaluation_result["predictions"])

    # 1. Extract training history, if available
    history_dict = None

    if history is not None:
        if hasattr(history, "history"):
            history_dict = history.history
        elif isinstance(history, dict):
            history_dict = history.get("history_dict", history)
        else:
            raise ValueError(
                "history must be a Keras History object, "
                "a history dictionary, or None."
            )

    # 2. Plot available training curves
    if show_training_curves and history_dict:
        curve_groups = [
            ("loss", "val_loss", "Loss"),
        ]

        if task_type == "classification":
            if "auc" in history_dict:
                curve_groups.append(("auc", "val_auc", "AUC"))
            if "accuracy" in history_dict:
                curve_groups.append(
                    ("accuracy", "val_accuracy", "Accuracy")
                )
        else:
            if "mae" in history_dict:
                curve_groups.append(("mae", "val_mae", "MAE"))
            if "rmse" in history_dict:
                curve_groups.append(("rmse", "val_rmse", "RMSE"))

        for train_key, val_key, title in curve_groups:
            if train_key not in history_dict:
                continue

            plt.figure(figsize=(9, 5))
            plt.plot(
                history_dict[train_key],
                label=f"Training {title}",
            )

            if val_key in history_dict:
                plt.plot(
                    history_dict[val_key],
                    label=f"Validation {title}",
                )

            plt.xlabel("Epoch")
            plt.ylabel(title)
            plt.title(f"Deep Learning: {title} by Epoch")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    # 3. Classification plots
    if task_type == "classification":
        y_pred = (predictions >= metrics["threshold"]).astype(int)

        cm = np.asarray(metrics["confusion_matrix"])

        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Not Churned", "Churned"],
        ).plot(ax=ax, values_format="d", colorbar=False)
        ax.set_title("Churn Classification: Confusion Matrix")
        fig.tight_layout()
        plt.show()

        if metrics.get("auc") is not None:
            fpr, tpr, _ = roc_curve(y_true, predictions)

            plt.figure(figsize=(7, 5))
            plt.plot(
                fpr,
                tpr,
                label=f"AUC = {metrics['auc']:.3f}",
            )
            plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Churn Classification: ROC Curve")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    # 4. Regression plots
    else:
        residuals = y_true - predictions

        plt.figure(figsize=(7, 6))
        plt.scatter(y_true, predictions, alpha=0.5)
        lower = min(float(y_true.min()), float(predictions.min()))
        upper = max(float(y_true.max()), float(predictions.max()))
        plt.plot(
            [lower, upper],
            [lower, upper],
            linestyle="--",
            label="Perfect prediction",
        )
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Regression: Actual vs Predicted")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(7, 5))
        plt.scatter(predictions, residuals, alpha=0.5)
        plt.axhline(0, linestyle="--")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals (Actual - Predicted)")
        plt.title("Regression: Residual Analysis")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()



################################################################
#===============================================================
#------------------------ AUTO ENCODER --------------------------------------
#===============================================================
################################################################

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
#  1. build Autoencoder --------------------------------
def build_autoencoder(
    input_dim: int,
    encoding_dim: int = 8,
    hidden_dim: int = 16,
    learning_rate: float = 0.001,
) -> Model:
    """
    Build a dense autoencoder for anomaly detection.
    """

    if input_dim <= 0:
        raise ValueError("input_dim must be positive.")

    if not 0 < encoding_dim < hidden_dim < input_dim:
        raise ValueError(
            "Require encoding_dim < hidden_dim < input_dim."
        )

    inputs = Input(shape=(input_dim,), name="customer_features")

    # Encoder
    encoded = Dense(
        hidden_dim,
        activation="relu",
        name="encoder_hidden",
    )(inputs)

    latent = Dense(
        encoding_dim,
        activation="relu",
        name="latent_representation",
    )(encoded)

    # Decoder
    decoded = Dense(
        hidden_dim,
        activation="relu",
        name="decoder_hidden",
    )(latent)

    outputs = Dense(
        input_dim,
        activation="linear",
        name="reconstructed_features",
    )(decoded)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="customer_anomaly_autoencoder",
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
    )

    return model


#  2. train autoencoder --------------------------------
def train_autoencoder(
    model: Model,
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 10,
):
    """
    Train the autoencoder using input features as targets.
    """

    if list(X_train.columns) != list(X_validation.columns):
        raise ValueError(
            "Training and validation features must match."
        )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train.to_numpy(dtype="float32"),
        X_train.to_numpy(dtype="float32"),
        validation_data=(
            X_validation.to_numpy(dtype="float32"),
            X_validation.to_numpy(dtype="float32"),
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        shuffle=True,
        verbose=1,
    )

    return history


# 3. calculate reconstruct erros
def calculate_reconstruction_errors(
    model,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate per-customer reconstruction errors.

    The anomaly score is the mean squared reconstruction
    error across all input features.
    """

    if X.empty:
        raise ValueError("Input feature dataset is empty.")

    if not np.isfinite(
        X.to_numpy(dtype=np.float64)
    ).all():
        raise ValueError(
            "Input features contain missing or infinite values."
        )

    reconstructed = model.predict(
        X.to_numpy(dtype="float32"),
        verbose=0,
    )

    original = X.to_numpy(dtype="float32")

    if reconstructed.shape != original.shape:
        raise ValueError(
            "Reconstructed shape does not match input shape."
        )

    feature_errors = np.square(
        original - reconstructed
    )

    mean_squared_error = feature_errors.mean(axis=1)

    return pd.DataFrame(
        {
            "reconstruction_error": mean_squared_error,
        },
        index=X.index,
    )


# 4. Validate anamoly threashold
def validate_anomaly_thresholds(
    train_errors: pd.DataFrame,
    validation_errors: pd.DataFrame,
    percentiles: tuple[float, ...] = (95, 97.5, 99),
) -> pd.DataFrame:
    """
    Compare candidate thresholds derived from training errors.

    Validation errors are used to measure how many validation
    customers exceed each threshold, not to claim that those
    customers are confirmed anomalies.
    """

    required_column = "reconstruction_error"

    for name, errors in (
        ("train_errors", train_errors),
        ("validation_errors", validation_errors),
    ):
        if required_column not in errors.columns:
            raise ValueError(
                f"{name} must contain '{required_column}'."
            )

        values = errors[required_column]

        if values.empty or not values.notna().all():
            raise ValueError(
                f"{name} contains empty or missing errors."
            )

        if (values < 0).any():
            raise ValueError(
                f"{name} contains negative reconstruction errors."
            )

        if not values.map(
            lambda value: value < float("inf")
        ).all():
            raise ValueError(
                f"{name} contains infinite errors."
            )

    if any(
        percentile <= 0 or percentile >= 100
        for percentile in percentiles
    ):
        raise ValueError(
            "Percentiles must be between 0 and 100."
        )

    train_values = train_errors[
        required_column
    ]

    validation_values = validation_errors[
        required_column
    ]

    results = []

    for percentile in percentiles:
        threshold = float(
            train_values.quantile(percentile / 100)
        )

        train_flagged = train_values > threshold
        validation_flagged = validation_values > threshold

        results.append({
            "percentile": percentile,
            "threshold": threshold,
            "train_flagged_count": int(train_flagged.sum()),
            "train_flagged_pct": (
                train_flagged.mean() * 100
            ),
            "validation_flagged_count": int(
                validation_flagged.sum()
            ),
            "validation_flagged_pct": (
                validation_flagged.mean() * 100
            ),
        })

    return pd.DataFrame(results)


# 5. Predict customer anamolies


def predict_customer_anomalies(
    model,
    X_scaled: pd.DataFrame,
    customer_ids: pd.Series,
    threshold: float,
) -> pd.DataFrame:
    """
    Flag customers whose reconstruction errors exceed
    the supplied anomaly-review threshold.
    """

    if len(X_scaled) != len(customer_ids):
        raise ValueError(
            "Customer IDs and features must have equal lengths."
        )

    if not X_scaled.index.equals(customer_ids.index):
        raise ValueError(
            "Customer IDs and features must have matching indexes."
        )

    if not np.isfinite(
        X_scaled.to_numpy(dtype=np.float64)
    ).all():
        raise ValueError(
            "Scaled features contain missing or infinite values."
        )

    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError(
            "Threshold must be a finite, non-negative number."
        )

    errors = calculate_reconstruction_errors(
        model=model,
        X=X_scaled,
    )

    results = pd.DataFrame(
        {
            "Customer ID": customer_ids.to_numpy(),
            "reconstruction_error": (
                errors["reconstruction_error"].to_numpy()
            ),
        },
        index=X_scaled.index,
    )

    results["anomaly_review_flag"] = (
        results["reconstruction_error"] > threshold
    )

    results["review_status"] = np.where(
        results["anomaly_review_flag"],
        "Flagged for review",
        "Not flagged",
    )

    return results.sort_values(
        "reconstruction_error",
        ascending=False,
    ).reset_index(drop=True)
