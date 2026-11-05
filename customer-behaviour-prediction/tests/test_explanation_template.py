
import pandas as pd
import pytest

from src.explainability.explanation_template import generate_churn_explanation


def make_shap_data():
    """Create sample SHAP contributions for testing."""
    return pd.DataFrame({
        "Feature": [
            "Recency",
            "Frequency",
            "Engagement_Score",
            "Average_Order_Spend",
        ],
        "SHAP_Value": [0.8, -0.6, 0.4, -0.2],
    })


def test_churn_explanation_shap_high_risk():
    """Explain a churn prediction with high risk."""
    result = generate_churn_explanation(
        prediction=1,
        probabilities={"0": 0.15, "1": 0.85},
        contributions=make_shap_data(),
        customer_idx=12,
        top_n=2,
    )

    assert result["prediction"] == 1
    assert result["prediction_label"] == "Churn"
    assert result["churn_probability"] == 0.85
    assert result["risk_level"] == "High"
    assert result["explanation_method"] == "SHAP"
    assert result["customer_idx"] == 12
    assert result["supporting_features"] == ["Recency", "Engagement_Score"]
    assert result["opposing_features"] == ["Frequency", "Average_Order_Spend"]
    assert "85.0%" in result["summary"]
    assert "Suggested next step" in result["explanation_text"]


def test_churn_explanation_moderate_and_lower_risk():
    """Test the moderate and lower churn risk thresholds."""
    data = make_shap_data()

    moderate = generate_churn_explanation(
        1, {"1": 0.55}, data
    )
    lower = generate_churn_explanation(
        1, {"1": 0.25}, data
    )

    assert moderate["risk_level"] == "Moderate"
    assert lower["risk_level"] == "Lower"


def test_no_churn_explanation():
    """Explain a non-churn prediction and reverse contribution direction."""
    result = generate_churn_explanation(
        prediction=0,
        probabilities={"0": 0.8, "1": 0.2},
        contributions=make_shap_data(),
        top_n=2,
    )

    assert result["prediction_label"] == "No Churn"
    assert result["risk_level"] == "Lower"
    assert result["supporting_features"] == ["Frequency", "Average_Order_Spend"]
    assert result["opposing_features"] == ["Recency", "Engagement_Score"]
    assert "monitoring customer behaviour" in result["recommendation"]


def test_no_churn_elevated_risk():
    """A non-churn prediction can still have elevated churn probability."""
    result = generate_churn_explanation(
        prediction=0,
        probabilities={"1": 0.65},
        contributions=make_shap_data(),
    )

    assert result["risk_level"] == "Elevated"


def test_lime_explanation_feature_condition():
    """Support the LIME Feature_Condition column."""
    contributions = pd.DataFrame({
        "Feature_Condition": [
            "Recency > 90",
            "Frequency <= 2",
            "Engagement_Score < 0.3",
        ],
        "LIME_Weight": [0.7, -0.5, 0.2],
    })

    result = generate_churn_explanation(
        1, {"1": 0.75}, contributions, top_n=2
    )

    assert result["explanation_method"] == "LIME"
    assert result["supporting_features"] == [
        "Recency > 90",
        "Engagement_Score < 0.3",
    ]
    assert result["opposing_features"] == ["Frequency <= 2"]


def test_lime_explanation_feature_column():
    """Support LIME data using the generic Feature column."""
    contributions = pd.DataFrame({
        "Feature": ["Recency", "Frequency"],
        "LIME_Weight": [0.6, -0.4],
    })

    result = generate_churn_explanation(
        1, {"1": 0.8}, contributions
    )

    assert result["explanation_method"] == "LIME"
    assert result["supporting_features"] == ["Recency"]
    assert result["opposing_features"] == ["Frequency"]


@pytest.mark.parametrize("prediction", [-1, 2, None, "1"])
def test_invalid_prediction(prediction):
    """Reject unsupported prediction classes."""
    with pytest.raises(ValueError, match="prediction must be 0 or 1"):
        generate_churn_explanation(
            prediction, {"1": 0.5}, make_shap_data()
        )


@pytest.mark.parametrize("probabilities", [None, [], "0.5"])
def test_invalid_probabilities_type(probabilities):
    """Require probabilities to be a dictionary."""
    with pytest.raises(TypeError, match="probabilities must be a dictionary"):
        generate_churn_explanation(
            1, probabilities, make_shap_data()
        )


@pytest.mark.parametrize("contributions", [None, [], {"Feature": "Recency"}])
def test_invalid_contributions_type(contributions):
    """Require contributions to be a pandas DataFrame."""
    with pytest.raises(TypeError, match="contributions must be a DataFrame"):
        generate_churn_explanation(
            1, {"1": 0.5}, contributions
        )


def test_empty_contributions():
    """Reject an empty contributions DataFrame."""
    with pytest.raises(ValueError, match="contributions cannot be empty"):
        generate_churn_explanation(
            1, {"1": 0.5}, pd.DataFrame()
        )


@pytest.mark.parametrize("top_n", [0, -1])
def test_invalid_top_n(top_n):
    """Require top_n to be positive."""
    with pytest.raises(ValueError, match="top_n must be positive"):
        generate_churn_explanation(
            1, {"1": 0.5}, make_shap_data(), top_n=top_n
        )


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_churn_probability(probability):
    """Reject probabilities outside the valid range or non-finite values."""
    with pytest.raises(ValueError):
        generate_churn_explanation(
            1, {"1": probability}, make_shap_data()
        )


def test_missing_contribution_column():
    """Reject data with neither SHAP_Value nor LIME_Weight."""
    contributions = pd.DataFrame({"Feature": ["Recency"]})

    with pytest.raises(
        ValueError,
        match="Contributions must contain SHAP_Value or LIME_Weight",
    ):
        generate_churn_explanation(1, {"1": 0.5}, contributions)


def test_missing_shap_feature_column():
    """Reject SHAP contributions without a Feature column."""
    contributions = pd.DataFrame({"SHAP_Value": [0.5]})

    with pytest.raises(ValueError, match="Missing feature column: Feature"):
        generate_churn_explanation(1, {"1": 0.5}, contributions)


def test_missing_lime_feature_column():
    """Reject LIME contributions without either supported feature column."""
    contributions = pd.DataFrame({"LIME_Weight": [0.5]})

    with pytest.raises(
        ValueError,
        match="Missing feature column: Feature",
    ):
        generate_churn_explanation(1, {"1": 0.5}, contributions)


def test_non_numeric_contributions():
    """Reject contribution values that cannot be converted to numbers."""
    contributions = pd.DataFrame({
        "Feature": ["Recency"],
        "SHAP_Value": ["not-a-number"],
    })

    with pytest.raises(ValueError):
        generate_churn_explanation(1, {"1": 0.5}, contributions)


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_contributions(value):
    """Reject infinite and missing contribution values."""
    contributions = pd.DataFrame({
        "Feature": ["Recency"],
        "SHAP_Value": [value],
    })

    with pytest.raises(ValueError, match="Contribution values must be finite"):
        generate_churn_explanation(1, {"1": 0.5}, contributions)


def test_missing_churn_probability_defaults_to_zero():
    """Use zero when class-1 probability is not provided."""
    result = generate_churn_explanation(
        1, {"0": 1.0}, make_shap_data()
    )

    assert result["churn_probability"] == 0.0
    assert result["risk_level"] == "Lower"


def test_feature_names_are_formatted_in_explanation():
    """Convert underscores to spaces in readable explanation text."""
    result = generate_churn_explanation(
        1, {"1": 0.8}, make_shap_data()
    )

    assert "Engagement Score" in result["explanation_text"]
    assert "Engagement_Score" not in result["explanation_text"]


##################################################################################################################################
#----------------------------------------------- lime explainability -------------------------------------------------------------
##################################################################################################################################



import matplotlib
matplotlib.use("Agg")

from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.explainability.lime_explainer import explain_customer_with_lime
from src.explainability.lime_explainer import compare_lime_with_shap


def test_lime_explainability(tmp_path):
    """
    Test LIME explanation generation, validation, output paths,
    and comparison with SHAP without changing production code.
    """
    import numpy as np
    import pandas as pd
    import pytest
    import matplotlib.pyplot as plt
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    from src.explainability.lime_explainer import (
        explain_customer_with_lime,
        compare_lime_with_shap,
    )

    module = "src.explainability.lime_explainer"

    try:
        # ---------------------------------------------------------
        # 1. Prepare training data and mock classifier
        # ---------------------------------------------------------
        X_train = pd.DataFrame(
            {
                "Recency": [10, 20, 30, 40, 50, 60],
                "Frequency": [8, 7, 6, 5, 4, 3],
                "Monetary": [1000, 900, 800, 700, 600, 500],
            }
        )

        customer = pd.DataFrame(
            {
                "Recency": [25],
                "Frequency": [6],
                "Monetary": [850],
            }
        )

        feature_names = ["Recency", "Frequency", "Monetary"]

        model = MagicMock()
        model.classes_ = np.array([0, 1])
        model.n_features_in_ = 3
        model.predict_proba.return_value = np.array([[0.25, 0.75]])
        model.predict.return_value = np.array([1])

        fake_explanation = MagicMock()
        fake_explanation.as_list.return_value = [
            ("Recency > 20", 0.35),
            ("Frequency <= 6", -0.20),
            ("Monetary > 800", 0.10),
        ]

        fake_figure = MagicMock()
        fake_explanation.as_pyplot_figure.return_value = fake_figure

        # ---------------------------------------------------------
        # 2. Test successful LIME explanation
        # ---------------------------------------------------------
        with (
            patch(f"{module}.LimeTabularExplainer") as explainer_class,
            patch(f"{module}.plt.close") as close_mock,
        ):
            explainer_class.return_value.explain_instance.return_value = (
                fake_explanation
            )

            result = explain_customer_with_lime(
                model=model,
                X_train=X_train,
                customer_features=customer,
                feature_names=feature_names,
                num_features=3,
                num_samples=100,
                output_dir=tmp_path,
                show=False,
            )

            assert isinstance(result, dict)
            assert result["prediction"] == 1
            assert result["prediction_label"] == "Churn"

            assert result["class_probabilities"] == {
                "0": pytest.approx(0.25),
                "1": pytest.approx(0.75),
            }

            contributions = result["contributions"]

            assert isinstance(contributions, pd.DataFrame)
            assert {
                "Feature_Condition",
                "LIME_Weight",
                "Direction",
            }.issubset(contributions.columns)

            assert len(contributions) == 3
            assert contributions["LIME_Weight"].notna().all()
            assert set(contributions["Direction"]).issubset(
                {"Supports churn", "Opposes churn", "Neutral"}
            )

            explainer_class.return_value.explain_instance.assert_called_once()
            fake_explanation.as_pyplot_figure.assert_called()
            close_mock.assert_called()

        # ---------------------------------------------------------
        # 3. Verify returned output paths
        # ---------------------------------------------------------
        assert result["plot_path"] is not None
        assert result["csv_path"] is not None

        plot_path = Path(result["plot_path"])
        csv_path = Path(result["csv_path"])

        assert plot_path.name == "lime_customer_explanation.png"
        assert csv_path.name == "lime_customer_contributions.csv"

        # The figure is mocked, so do not require a physical PNG file.
        # Validate the returned paths and generated contribution data.

        # ---------------------------------------------------------
        # 4. Test show=True behavior
        # ---------------------------------------------------------
        with (
            patch(f"{module}.LimeTabularExplainer") as show_explainer_class,
            patch(f"{module}.plt.show") as show_mock,
        ):
            show_explainer_class.return_value.explain_instance.return_value = (
                fake_explanation
            )

            show_result = explain_customer_with_lime(
                model=model,
                X_train=X_train,
                customer_features=customer,
                feature_names=feature_names,
                num_features=2,
                num_samples=50,
                output_dir=None,
                show=True,
            )

            assert show_result["prediction"] == 1
            show_mock.assert_called_once()

        # ---------------------------------------------------------
        # 5. Test input validation
        # ---------------------------------------------------------
        with pytest.raises((TypeError, ValueError)):
            explain_customer_with_lime(
                model=model,
                X_train=None,
                customer_features=customer,
            )

        with pytest.raises((TypeError, ValueError)):
            explain_customer_with_lime(
                model=model,
                X_train=X_train.iloc[0:0],
                customer_features=customer,
            )

        multiple_customers = pd.concat(
            [customer, customer],
            ignore_index=True,
        )

        with pytest.raises((TypeError, ValueError)):
            explain_customer_with_lime(
                model=model,
                X_train=X_train,
                customer_features=multiple_customers,
            )

        model_without_proba = MagicMock(spec=["predict"])

        with pytest.raises((TypeError, ValueError, AttributeError)):
            explain_customer_with_lime(
                model=model_without_proba,
                X_train=X_train,
                customer_features=customer,
            )

        with pytest.raises((TypeError, ValueError)):
            explain_customer_with_lime(
                model=model,
                X_train=X_train,
                customer_features=customer,
                feature_names=["Recency", "Recency", "Monetary"],
            )

        missing_feature = customer.drop(columns=["Frequency"])

        with pytest.raises((TypeError, ValueError)):
            explain_customer_with_lime(
                model=model,
                X_train=X_train,
                customer_features=missing_feature,
            )

        with pytest.raises((TypeError, ValueError)):
            explain_customer_with_lime(
                model=model,
                X_train=X_train,
                customer_features=customer,
                num_features=0,
            )

        with pytest.raises((TypeError, ValueError)):
            explain_customer_with_lime(
                model=model,
                X_train=X_train,
                customer_features=customer,
                num_samples=0,
            )

        invalid_customer = customer.copy()
        invalid_customer["Recency"] = "unknown"

        with pytest.raises((TypeError, ValueError)):
            explain_customer_with_lime(
                model=model,
                X_train=X_train,
                customer_features=invalid_customer,
            )

        invalid_training = X_train.copy()
        invalid_training["Recency"] = invalid_training["Recency"].astype(float)
        invalid_training.loc[0, "Recency"] = np.inf

        with pytest.raises((TypeError, ValueError)):
            explain_customer_with_lime(
                model=model,
                X_train=invalid_training,
                customer_features=customer,
            )

        invalid_customer = customer.copy()
        invalid_customer["Recency"] = np.nan

        with pytest.raises((TypeError, ValueError)):
            explain_customer_with_lime(
                model=model,
                X_train=X_train,
                customer_features=invalid_customer,
            )

        wrong_feature_count_model = MagicMock()
        wrong_feature_count_model.classes_ = np.array([0, 1])
        wrong_feature_count_model.n_features_in_ = 4
        wrong_feature_count_model.predict_proba.return_value = np.array(
            [[0.25, 0.75]]
        )
        wrong_feature_count_model.predict.return_value = np.array([1])

        with pytest.raises((TypeError, ValueError)):
            explain_customer_with_lime(
                model=wrong_feature_count_model,
                X_train=X_train,
                customer_features=customer,
            )

        # ---------------------------------------------------------
        # 6. Test fallback when class 1 is unavailable
        # ---------------------------------------------------------
        fallback_model = MagicMock()
        fallback_model.classes_ = np.array([2, 3])
        fallback_model.n_features_in_ = 3
        fallback_model.predict_proba.return_value = np.array([[0.8, 0.2]])
        fallback_model.predict.return_value = np.array([2])

        fallback_explanation = MagicMock()
        fallback_explanation.as_list.return_value = [
            ("Recency > 20", 0.2)
        ]
        fallback_explanation.as_pyplot_figure.return_value = MagicMock()

        with patch(f"{module}.LimeTabularExplainer") as fallback_class:
            fallback_class.return_value.explain_instance.return_value = (
                fallback_explanation
            )

            fallback_result = explain_customer_with_lime(
                model=fallback_model,
                X_train=X_train,
                customer_features=customer,
                feature_names=feature_names,
                output_dir=None,
                show=False,
            )

            assert fallback_result["prediction"] == 2
            assert fallback_result["prediction_label"] == "No Churn"

        # ---------------------------------------------------------
        # 7. Compare LIME and SHAP contributions
        # ---------------------------------------------------------
        lime_result = {
            "contributions": pd.DataFrame(
                {
                    "Feature_Condition": [
                        "Recency > 20",
                        "Frequency <= 6",
                        "Monetary > 800",
                    ],
                    "LIME_Weight": [0.35, -0.20, 0.10],
                    "Direction": [
                        "Supports churn",
                        "Opposes churn",
                        "Supports churn",
                    ],
                }
            )
        }

        shap = MagicMock()
        shap.values = np.array([[0.4, -0.3, 0.1]])
        shap.feature_names = feature_names

        comparison = compare_lime_with_shap(
            lime_result=lime_result,
            shap_explanation=shap,
            customer_idx=0,
            top_n=3,
        )

        assert isinstance(comparison, pd.DataFrame)

        # Use the actual column name returned by the implementation.
        assert {
            "Feature",
            "SHAP_Value",
            "LIME_Weight",
            "Absolute_SHAP",
            "Absolute_LIME",
        }.issubset(comparison.columns)

        assert len(comparison) == 3
        assert set(comparison["Feature"]) == set(feature_names)

        shap_values_by_feature = comparison.set_index("Feature")["SHAP_Value"]

        assert shap_values_by_feature["Recency"] == pytest.approx(0.4)
        assert shap_values_by_feature["Frequency"] == pytest.approx(-0.3)
        assert shap_values_by_feature["Monetary"] == pytest.approx(0.1)

        lime_values_by_feature = comparison.set_index("Feature")["LIME_Weight"]

        assert lime_values_by_feature["Recency"] == pytest.approx(0.35)
        assert lime_values_by_feature["Frequency"] == pytest.approx(-0.20)
        assert lime_values_by_feature["Monetary"] == pytest.approx(0.10)

        assert comparison["Absolute_SHAP"].tolist() == sorted(
            comparison["Absolute_SHAP"].tolist(),
            reverse=True,
        )

        assert (
            comparison["Absolute_SHAP"]
            == comparison["SHAP_Value"].abs()
        ).all()

        assert (
            comparison["Absolute_LIME"]
            == comparison["LIME_Weight"].abs()
        ).all()

        # ---------------------------------------------------------
        # 8. Test invalid comparison inputs
        # ---------------------------------------------------------
        with pytest.raises(ValueError, match="top_n"):
            compare_lime_with_shap(
                lime_result=lime_result,
                shap_explanation=shap,
                customer_idx=0,
                top_n=0,
            )

        invalid_shap = MagicMock()
        invalid_shap.values = np.array([0.1, 0.2, 0.3])
        invalid_shap.feature_names = feature_names

        with pytest.raises(ValueError, match="shape"):
            compare_lime_with_shap(
                lime_result=lime_result,
                shap_explanation=invalid_shap,
                customer_idx=0,
            )

        with pytest.raises(IndexError, match="customer_idx"):
            compare_lime_with_shap(
                lime_result=lime_result,
                shap_explanation=shap,
                customer_idx=1,
            )

        # Preserve the current production behavior when names are None.
        shap_with_none_names = MagicMock()
        shap_with_none_names.values = np.array([[0.1, 0.2, 0.3]])
        shap_with_none_names.feature_names = None

        with pytest.raises(TypeError, match="NoneType.*iterable"):
            compare_lime_with_shap(
                lime_result=lime_result,
                shap_explanation=shap_with_none_names,
                customer_idx=0,
            )

        # Empty feature names should trigger the existing validation.
        shap_with_empty_names = MagicMock()
        shap_with_empty_names.values = np.array([[0.1, 0.2, 0.3]])
        shap_with_empty_names.feature_names = []

        with pytest.raises(ValueError, match="feature names"):
            compare_lime_with_shap(
                lime_result=lime_result,
                shap_explanation=shap_with_empty_names,
                customer_idx=0,
            )

    finally:
        plt.close("all")

#########################################################################################################################################
#-------------------------- PDP Tests 
########################################################################################################################################

def test_generate_pdp_analysis(tmp_path):
    """Test PDP generation, artifacts, and input validation."""
    import numpy as np
    import pandas as pd
    import pytest
    import matplotlib.pyplot as plt
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    from src.explainability.pdp_analysis import generate_pdp_analysis

    module = "src.explainability.pdp_analysis"

    try:
        # ---------------------------------------------------------
        # 1. Prepare sample data and fitted classifier
        # ---------------------------------------------------------
        X = pd.DataFrame(
            {
                "Recency": [10, 20, 30, 40, 50, 60],
                "Engagement_Score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "Frequency": [8, 7, 6, 5, 4, 3],
                "Monetary": [1000, 900, 800, 700, 600, 500],
            }
        )

        model = MagicMock()
        model.classes_ = np.array([0, 1])
        model.n_features_in_ = 4
        model.predict_proba.return_value = np.array([[0.25, 0.75]])

        # Mock sklearn's PDP display.
        fake_display = MagicMock()
        fake_display.pd_results = [
            {
                "grid_values": [
                    np.array([10.0, 20.0, 30.0])
                ],
                "average": np.array([[0.2, 0.5, 0.8]]),
            }
        ]

        # ---------------------------------------------------------
        # 2. Test successful PDP generation and artifact saving
        # ---------------------------------------------------------
        with patch(
            f"{module}.PartialDependenceDisplay.from_estimator",
            return_value=fake_display,
        ) as pdp_mock:
            result = generate_pdp_analysis(
                model=model,
                X=X,
                features=["Recency"],
                feature_names=[
                    "Recency",
                    "Engagement_Score",
                    "Frequency",
                    "Monetary",
                ],
                output_dir=tmp_path,
                target_class=1,
                grid_resolution=3,
                max_samples=100,
                show=False,
            )

        assert isinstance(result, dict)
        assert result["target_class"] == 1
        assert result["features"] == ["Recency"]
        assert "Recency" in result["pdp_data"]
        assert "Recency" in result["plots"]

        pdp_table = result["pdp_data"]["Recency"]

        assert isinstance(pdp_table, pd.DataFrame)
        assert list(pdp_table.columns) == [
            "Recency",
            "Average_Predicted_Probability",
        ]
        assert pdp_table["Recency"].tolist() == [10.0, 20.0, 30.0]
        assert pdp_table[
            "Average_Predicted_Probability"
        ].tolist() == pytest.approx([0.2, 0.5, 0.8])

        plot_info = result["plots"]["Recency"]

        assert plot_info["plot_path"] == tmp_path / "pdp_Recency.png"
        assert plot_info["csv_path"] == tmp_path / "pdp_Recency.csv"
        assert plot_info["plot_path"].exists()
        assert plot_info["csv_path"].exists()

        saved_csv = pd.read_csv(plot_info["csv_path"])
        assert len(saved_csv) == 3
        assert list(saved_csv.columns) == list(pdp_table.columns)

        pdp_mock.assert_called_once()
        assert pdp_mock.call_args.kwargs["target"] == 1
        assert pdp_mock.call_args.kwargs["response_method"] == "predict_proba"
        assert pdp_mock.call_args.kwargs["grid_resolution"] == 3

        # ---------------------------------------------------------
        # 3. Test default feature selection
        # ---------------------------------------------------------
        default_display = MagicMock()
        default_display.pd_results = [
            {
                "grid_values": [np.array([1.0, 2.0])],
                "average": np.array([[0.3, 0.6]]),
            }
        ]

        with patch(
            f"{module}.PartialDependenceDisplay.from_estimator",
            return_value=default_display,
        ) as default_pdp_mock:
            default_result = generate_pdp_analysis(
                model=model,
                X=X,
                output_dir=None,
                show=False,
            )

        assert default_result["features"] == [
            "Recency",
            "Engagement_Score",
            "Frequency",
        ]
        assert len(default_pdp_mock.call_args_list) == 3

        # ---------------------------------------------------------
        # 4. Test show=True
        # ---------------------------------------------------------
        show_display = MagicMock()
        show_display.pd_results = [
            {
                "grid_values": [np.array([10.0, 20.0])],
                "average": np.array([[0.2, 0.7]]),
            }
        ]

        with (
            patch(
                f"{module}.PartialDependenceDisplay.from_estimator",
                return_value=show_display,
            ),
            patch(f"{module}.plt.show") as show_mock,
        ):
            show_result = generate_pdp_analysis(
                model=model,
                X=X,
                features=["Recency"],
                output_dir=None,
                show=True,
            )

        assert "Recency" in show_result["pdp_data"]
        show_mock.assert_called_once()

        # ---------------------------------------------------------
        # 5. Test target class selection
        # ---------------------------------------------------------
        multiclass_model = MagicMock()
        multiclass_model.classes_ = np.array([2, 5, 9])
        multiclass_model.n_features_in_ = 4

        multiclass_display = MagicMock()
        multiclass_display.pd_results = [
            {
                "grid_values": [np.array([10.0, 20.0])],
                "average": np.array([[0.3, 0.6]]),
            }
        ]

        with patch(
            f"{module}.PartialDependenceDisplay.from_estimator",
            return_value=multiclass_display,
        ) as multiclass_mock:
            multiclass_result = generate_pdp_analysis(
                model=multiclass_model,
                X=X,
                features=["Recency"],
                target_class=5,
                output_dir=None,
            )

        assert multiclass_result["target_class"] == 5
        assert multiclass_mock.call_args.kwargs["target"] == 1

        with pytest.raises(ValueError, match="Target class"):
            generate_pdp_analysis(
                model=multiclass_model,
                X=X,
                features=["Recency"],
                target_class=7,
                output_dir=None,
            )

        # ---------------------------------------------------------
        # 6. Test input validation
        # ---------------------------------------------------------
        with pytest.raises(TypeError, match="DataFrame"):
            generate_pdp_analysis(model=model, X=np.array([[1, 2]]))

        with pytest.raises(ValueError, match="empty"):
            generate_pdp_analysis(
                model=model,
                X=pd.DataFrame(),
            )

        duplicate_columns = pd.DataFrame(
            [[1, 2, 3, 4]],
            columns=["Recency", "Recency", "Frequency", "Monetary"],
        )

        with pytest.raises(ValueError, match="duplicate"):
            generate_pdp_analysis(
                model=model,
                X=duplicate_columns,
            )

        model_without_proba = MagicMock(spec=["predict"])

        with pytest.raises(TypeError, match="predict_proba"):
            generate_pdp_analysis(
                model=model_without_proba,
                X=X,
            )

        with pytest.raises(ValueError, match="No model features"):
            generate_pdp_analysis(
                model=model,
                X=X,
                feature_names=[],
            )

        with pytest.raises(ValueError, match="duplicates"):
            generate_pdp_analysis(
                model=model,
                X=X,
                feature_names=[
                    "Recency",
                    "Recency",
                ],
            )

        with pytest.raises(ValueError, match="Missing model features"):
            generate_pdp_analysis(
                model=model,
                X=X,
                feature_names=[
                    "Recency",
                    "MissingFeature",
                ],
            )

        nonnumeric_X = X.copy()
        nonnumeric_X["Recency"] = "unknown"

        with pytest.raises(ValueError, match="numeric"):
            generate_pdp_analysis(
                model=model,
                X=nonnumeric_X,
            )

        invalid_X = X.copy()
        invalid_X["Recency"] = invalid_X["Recency"].astype(float)
        invalid_X.loc[0, "Recency"] = np.inf

        with pytest.raises(ValueError, match="infinite"):
            generate_pdp_analysis(
                model=model,
                X=invalid_X,
            )

        wrong_count_model = MagicMock()
        wrong_count_model.classes_ = np.array([0, 1])
        wrong_count_model.n_features_in_ = 5

        with pytest.raises(ValueError, match="expects"):
            generate_pdp_analysis(
                model=wrong_count_model,
                X=X,
            )

        with pytest.raises(ValueError, match="grid_resolution"):
            generate_pdp_analysis(
                model=model,
                X=X,
                grid_resolution=1,
            )

        with pytest.raises(TypeError, match="max_samples"):
            generate_pdp_analysis(
                model=model,
                X=X,
                max_samples=1.5,
            )

        with pytest.raises(ValueError, match="max_samples"):
            generate_pdp_analysis(
                model=model,
                X=X,
                max_samples=0,
            )

        model_without_classes = MagicMock(spec=["predict_proba"])
        
        with pytest.raises(ValueError, match="classes_"):
            generate_pdp_analysis(
                model=model_without_classes,
                X=X,
            )

        with pytest.raises(ValueError, match="No valid PDP features"):
            generate_pdp_analysis(
                model=model,
                X=X,
                features=[],
            )

        with pytest.raises(ValueError, match="duplicates"):
            generate_pdp_analysis(
                model=model,
                X=X,
                features=["Recency", "Recency"],
            )

        with pytest.raises(ValueError, match="missing"):
            generate_pdp_analysis(
                model=model,
                X=X,
                features=["MissingFeature"],
            )

        # ---------------------------------------------------------
        # 7. Test malformed PDP results
        # ---------------------------------------------------------
        mismatched_display = MagicMock()
        mismatched_display.pd_results = [
            {
                "grid_values": [np.array([10.0, 20.0])],
                "average": np.array([[0.2, 0.4, 0.6]]),
            }
        ]

        with patch(
            f"{module}.PartialDependenceDisplay.from_estimator",
            return_value=mismatched_display,
        ):
            with pytest.raises(ValueError, match="length mismatch"):
                generate_pdp_analysis(
                    model=model,
                    X=X,
                    features=["Recency"],
                    output_dir=None,
                )

        nonfinite_display = MagicMock()
        nonfinite_display.pd_results = [
            {
                "grid_values": [np.array([10.0, 20.0])],
                "average": np.array([[0.2, np.nan]]),
            }
        ]

        with patch(
            f"{module}.PartialDependenceDisplay.from_estimator",
            return_value=nonfinite_display,
        ):
            with pytest.raises(ValueError, match="not finite"):
                generate_pdp_analysis(
                    model=model,
                    X=X,
                    features=["Recency"],
                    output_dir=None,
                )

    finally:
        plt.close("all")




        










