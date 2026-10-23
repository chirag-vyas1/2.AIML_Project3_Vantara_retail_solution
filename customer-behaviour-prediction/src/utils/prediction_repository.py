
from sqlalchemy import text
from src.utils.database import get_database_engine


def get_customer_features(customer_id: int) -> dict | None:
    """Fetch the 27 model features for an existing customer."""

    query = text("""
        SELECT
            recency AS "Recency",
            frequency AS "Frequency",
            total_monetary_value AS "Total_Monetary_Value",
            average_order_spend AS "Average_Order_Spend",
            customer_lifetime_days AS "Customer_Lifetime_Days",
            average_basket_size AS "Average_Basket_Size",
            purchase_frequency_trend AS "Purchase_Frequency_Trend",
            interpurchase_variance AS "Interpurchase_Variance",
            seasonal_purchase_concentration
                AS "Seasonal_Purchase_Concentration",
            home_fragrance_affinity AS "Home_Fragrance_Affinity",
            lighting_candles_affinity AS "Lighting_Candles_Affinity",
            stationery_paper_affinity AS "Stationery_Paper_Affinity",
            jewelry_fashion_accessories_affinity
                AS "Jewelry_Fashion_Accessories_Affinity",
            bags_travel_accessories_affinity
                AS "Bags_Travel_Accessories_Affinity",
            kitchen_dining_affinity AS "Kitchen_Dining_Affinity",
            storage_organization_affinity
                AS "Storage_Organization_Affinity",
            garden_outdoor_affinity AS "Garden_Outdoor_Affinity",
            bath_personal_care_affinity AS "Bath_Personal_Care_Affinity",
            toys_games_affinity AS "Toys_Games_Affinity",
            party_celebration_affinity AS "Party_Celebration_Affinity",
            gifts_seasonal_affinity AS "Gifts_Seasonal_Affinity",
            food_consumables_affinity AS "Food_Consumables_Affinity",
            home_decor_affinity AS "Home_Decor_Affinity",
            other_unclassified_affinity AS "Other___Unclassified_Affinity",
            return_rate AS "Return_Rate",
            discount_markdown_sensitivity
                AS "Discount_Markdown_Sensitivity",
            engagement_score AS "Engagement_Score"
        FROM vantara.customer_features
        WHERE customer_id = :customer_id
    """)

    engine = get_database_engine()

    try:
        with engine.connect() as connection:
            row = connection.execute(
                query, {"customer_id": customer_id}
            ).mappings().first()

            return dict(row) if row is not None else None
    finally:
        engine.dispose()


def save_prediction(
    customer_id: int,
    model_name: str,
    predicted_class: int,
    churn_probability: float,
) -> None:
    """Save a prediction for a customer with stored features."""

    query = text("""
        INSERT INTO vantara.churn_predictions (
            customer_id,
            model_name,
            predicted_class,
            churn_probability
        )
        SELECT
            c.customer_id,
            :model_name,
            :predicted_class,
            :churn_probability
        FROM vantara.customers AS c
        INNER JOIN vantara.customer_features AS f
            ON f.customer_id = c.customer_id
        WHERE c.customer_id = :customer_id
    """)

    engine = get_database_engine()

    try:
        with engine.begin() as connection:
            result = connection.execute(
                query,
                {
                    "customer_id": customer_id,
                    "model_name": model_name,
                    "predicted_class": predicted_class,
                    "churn_probability": churn_probability,
                },
            )

            if result.rowcount != 1:
                raise ValueError(
                    f"Customer {customer_id} does not have both "
                    "a customer record and a feature record."
                )
    finally:
        engine.dispose()
