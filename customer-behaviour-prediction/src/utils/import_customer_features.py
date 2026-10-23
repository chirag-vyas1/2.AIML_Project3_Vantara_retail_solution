
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.utils.database import get_database_engine
from src.models.model_persistance import get_project_root


def import_customer_features() -> None:
    """Import validated customer data into PostgreSQL."""

    project_root = Path(get_project_root()).resolve()
    csv_path = (
        project_root
        / "data"
        / "processed"
        / "customer_features.csv"
    )

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Customer features CSV not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    required_columns = [
        "Customer ID",
        "Recency",
        "Frequency",
        "Total_Monetary_Value",
        "Average_Order_Spend",
        "Historical_CLV",
        "Customer_Lifetime_Days",
        "Historical_Average_Order_Value",
        "Historical_Invoice_Count",
        "First_Purchase_Date",
        "Last_Purchase_Date",
        "Average_Basket_Size",
        "Purchase_Frequency_Trend",
        "Interpurchase_Variance",
        "Seasonal_Purchase_Concentration",
        "Home_Fragrance_Affinity",
        "Lighting_Candles_Affinity",
        "Stationery_Paper_Affinity",
        "Jewelry_Fashion_Accessories_Affinity",
        "Bags_Travel_Accessories_Affinity",
        "Kitchen_Dining_Affinity",
        "Storage_Organization_Affinity",
        "Garden_Outdoor_Affinity",
        "Bath_Personal_Care_Affinity",
        "Toys_Games_Affinity",
        "Party_Celebration_Affinity",
        "Gifts_Seasonal_Affinity",
        "Food_Consumables_Affinity",
        "Home_Decor_Affinity",
        "Other___Unclassified_Affinity",
        "Return_Rate",
        "Discount_Markdown_Sensitivity",
        "Engagement_Score",
        "Engagement_Band",
    ]

    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing CSV columns: {sorted(missing_columns)}"
        )

    if df["Customer ID"].isna().any():
        raise ValueError("Customer ID contains missing values.")

    if df["Customer ID"].duplicated().any():
        raise ValueError("Customer ID contains duplicates.")

    df["First_Purchase_Date"] = pd.to_datetime(
        df["First_Purchase_Date"], errors="raise"
    )
    df["Last_Purchase_Date"] = pd.to_datetime(
        df["Last_Purchase_Date"], errors="raise"
    )

    if not (df["Frequency"] % 1 == 0).all():
        raise ValueError("Frequency contains non-integer values.")

    if not (df["Historical_Invoice_Count"] % 1 == 0).all():
        raise ValueError(
            "Historical_Invoice_Count contains non-integer values."
        )

    customer_df = df[
        [
            "Customer ID",
            "First_Purchase_Date",
            "Last_Purchase_Date",
        ]
    ].rename(
        columns={
            "Customer ID": "customer_id",
            "First_Purchase_Date": "first_purchase_date",
            "Last_Purchase_Date": "last_purchase_date",
        }
    )

    feature_column_map = {
        "Customer ID": "customer_id",
        "Recency": "recency",
        "Frequency": "frequency",
        "Total_Monetary_Value": "total_monetary_value",
        "Average_Order_Spend": "average_order_spend",
        "Historical_CLV": "historical_clv",
        "Customer_Lifetime_Days": "customer_lifetime_days",
        "Historical_Average_Order_Value":
            "historical_average_order_value",
        "Historical_Invoice_Count": "historical_invoice_count",
        "Average_Basket_Size": "average_basket_size",
        "Purchase_Frequency_Trend": "purchase_frequency_trend",
        "Interpurchase_Variance": "interpurchase_variance",
        "Seasonal_Purchase_Concentration":
            "seasonal_purchase_concentration",
        "Home_Fragrance_Affinity": "home_fragrance_affinity",
        "Lighting_Candles_Affinity": "lighting_candles_affinity",
        "Stationery_Paper_Affinity": "stationery_paper_affinity",
        "Jewelry_Fashion_Accessories_Affinity":
            "jewelry_fashion_accessories_affinity",
        "Bags_Travel_Accessories_Affinity":
            "bags_travel_accessories_affinity",
        "Kitchen_Dining_Affinity": "kitchen_dining_affinity",
        "Storage_Organization_Affinity": "storage_organization_affinity",
        "Garden_Outdoor_Affinity": "garden_outdoor_affinity",
        "Bath_Personal_Care_Affinity": "bath_personal_care_affinity",
        "Toys_Games_Affinity": "toys_games_affinity",
        "Party_Celebration_Affinity": "party_celebration_affinity",
        "Gifts_Seasonal_Affinity": "gifts_seasonal_affinity",
        "Food_Consumables_Affinity": "food_consumables_affinity",
        "Home_Decor_Affinity": "home_decor_affinity",
        "Other___Unclassified_Affinity": "other_unclassified_affinity",
        "Return_Rate": "return_rate",
        "Discount_Markdown_Sensitivity":
            "discount_markdown_sensitivity",
        "Engagement_Score": "engagement_score",
        "Engagement_Band": "engagement_band",
    }

    feature_df = df[list(feature_column_map)].rename(
        columns=feature_column_map
    )

    feature_df["frequency"] = feature_df["frequency"].astype(int)
    feature_df["historical_invoice_count"] = (
        feature_df["historical_invoice_count"].astype(int)
    )

    # actual_churn is deliberately omitted because no confirmed
    # actual-churn label is present in this CSV.
    customer_records = customer_df.to_dict(orient="records")
    feature_records = feature_df.to_dict(orient="records")

    engine = get_database_engine()

    try:
        with engine.begin() as connection:
            # Insert parent customer rows first.
            connection.execute(
                text("""
                    INSERT INTO vantara.customers (
                        customer_id,
                        first_purchase_date,
                        last_purchase_date
                    )
                    VALUES (
                        :customer_id,
                        :first_purchase_date,
                        :last_purchase_date
                    )
                    ON CONFLICT (customer_id) DO NOTHING
                """),
                customer_records,
            )

            # Insert or update feature values for each customer.
            connection.execute(
                text("""
                    INSERT INTO vantara.customer_features (
                        customer_id,
                        recency,
                        frequency,
                        total_monetary_value,
                        average_order_spend,
                        historical_clv,
                        customer_lifetime_days,
                        historical_average_order_value,
                        historical_invoice_count,
                        average_basket_size,
                        purchase_frequency_trend,
                        interpurchase_variance,
                        seasonal_purchase_concentration,
                        home_fragrance_affinity,
                        lighting_candles_affinity,
                        stationery_paper_affinity,
                        jewelry_fashion_accessories_affinity,
                        bags_travel_accessories_affinity,
                        kitchen_dining_affinity,
                        storage_organization_affinity,
                        garden_outdoor_affinity,
                        bath_personal_care_affinity,
                        toys_games_affinity,
                        party_celebration_affinity,
                        gifts_seasonal_affinity,
                        food_consumables_affinity,
                        home_decor_affinity,
                        other_unclassified_affinity,
                        return_rate,
                        discount_markdown_sensitivity,
                        engagement_score,
                        engagement_band
                    )
                    VALUES (
                        :customer_id,
                        :recency,
                        :frequency,
                        :total_monetary_value,
                        :average_order_spend,
                        :historical_clv,
                        :customer_lifetime_days,
                        :historical_average_order_value,
                        :historical_invoice_count,
                        :average_basket_size,
                        :purchase_frequency_trend,
                        :interpurchase_variance,
                        :seasonal_purchase_concentration,
                        :home_fragrance_affinity,
                        :lighting_candles_affinity,
                        :stationery_paper_affinity,
                        :jewelry_fashion_accessories_affinity,
                        :bags_travel_accessories_affinity,
                        :kitchen_dining_affinity,
                        :storage_organization_affinity,
                        :garden_outdoor_affinity,
                        :bath_personal_care_affinity,
                        :toys_games_affinity,
                        :party_celebration_affinity,
                        :gifts_seasonal_affinity,
                        :food_consumables_affinity,
                        :home_decor_affinity,
                        :other_unclassified_affinity,
                        :return_rate,
                        :discount_markdown_sensitivity,
                        :engagement_score,
                        :engagement_band
                    )
                    ON CONFLICT (customer_id) DO UPDATE SET
                        recency = EXCLUDED.recency,
                        frequency = EXCLUDED.frequency,
                        total_monetary_value =
                            EXCLUDED.total_monetary_value,
                        average_order_spend = EXCLUDED.average_order_spend,
                        historical_clv = EXCLUDED.historical_clv,
                        customer_lifetime_days =
                            EXCLUDED.customer_lifetime_days,
                        historical_average_order_value =
                            EXCLUDED.historical_average_order_value,
                        historical_invoice_count =
                            EXCLUDED.historical_invoice_count,
                        average_basket_size = EXCLUDED.average_basket_size,
                        purchase_frequency_trend =
                            EXCLUDED.purchase_frequency_trend,
                        interpurchase_variance =
                            EXCLUDED.interpurchase_variance,
                        seasonal_purchase_concentration =
                            EXCLUDED.seasonal_purchase_concentration,
                        home_fragrance_affinity =
                            EXCLUDED.home_fragrance_affinity,
                        lighting_candles_affinity =
                            EXCLUDED.lighting_candles_affinity,
                        stationery_paper_affinity =
                            EXCLUDED.stationery_paper_affinity,
                        jewelry_fashion_accessories_affinity =
                            EXCLUDED.jewelry_fashion_accessories_affinity,
                        bags_travel_accessories_affinity =
                            EXCLUDED.bags_travel_accessories_affinity,
                        kitchen_dining_affinity =
                            EXCLUDED.kitchen_dining_affinity,
                        storage_organization_affinity =
                            EXCLUDED.storage_organization_affinity,
                        garden_outdoor_affinity =
                            EXCLUDED.garden_outdoor_affinity,
                        bath_personal_care_affinity =
                            EXCLUDED.bath_personal_care_affinity,
                        toys_games_affinity =
                            EXCLUDED.toys_games_affinity,
                        party_celebration_affinity =
                            EXCLUDED.party_celebration_affinity,
                        gifts_seasonal_affinity =
                            EXCLUDED.gifts_seasonal_affinity,
                        food_consumables_affinity =
                            EXCLUDED.food_consumables_affinity,
                        home_decor_affinity = EXCLUDED.home_decor_affinity,
                        other_unclassified_affinity =
                            EXCLUDED.other_unclassified_affinity,
                        return_rate = EXCLUDED.return_rate,
                        discount_markdown_sensitivity =
                            EXCLUDED.discount_markdown_sensitivity,
                        engagement_score = EXCLUDED.engagement_score,
                        engagement_band = EXCLUDED.engagement_band,
                        updated_at = CURRENT_TIMESTAMP
                """),
                feature_records,
            )

            customer_count = connection.execute(
                text("SELECT COUNT(*) FROM vantara.customers")
            ).scalar_one()

            feature_count = connection.execute(
                text("SELECT COUNT(*) FROM vantara.customer_features")
            ).scalar_one()

        print(f"CSV records processed: {len(df)}")
        print(f"Customers in database: {customer_count}")
        print(f"Customer feature records: {feature_count}")

    finally:
        engine.dispose()


if __name__ == "__main__":
    import_customer_features()
