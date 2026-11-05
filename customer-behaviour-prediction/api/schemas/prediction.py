
from pydantic import BaseModel, ConfigDict, FiniteFloat, StrictInt


class CustomerFeatures(BaseModel):
    """Validate the 27 engineered features used by the churn model."""

    model_config = ConfigDict(extra="forbid")

    Recency: FiniteFloat
    Frequency: StrictInt
    Total_Monetary_Value: FiniteFloat
    Average_Order_Spend: FiniteFloat
    Customer_Lifetime_Days: FiniteFloat
    Average_Basket_Size: FiniteFloat
    Purchase_Frequency_Trend: FiniteFloat
    Interpurchase_Variance: FiniteFloat
    Seasonal_Purchase_Concentration: FiniteFloat
    Home_Fragrance_Affinity: FiniteFloat
    Lighting_Candles_Affinity: FiniteFloat
    Stationery_Paper_Affinity: FiniteFloat
    Jewelry_Fashion_Accessories_Affinity: FiniteFloat
    Bags_Travel_Accessories_Affinity: FiniteFloat
    Kitchen_Dining_Affinity: FiniteFloat
    Storage_Organization_Affinity: FiniteFloat
    Garden_Outdoor_Affinity: FiniteFloat
    Bath_Personal_Care_Affinity: FiniteFloat
    Toys_Games_Affinity: FiniteFloat
    Party_Celebration_Affinity: FiniteFloat
    Gifts_Seasonal_Affinity: FiniteFloat
    Food_Consumables_Affinity: FiniteFloat
    Home_Decor_Affinity: FiniteFloat
    Other___Unclassified_Affinity: FiniteFloat
    Return_Rate: FiniteFloat
    Discount_Markdown_Sensitivity: FiniteFloat
    Engagement_Score: FiniteFloat
