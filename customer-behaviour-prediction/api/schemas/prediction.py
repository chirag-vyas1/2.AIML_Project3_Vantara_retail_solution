
from pydantic import BaseModel, ConfigDict, StrictInt, StrictFloat


class CustomerFeatures(BaseModel):
    """The 27 engineered features required by the churn model."""

    model_config = ConfigDict(extra="forbid")

    Recency: StrictFloat
    Frequency: StrictInt
    Total_Monetary_Value: StrictFloat
    Average_Order_Spend: StrictFloat
    Customer_Lifetime_Days: StrictFloat
    Average_Basket_Size: StrictFloat
    Purchase_Frequency_Trend: StrictFloat
    Interpurchase_Variance: StrictFloat
    Seasonal_Purchase_Concentration: StrictFloat
    Home_Fragrance_Affinity: StrictFloat
    Lighting_Candles_Affinity: StrictFloat
    Stationery_Paper_Affinity: StrictFloat
    Jewelry_Fashion_Accessories_Affinity: StrictFloat
    Bags_Travel_Accessories_Affinity: StrictFloat
    Kitchen_Dining_Affinity: StrictFloat
    Storage_Organization_Affinity: StrictFloat
    Garden_Outdoor_Affinity: StrictFloat
    Bath_Personal_Care_Affinity: StrictFloat
    Toys_Games_Affinity: StrictFloat
    Party_Celebration_Affinity: StrictFloat
    Gifts_Seasonal_Affinity: StrictFloat
    Food_Consumables_Affinity: StrictFloat
    Home_Decor_Affinity: StrictFloat
    Other___Unclassified_Affinity: StrictFloat
    Return_Rate: StrictFloat
    Discount_Markdown_Sensitivity: StrictFloat
    Engagement_Score: StrictFloat
