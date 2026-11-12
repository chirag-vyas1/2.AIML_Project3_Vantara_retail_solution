# Vantara Customer Behaviour Prediction API

**API version:** 1.0.0  
**Framework:** FastAPI  
**Purpose:** Customer churn prediction and SHAP-based model explanations

This document describes the endpoints and request/response structures implemented in the supplied API source files.

## 1. Running the API

When the API starts, it loads configuration and the classification model artifact using the `xgboost` model key. It stores the loaded model bundle in the FastAPI application state and initializes a SHAP `TreeExplainer` for reuse by explanation requests.

With the local Docker deployment running, use:

- API base URL: `http://localhost:8000`
- Interactive Swagger documentation: `http://localhost:8000/docs`
- OpenAPI schema: `http://localhost:8000/openapi.json`
- Health check: `http://localhost:8000/health`

The API application declares version `1.0.0`.

## 2. Endpoint summary

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check API service health |
| GET | `/metadata` | Retrieve loaded model metadata |
| POST | `/predict` | Predict churn for one customer supplied in the request |
| POST | `/predict/batch` | Predict churn for multiple customers supplied in the request |
| POST | `/predict/customer/{customer_id}` | Retrieve stored customer features, predict churn, and persist the prediction |
| POST | `/predict/batch/csv` | Upload a CSV and download a CSV prediction report |
| POST | `/explain/shap/customer` | Return an individual prediction and SHAP contributions |
| POST | `/explain/shap/global` | Return SHAP values and global mean-absolute feature importance |

## 3. Shared customer input schema

The single-customer prediction and individual SHAP endpoints accept a JSON object containing **all 27 engineered features** listed below.

The same `CustomerFeatures` schema is used for each customer in `/predict/batch` and for each item in the `customers` array of `/explain/shap/global`.

Validation rules implemented by Pydantic:

- All 27 fields are required.
- All fields except `Frequency` must be finite numeric values (`FiniteFloat`).
- `Frequency` must be a strict integer (`StrictInt`); a floating-point value such as `12.0` is not accepted as an integer.
- Additional/unrecognized fields are forbidden.
- Non-finite numeric values such as `NaN` and infinity are not accepted.

| # | Field | Type |
|---:|---|---|
| 1 | `Recency` | Finite number |
| 2 | `Frequency` | Strict integer |
| 3 | `Total_Monetary_Value` | Finite number |
| 4 | `Average_Order_Spend` | Finite number |
| 5 | `Customer_Lifetime_Days` | Finite number |
| 6 | `Average_Basket_Size` | Finite number |
| 7 | `Purchase_Frequency_Trend` | Finite number |
| 8 | `Interpurchase_Variance` | Finite number |
| 9 | `Seasonal_Purchase_Concentration` | Finite number |
| 10 | `Home_Fragrance_Affinity` | Finite number |
| 11 | `Lighting_Candles_Affinity` | Finite number |
| 12 | `Stationery_Paper_Affinity` | Finite number |
| 13 | `Jewelry_Fashion_Accessories_Affinity` | Finite number |
| 14 | `Bags_Travel_Accessories_Affinity` | Finite number |
| 15 | `Kitchen_Dining_Affinity` | Finite number |
| 16 | `Storage_Organization_Affinity` | Finite number |
| 17 | `Garden_Outdoor_Affinity` | Finite number |
| 18 | `Bath_Personal_Care_Affinity` | Finite number |
| 19 | `Toys_Games_Affinity` | Finite number |
| 20 | `Party_Celebration_Affinity` | Finite number |
| 21 | `Gifts_Seasonal_Affinity` | Finite number |
| 22 | `Food_Consumables_Affinity` | Finite number |
| 23 | `Home_Decor_Affinity` | Finite number |
| 24 | `Other___Unclassified_Affinity` | Finite number |
| 25 | `Return_Rate` | Finite number |
| 26 | `Discount_Markdown_Sensitivity` | Finite number |
| 27 | `Engagement_Score` | Finite number |

### Illustrative customer payload

The values below are synthetic examples to demonstrate the JSON structure. They are not recommended business thresholds or a sample from the project dataset.

```json
{
  "Recency": 30.0,
  "Frequency": 12,
  "Total_Monetary_Value": 1250.50,
  "Average_Order_Spend": 104.21,
  "Customer_Lifetime_Days": 365.0,
  "Average_Basket_Size": 5.5,
  "Purchase_Frequency_Trend": 0.2,
  "Interpurchase_Variance": 12.5,
  "Seasonal_Purchase_Concentration": 0.3,
  "Home_Fragrance_Affinity": 0.1,
  "Lighting_Candles_Affinity": 0.05,
  "Stationery_Paper_Affinity": 0.02,
  "Jewelry_Fashion_Accessories_Affinity": 0.08,
  "Bags_Travel_Accessories_Affinity": 0.04,
  "Kitchen_Dining_Affinity": 0.12,
  "Storage_Organization_Affinity": 0.06,
  "Garden_Outdoor_Affinity": 0.03,
  "Bath_Personal_Care_Affinity": 0.07,
  "Toys_Games_Affinity": 0.02,
  "Party_Celebration_Affinity": 0.04,
  "Gifts_Seasonal_Affinity": 0.05,
  "Food_Consumables_Affinity": 0.1,
  "Home_Decor_Affinity": 0.15,
  "Other___Unclassified_Affinity": 0.07,
  "Return_Rate": 0.01,
  "Discount_Markdown_Sensitivity": 0.25,
  "Engagement_Score": 0.72
}
```

## 4. Endpoint details

### 4.1 Health check

`GET /health`

Checks that the API health route is responding.

**Successful response — `200 OK`**

```json
{
  "status": "healthy",
  "service": "Vantara Customer Behaviour Prediction API"
}
```

This route reports the API service status. It does not perform a separate database health check.

### 4.2 Model metadata

`GET /metadata`

Returns metadata for the model bundle loaded when the API started.

**Successful response — `200 OK`**

```json
{
  "model_name": "loaded-model-name",
  "model_type": "classification",
  "n_features": 27,
  "feature_names": [
    "Recency",
    "Frequency"
  ]
}
```

The `feature_names` array contains the full feature list returned by the loaded model bundle. The shortened array above is illustrative, not a complete live response.

### 4.3 Single-customer prediction

`POST /predict`

Accepts one `CustomerFeatures` JSON object and returns a churn prediction.

**Request body:** One customer object using all 27 fields in Section 3.

**Successful response — `200 OK`**

```json
{
  "model_name": "loaded-model-name",
  "predicted_class": 1,
  "churn_probability": 0.853
}
```

Response fields:

- `model_name`: Name of the loaded model artifact.
- `predicted_class`: Predicted class returned by the classifier.
- `churn_probability`: Probability corresponding specifically to class `1` (the churn class in this implementation).

**Example using curl**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @customer.json
```

Here, `customer.json` contains one complete customer payload matching Section 3.

**Errors**

- `422 Unprocessable Entity`: Request body fails Pydantic validation, such as a missing field, extra field, invalid number, or non-integer `Frequency`.
- `500 Internal Server Error`: The model prediction fails. The endpoint returns the detail message `The model could not generate a prediction.`

### 4.4 Batch prediction using JSON

`POST /predict/batch`

Accepts a JSON array of customer objects and predicts all customers in one model batch.

**Request body:** An array of one or more complete customer objects matching Section 3.

```json
[
  {
    "Recency": 30.0,
    "Frequency": 12,
    "Total_Monetary_Value": 1250.50,
    "Average_Order_Spend": 104.21,
    "Customer_Lifetime_Days": 365.0,
    "Average_Basket_Size": 5.5,
    "Purchase_Frequency_Trend": 0.2,
    "Interpurchase_Variance": 12.5,
    "Seasonal_Purchase_Concentration": 0.3,
    "Home_Fragrance_Affinity": 0.1,
    "Lighting_Candles_Affinity": 0.05,
    "Stationery_Paper_Affinity": 0.02,
    "Jewelry_Fashion_Accessories_Affinity": 0.08,
    "Bags_Travel_Accessories_Affinity": 0.04,
    "Kitchen_Dining_Affinity": 0.12,
    "Storage_Organization_Affinity": 0.06,
    "Garden_Outdoor_Affinity": 0.03,
    "Bath_Personal_Care_Affinity": 0.07,
    "Toys_Games_Affinity": 0.02,
    "Party_Celebration_Affinity": 0.04,
    "Gifts_Seasonal_Affinity": 0.05,
    "Food_Consumables_Affinity": 0.1,
    "Home_Decor_Affinity": 0.15,
    "Other___Unclassified_Affinity": 0.07,
    "Return_Rate": 0.01,
    "Discount_Markdown_Sensitivity": 0.25,
    "Engagement_Score": 0.72
  }
]
```

**Successful response — `200 OK`**

```json
{
  "model_name": "loaded-model-name",
  "total_customers": 1,
  "predictions": [
    {
      "model_name": "loaded-model-name",
      "predicted_class": 1,
      "churn_probability": 0.853
    }
  ]
}
```

**Errors**

- `422 Unprocessable Entity`: Invalid customer payload or an empty array. The explicit empty-array detail is `At least one customer must be provided.`
- `500 Internal Server Error`: The model could not generate batch predictions.

### 4.5 Prediction for a customer stored in PostgreSQL

`POST /predict/customer/{customer_id}`

Uses a customer ID to retrieve that customer's stored feature values, generate a churn prediction, save the prediction, and return the result.

**Path parameter**

| Parameter | Type | Description |
|---|---|---|
| `customer_id` | Integer | Positive customer identifier used to retrieve stored features |

**Example**

```bash
curl -X POST "http://localhost:8000/predict/customer/12345"
```

Replace `12345` with a customer ID whose feature record exists in the configured PostgreSQL database.

**Successful response — `200 OK`**

```json
{
  "model_name": "loaded-model-name",
  "predicted_class": 1,
  "churn_probability": 0.853
}
```

On success, the endpoint calls the prediction persistence function with the customer ID, model name, predicted class, and churn probability.

**Errors**

- `400 Bad Request`: Customer ID is not a positive integer. Detail: `Customer ID must be a positive integer.`
- `404 Not Found`: No stored customer feature record was found for that ID.
- `422 Unprocessable Entity`: A `ValueError` occurred while processing the stored customer features.
- `500 Internal Server Error`: Prediction or database persistence failed.

### 4.6 Batch prediction using CSV upload

`POST /predict/batch/csv`

Accepts a multipart file upload and returns a downloadable CSV report.

**Request format:** `multipart/form-data`

| Form field | Type | Required | Description |
|---|---|---:|---|
| `file` | File | Yes | CSV file containing customer features |

The uploaded filename must have a `.csv` extension. The endpoint writes the upload to a temporary directory, calls the batch CSV prediction service, reads the generated report, and streams it back. The temporary directory is removed after processing and the upload is closed.

**Example using curl**

```bash
curl -X POST "http://localhost:8000/predict/batch/csv" \
  -F "file=@customer_features.csv" \
  --output churn_predictions.csv
```

**Successful response — `200 OK`**

- Media type: `text/csv`
- Download filename: `churn_predictions.csv`
- Response body: CSV content produced by the batch prediction service.

**Errors**

- `400 Bad Request`: Missing filename or filename extension is not `.csv`. Detail: `Please upload a CSV file.`
- `422 Unprocessable Entity`: CSV parsing, file reading, or input validation error; the endpoint returns the underlying exception message.
- `500 Internal Server Error`: Other CSV batch prediction failure.

**Important implementation note:** The uploaded router delegates CSV column validation, prediction processing, and output-column construction to `src.services.batch_prediction_service.predict_batch_from_csv`. That service source was not included with the files used to prepare this document. Therefore, this document does not invent a CSV column order, extra CSV columns, or the exact output CSV headers. Confirm those details against the service implementation or a real generated report before publishing a CSV template.

### 4.7 Individual SHAP explanation

`POST /explain/shap/customer`

Accepts one complete customer feature object and returns the model prediction together with SHAP contributions for that customer.

**Request body:** One customer object using all 27 fields in Section 3.

**Successful response — `200 OK`**

```json
{
  "model_name": "loaded-model-name",
  "predicted_class": 1,
  "prediction_label": "Churn",
  "churn_probability": 0.853,
  "contributions": [
    {
      "feature": "Recency",
      "value": 30.0,
      "shap_value": 0.42,
      "absolute_shap": 0.42
    }
  ],
  "base_value": 0.1,
  "feature_names": ["Recency", "Frequency"],
  "feature_values": [30.0, 12],
  "shap_values": [0.42, -0.08]
}
```

The values and shortened arrays above are illustrative. A real response contains the complete feature arrays and contribution records. Contribution records are sorted by descending absolute SHAP value.

Response fields:

- `prediction_label`: `Churn` when predicted class is `1`, otherwise `No Churn`.
- `churn_probability`: Probability for class `1`.
- `contributions`: Feature name, input value, signed SHAP value, and absolute SHAP value for each feature.
- `base_value`: Explainer base value returned by SHAP.
- `feature_names`, `feature_values`, `shap_values`: Arrays aligned to the model feature order.

**Errors**

- `422 Unprocessable Entity`: Invalid customer feature payload.
- `500 Internal Server Error`: SHAP explanation could not be generated. Detail: `Unable to generate the customer SHAP explanation.`

### 4.8 Global SHAP explanation

`POST /explain/shap/global`

Returns SHAP values for a group of customers and global feature importance calculated as mean absolute SHAP value across the submitted customers.

**Request body**

```json
{
  "customers": [
    {
      "Recency": 30.0,
      "Frequency": 12,
      "Total_Monetary_Value": 1250.50,
      "Average_Order_Spend": 104.21,
      "Customer_Lifetime_Days": 365.0,
      "Average_Basket_Size": 5.5,
      "Purchase_Frequency_Trend": 0.2,
      "Interpurchase_Variance": 12.5,
      "Seasonal_Purchase_Concentration": 0.3,
      "Home_Fragrance_Affinity": 0.1,
      "Lighting_Candles_Affinity": 0.05,
      "Stationery_Paper_Affinity": 0.02,
      "Jewelry_Fashion_Accessories_Affinity": 0.08,
      "Bags_Travel_Accessories_Affinity": 0.04,
      "Kitchen_Dining_Affinity": 0.12,
      "Storage_Organization_Affinity": 0.06,
      "Garden_Outdoor_Affinity": 0.03,
      "Bath_Personal_Care_Affinity": 0.07,
      "Toys_Games_Affinity": 0.02,
      "Party_Celebration_Affinity": 0.04,
      "Gifts_Seasonal_Affinity": 0.05,
      "Food_Consumables_Affinity": 0.1,
      "Home_Decor_Affinity": 0.15,
      "Other___Unclassified_Affinity": 0.07,
      "Return_Rate": 0.01,
      "Discount_Markdown_Sensitivity": 0.25,
      "Engagement_Score": 0.72
    }
  ],
  "max_features": 15
}
```

Request constraints:

- `customers` is required and must contain between 1 and 500 customer objects.
- Each customer must satisfy the complete `CustomerFeatures` schema.
- `max_features` is optional; default is `15`, minimum is `1`, maximum is `27`.

**Successful response — `200 OK`**

```json
{
  "model_name": "loaded-model-name",
  "customers_analysed": 1,
  "feature_names": ["Recency", "Frequency"],
  "feature_values": [[30.0, 12]],
  "shap_values": [[0.42, -0.08]],
  "importance": [
    {
      "feature": "Recency",
      "mean_absolute_shap": 0.42
    }
  ]
}
```

The shortened arrays above are illustrative. A real response contains all model features in `feature_names`, `feature_values`, and `shap_values`. The `importance` list is sorted by descending `mean_absolute_shap` and limited by `max_features`.

**Errors**

- `422 Unprocessable Entity`: Invalid request structure, customer features, empty or oversized customer list, or invalid `max_features`.
- `500 Internal Server Error`: Global SHAP insights could not be generated. Detail: `Unable to generate global SHAP insights.`

## 5. Prediction interpretation

The implementation treats class `1` as the churn class when selecting the probability column. Interpret the response as follows:

- `predicted_class = 1`: model predicts churn.
- `predicted_class = 0`: model predicts no churn.
- `churn_probability`: model probability for class `1`, represented as a numeric value between 0 and 1.

A probability is a model estimate, not a guarantee that a customer will churn. SHAP values describe how features contribute to the model output; they should not be interpreted as proof of causation.

## 6. Validation and error handling

FastAPI/Pydantic request validation is used for the JSON endpoints. Invalid request bodies generally return `422 Unprocessable Entity`. Endpoint-specific errors are listed under each endpoint above.

The application also exposes FastAPI's generated interactive API documentation at `/docs` and the OpenAPI schema at `/openapi.json`.

## 7. Implementation notes and scope

- The model and SHAP explainer are loaded once during application startup and stored on `app.state`.
- The API startup code requests the `xgboost` classification artifact.
- Model feature order is taken from the loaded model bundle when constructing prediction DataFrames.
- The SHAP implementation uses `shap.TreeExplainer`.
- The stored-customer endpoint reads customer features and persists its prediction through repository functions.
- The CSV endpoint delegates CSV processing to a separate service module.
- The supplied source files do not define authentication, authorization, rate limiting, or an explicit API-wide request-size limit. Do not assume those controls are present.
- Database connection settings and schema/table definitions are outside the supplied API source files and are not specified here.

## 8. Source files reviewed

This documentation was prepared from the following supplied files:

- `api/main.py`
- `api/schemas/prediction.py`
- `api/routers/predict.py`
- `api/routers/metadata.py`
- `api/routers/explainability.py`

Before final submission, verify the CSV input/output contract against `src/services/batch_prediction_service.py` and verify the database behavior against the repository/persistence modules. Also run the live requests against the deployed container and compare the actual responses with the illustrative examples in this document.
