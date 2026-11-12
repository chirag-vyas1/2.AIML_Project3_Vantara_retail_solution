# Vantara Customer Behaviour Prediction API

**API Version:** 1.0.0  
**Framework:** FastAPI  
**Model:** XGBoost Tuned Classifier (`xgboost_tuned_v2`)  
**Purpose:** Customer churn prediction, batch inference and SHAP-based model explainability.

---

# 1. Introduction

The Vantara Customer Behaviour Prediction API provides machine-learning-powered customer churn predictions using engineered retail customer behavioural features.

The API exposes endpoints for:

- Individual customer churn prediction.
- Batch customer prediction using JSON.
- Batch customer prediction using CSV uploads.
- Prediction using customer records stored in PostgreSQL.
- Individual SHAP-based prediction explanations.
- Global SHAP-based feature importance analysis.
- Model metadata and service health monitoring.

The application is implemented using FastAPI and integrates the trained classification model, SHAP explainability and PostgreSQL persistence functionality.

## 1.1 API Deployment Information

When running through Docker Compose:

| Component | URL / Port |
|---|---|
| API Base URL | `http://localhost:8000` |
| Swagger UI | `http://localhost:8000/docs` |
| OpenAPI Schema | `http://localhost:8000/openapi.json` |
| Health Endpoint | `http://localhost:8000/health` |
| Dashboard | `http://localhost:8501` |

The API application declares version `1.0.0`.

## 1.2 Model Initialization

During application startup:

1. The application loads its configuration.
2. The classification model artifact is loaded using the `xgboost` model key.
3. The model bundle is stored in the FastAPI application state.
4. A SHAP `TreeExplainer` is initialized for explainability operations.
5. The loaded model is reused for subsequent prediction requests.

The model bundle contains the model information and expected feature names used during prediction.

---

# 2. API Endpoint Overview

The API provides eight endpoints.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Check API service health |
| GET | `/metadata` | Retrieve loaded model metadata |
| POST | `/predict` | Predict churn for an individual customer |
| POST | `/predict/batch` | Predict churn for multiple customers using JSON |
| POST | `/predict/customer/{customer_id}` | Retrieve stored customer features and persist prediction |
| POST | `/predict/batch/csv` | Upload a CSV and download prediction results |
| POST | `/explain/shap/customer` | Generate individual SHAP explanations |
| POST | `/explain/shap/global` | Generate global SHAP feature importance |

---

# 3. Customer Input Feature Schema

The prediction model expects 27 engineered customer behavioural features.

The `CustomerFeatures` schema is used by:

- `POST /predict`
- `POST /predict/batch`
- `POST /explain/shap/customer`
- `POST /explain/shap/global`

Each customer record must contain all 27 required features.

## 3.1 Feature Definitions and Data Types

| # | Feature Name | Data Type |
|---:|---|---|
| 1 | `Recency` | Finite numeric |
| 2 | `Frequency` | Strict integer |
| 3 | `Total_Monetary_Value` | Finite numeric |
| 4 | `Average_Order_Spend` | Finite numeric |
| 5 | `Customer_Lifetime_Days` | Finite numeric |
| 6 | `Average_Basket_Size` | Finite numeric |
| 7 | `Purchase_Frequency_Trend` | Finite numeric |
| 8 | `Interpurchase_Variance` | Finite numeric |
| 9 | `Seasonal_Purchase_Concentration` | Finite numeric |
| 10 | `Home_Fragrance_Affinity` | Finite numeric |
| 11 | `Lighting_Candles_Affinity` | Finite numeric |
| 12 | `Stationery_Paper_Affinity` | Finite numeric |
| 13 | `Jewelry_Fashion_Accessories_Affinity` | Finite numeric |
| 14 | `Bags_Travel_Accessories_Affinity` | Finite numeric |
| 15 | `Kitchen_Dining_Affinity` | Finite numeric |
| 16 | `Storage_Organization_Affinity` | Finite numeric |
| 17 | `Garden_Outdoor_Affinity` | Finite numeric |
| 18 | `Bath_Personal_Care_Affinity` | Finite numeric |
| 19 | `Toys_Games_Affinity` | Finite numeric |
| 20 | `Party_Celebration_Affinity` | Finite numeric |
| 21 | `Gifts_Seasonal_Affinity` | Finite numeric |
| 22 | `Food_Consumables_Affinity` | Finite numeric |
| 23 | `Home_Decor_Affinity` | Finite numeric |
| 24 | `Other___Unclassified_Affinity` | Finite numeric |
| 25 | `Return_Rate` | Finite numeric |
| 26 | `Discount_Markdown_Sensitivity` | Finite numeric |
| 27 | `Engagement_Score` | Finite numeric |

## 3.2 Input Validation Rules

The Pydantic schema implements the following validation rules:

1. All 27 features are mandatory.
2. `Frequency` must be a strict integer (`StrictInt`).
3. All other features must be finite numeric values (`FiniteFloat`).
4. Floating-point values such as `12.0` are not accepted for `Frequency`.
5. Additional or unrecognised JSON fields are not permitted.
6. Non-finite values such as NaN and infinity are rejected.

Invalid JSON payloads return HTTP `422 Unprocessable Entity`.

## 3.3 Illustrative Customer Payload

The following example demonstrates the expected JSON structure.

These values are synthetic examples and do not represent actual customer records or recommended business thresholds.

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

---

# 4. API Endpoint Documentation

## 4.1 Health Check

**Endpoint:** `GET /health`

**Purpose:** Verify that the API health endpoint is responding.

### Successful Response – HTTP 200

```json
{
  "status": "healthy",
  "service": "Vantara Customer Behaviour Prediction API"
}
```

**Response fields:**

| Field | Description |
|---|---|
| `status` | API service status |
| `service` | Application service name |

**Important:** This endpoint checks API service responsiveness. It does not independently verify PostgreSQL database health.

---

## 4.2 Model Metadata

**Endpoint:** `GET /metadata`

**Purpose:** Retrieve metadata associated with the classification model loaded during API startup.

### Successful Response – HTTP 200

```json
{
  "model_name": "xgboost_tuned_v2",
  "model_type": "classification",
  "n_features": 27,
  "feature_names": [
    "Recency",
    "Frequency"
  ]
}
```

The shortened `feature_names` array above is illustrative. The actual response contains the complete feature list returned by the loaded model bundle.

**Response fields:**

| Field | Description |
|---|---|
| `model_name` | Loaded model name |
| `model_type` | Model task type |
| `n_features` | Number of model input features |
| `feature_names` | Expected model feature names and order |

---

## 4.3 Individual Customer Prediction

**Endpoint:** `POST /predict`

**Purpose:** Generate a churn prediction for one customer.

### Request

- Content-Type: `application/json`
- Request body: One complete customer object containing all 27 features.

### Example using cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @customer.json
```

The `customer.json` file must contain a valid payload following Section 3.

### Successful Response – HTTP 200

```json
{
  "model_name": "xgboost_tuned_v2",
  "predicted_class": 1,
  "churn_probability": 0.853
}
```

### Response Fields

| Field | Description |
|---|---|
| `model_name` | Model used for prediction |
| `predicted_class` | Predicted customer class |
| `churn_probability` | Probability corresponding to class 1 |

### Error Responses

| HTTP Status | Description |
|---|---|
| 422 | Invalid request payload or Pydantic validation failure |
| 500 | Model prediction failure |

For model prediction errors, the endpoint returns:

`The model could not generate a prediction.`

---

## 4.4 JSON Batch Prediction

**Endpoint:** `POST /predict/batch`

**Purpose:** Generate predictions for multiple customers in a single request.

### Request

- Content-Type: `application/json`
- Request body: Array of complete customer objects.

Each customer must contain all 27 required features.

### Illustrative Request

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

### Successful Response – HTTP 200

```json
{
  "model_name": "xgboost_tuned_v2",
  "total_customers": 1,
  "predictions": [
    {
      "model_name": "xgboost_tuned_v2",
      "predicted_class": 1,
      "churn_probability": 0.853
    }
  ]
}
```

The example shows one customer for simplicity. The actual response contains one prediction record for each submitted customer.

### Validation and Errors

| HTTP Status | Description |
|---|---|
| 422 | Invalid customer payload |
| 422 | Empty customer array |
| 500 | Batch model prediction failure |

An empty customer array returns:

`At least one customer must be provided.`

---

## 4.5 Prediction Using a Customer Stored in PostgreSQL

**Endpoint:** `POST /predict/customer/{customer_id}`

**Purpose:** Retrieve an existing customer's stored feature values, generate a churn prediction and save the prediction result.

This endpoint integrates the prediction API with the PostgreSQL persistence layer.

### Path Parameter

| Parameter | Type | Description |
|---|---|---|
| `customer_id` | Integer | Positive identifier of the customer |

### Example

```bash
curl -X POST "http://localhost:8000/predict/customer/12345"
```

Replace `12345` with a valid customer ID whose feature record exists in the configured database.

### Processing Workflow

1. Receive the customer ID.
2. Retrieve the customer's stored feature values.
3. Generate a prediction using the loaded classification model.
4. Save the prediction through the persistence function.
5. Return the prediction response.

### Successful Response – HTTP 200

```json
{
  "model_name": "xgboost_tuned_v2",
  "predicted_class": 1,
  "churn_probability": 0.853
}
```

### Error Responses

| HTTP Status | Description |
|---|---|
| 400 | Customer ID is not positive |
| 404 | Customer feature record not found |
| 422 | ValueError during stored feature processing |
| 500 | Prediction or database persistence failure |

**Important:** The endpoint requires a valid customer feature record in PostgreSQL. Database integration should be validated using an existing customer record before claiming this workflow has passed live testing.

---

## 4.6 CSV Batch Prediction

**Endpoint:** `POST /predict/batch/csv`

**Purpose:** Upload a CSV containing multiple customer records and download a CSV containing their churn predictions.

This endpoint is suitable for batch inference when customer data is available in tabular format.

### Request Format

- Method: `POST`
- Content-Type: `multipart/form-data`
- Form field: `file`
- Accepted file extension: `.csv`

### Request Parameters

| Form Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | Yes | CSV containing customer features |

### Input CSV Requirements

The uploaded CSV must satisfy the following conditions:

1. The filename must have a `.csv` extension.
2. All 27 model feature columns must be present.
3. Additional columns are permitted.
4. All required model feature values must be numeric.
5. Missing values in required model feature columns are rejected.
6. Infinite or invalid model feature values are rejected.
7. The CSV must contain customer records.

### Processing Workflow

The CSV prediction service performs these operations:

1. Reads the uploaded CSV using Pandas.
2. Checks that the input contains customer records.
3. Validates the presence of all required model feature columns.
4. Allows additional input columns.
5. Selects the 27 model features in the expected model feature order.
6. Converts and validates feature values as numeric.
7. Checks for missing and invalid feature values.
8. Generates predicted classes and class-1 probabilities.
9. Preserves the original input columns.
10. Appends the prediction results to the output CSV.

### Example using cURL

```bash
curl -X POST "http://localhost:8000/predict/batch/csv" \
  -F "file=@customer_features.csv" \
  --output churn_predictions.csv
```

### Successful Response – HTTP 200

| Response Property | Value |
|---|---|
| Content-Type | `text/csv` |
| Download filename | `churn_predictions.csv` |
| Response body | Generated CSV prediction report |

### Output CSV Structure

The service preserves all original input columns and appends three prediction columns.

| Output Column | Description |
|---|---|
| `Predicted_Class` | Predicted customer class |
| `Churn_Probability` | Probability corresponding to class 1 |
| `Model_Name` | Model used to generate the prediction |

### Illustrative Output

The following example demonstrates the appended prediction columns. Original customer feature columns are omitted for readability.

| Customer | Predicted_Class | Churn_Probability | Model_Name |
|---|---:|---:|---|
| Customer 1 | 1 | 0.6181 | xgboost_tuned_v2 |
| Customer 2 | 1 | 0.5884 | xgboost_tuned_v2 |
| Customer 3 | 1 | 0.5709 | xgboost_tuned_v2 |

These values are from the synthetic records used during API validation.

### Error Responses

| HTTP Status | Description |
|---|---|
| 400 | Missing filename or invalid CSV file extension |
| 422 | CSV parsing or input validation failure |
| 500 | Other CSV batch prediction failure |

The service validates missing columns, empty records, missing feature values, non-numeric values and invalid/infinite feature values.

### Live API Validation

The endpoint was tested successfully using three synthetic customer records.

Observed results:

- HTTP status: `200`.
- Response content type: `text/csv`.
- Output CSV downloaded successfully.
- All original input columns were preserved.
- Three prediction columns were appended.
- CSV predictions matched the JSON batch predictions for the same three customer records.

**API Test Result: PASS**

---

## 4.7 Individual SHAP Explanation

**Endpoint:** `POST /explain/shap/customer`

**Purpose:** Generate an individual churn prediction with feature-level SHAP contributions.

The endpoint accepts one complete customer feature object and returns the prediction together with the contribution of each feature to the model output.

### Request

- Content-Type: `application/json`
- Request body: One customer object containing all 27 features.

### Successful Response – HTTP 200

```json
{
  "model_name": "xgboost_tuned_v2",
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

The shortened arrays and values are illustrative. The actual response contains all 27 feature records and corresponding arrays.

### Response Fields

| Field | Description |
|---|---|
| `model_name` | Model used for prediction |
| `predicted_class` | Predicted class |
| `prediction_label` | `Churn` or `No Churn` |
| `churn_probability` | Probability for class 1 |
| `contributions` | Individual feature-level SHAP contributions |
| `base_value` | SHAP explainer base value |
| `feature_names` | Model feature names |
| `feature_values` | Input feature values |
| `shap_values` | Corresponding SHAP values |

Contribution records are sorted in descending order of absolute SHAP value.

### Error Responses

| HTTP Status | Description |
|---|---|
| 422 | Invalid customer feature payload |
| 500 | SHAP explanation generation failure |

For SHAP explanation failures, the endpoint returns:

`Unable to generate the customer SHAP explanation.`

### Live API Validation

The endpoint was successfully tested using the same synthetic customer profile supplied to the individual prediction endpoint.

Observed results:

- Model: `xgboost_tuned_v2`.
- Predicted class: `1`.
- Churn probability: approximately `0.56655`.
- SHAP contributions returned for all 27 features.
- Base value and feature arrays returned successfully.

The highest absolute SHAP contributions in this test were:

1. `Engagement_Score`
2. `Recency`
3. `Seasonal_Purchase_Concentration`

**API Test Result: PASS**

---

## 4.8 Global SHAP Explanation

**Endpoint:** `POST /explain/shap/global`

**Purpose:** Generate SHAP values for multiple customer records and calculate global feature importance.

Global feature importance is calculated using the mean absolute SHAP value across the submitted customer records.

### Request Structure

The request contains:

- `customers`: Array of customer feature objects.
- `max_features`: Optional maximum number of features to return.

### Request Constraints

| Parameter | Requirement |
|---|---|
| `customers` | Required |
| Minimum customers | 1 |
| Maximum customers | 500 |
| Customer schema | All 27 required features |
| `max_features` | Optional |
| Default `max_features` | 15 |
| Minimum `max_features` | 1 |
| Maximum `max_features` | 27 |

### Illustrative Request

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
  "max_features": 10
}
```

### Successful Response – HTTP 200

```json
{
  "model_name": "xgboost_tuned_v2",
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

The shortened arrays above are illustrative. The actual response contains the complete feature arrays and the requested number of importance records.

### Response Fields

| Field | Description |
|---|---|
| `model_name` | Model used for explanation |
| `customers_analysed` | Number of submitted customers |
| `feature_names` | Model feature names |
| `feature_values` | Customer feature matrix |
| `shap_values` | SHAP contribution matrix |
| `importance` | Global feature importance ranking |

The `importance` list is sorted in descending order of `mean_absolute_shap` and limited by `max_features`.

### Error Responses

| HTTP Status | Description |
|---|---|
| 422 | Invalid request structure or customer features |
| 422 | Empty or oversized customer list |
| 422 | Invalid `max_features` |
| 500 | Global SHAP explanation failure |

For SHAP global explanation failures, the endpoint returns:

`Unable to generate global SHAP insights.`

### Live API Validation

The endpoint was successfully tested with three synthetic customer records and `max_features=10`.

Observed results:

- Customers analysed: 3.
- SHAP values returned for all 27 features.
- Global feature importance returned for 10 features.
- Importance records sorted by descending mean absolute SHAP value.

The leading features in this test were:

1. `Engagement_Score`
2. `Recency`
3. `Seasonal_Purchase_Concentration`
4. `Stationery_Paper_Affinity`
5. `Frequency`

**API Test Result: PASS**

---

# 5. Prediction Interpretation

The classification implementation treats class `1` as the churn class when selecting the probability column.

| Prediction Output | Interpretation |
|---|---|
| `predicted_class = 1` | Model predicts that the customer will churn |
| `predicted_class = 0` | Model predicts that the customer will not churn |
| `churn_probability` | Model-estimated probability of class 1 |

A churn probability is a model estimate, not a guarantee of future customer behaviour.

SHAP values explain how individual features contribute to the model output. They should not be interpreted as proof of causation.

---

# 6. Validation and Error Handling

The API uses FastAPI and Pydantic validation for JSON request bodies.

The main validation and error-handling mechanisms include:

- Required customer feature validation.
- Strict integer validation for `Frequency`.
- Finite numeric validation for other customer features.
- Rejection of unexpected JSON fields.
- Batch request size validation for global SHAP.
- CSV file extension validation.
- CSV column and data validation.
- Database record lookup and prediction persistence handling.
- Endpoint-specific error responses.

## 6.1 Common HTTP Status Codes

| Status Code | Meaning | Typical Scenario |
|---|---|---|
| 200 | OK | Successful API request |
| 400 | Bad Request | Invalid CSV file or customer ID |
| 404 | Not Found | Customer feature record not found |
| 422 | Unprocessable Entity | Invalid input or schema validation |
| 500 | Internal Server Error | Prediction, SHAP or persistence failure |

FastAPI also provides interactive API documentation through `/docs` and an OpenAPI specification through `/openapi.json`.

---

# 7. API Testing Summary

Live API testing was performed against the running Dockerized Vantara application.

The tests used synthetic customer profiles and the loaded `xgboost_tuned_v2` model.

## 7.1 Test Execution Results

| Test ID | Test Description | Result |
|---|---|---|
| TC-01 | Docker services status | PASS |
| TC-02 | Health and model metadata | PASS |
| TC-03 | Individual customer prediction | PASS |
| TC-04 | JSON batch prediction | PASS |
| TC-05 | CSV batch prediction | PASS |
| TC-06 | Individual SHAP explanation | PASS |
| TC-07 | Global SHAP explanation | PASS |
| TC-08 | PostgreSQL stored-customer prediction | PENDING |

## 7.2 Test Environment

| Component | Observed Configuration |
|---|---|
| API | Dockerized FastAPI application |
| Dashboard | Dockerized dashboard application |
| Database | PostgreSQL 18 |
| Model | `xgboost_tuned_v2` |
| Model Features | 27 |
| API Port | 8000 |
| Dashboard Port | 8501 |
| Database Port | 5432 |

## 7.3 Overall Test Status

| Metric | Result |
|---|---:|
| Total planned tests | 8 |
| Passed | 7 |
| Failed | 0 |
| Pending | 1 |
| Executed test pass rate | 100% |

The PostgreSQL database inspection confirmed the following application tables:

- `vantara.customers`
- `vantara.customer_features`
- `vantara.churn_predictions`

However, the table structures and stored customer records have not yet been validated for TC-08. Therefore, the stored-customer prediction endpoint remains pending.

**Note:** The executed API tests validate endpoint functionality using synthetic inputs. They do not independently establish model accuracy or business effectiveness.

---

# 8. Implementation Notes and Limitations

1. The classification model is loaded during application startup.
2. The model bundle is stored in FastAPI application state.
3. The API uses the `xgboost` model artifact key.
4. Model feature order is taken from the loaded model bundle.
5. SHAP explainability uses `shap.TreeExplainer`.
6. Individual SHAP explanations return feature-level contributions.
7. Global SHAP importance is calculated using mean absolute SHAP values.
8. Stored-customer prediction uses repository functions for retrieving features and persisting predictions.
9. CSV batch prediction delegates processing to `src/services/batch_prediction_service.py`.
10. The CSV endpoint preserves original input columns and appends prediction results.
11. The supplied implementation does not establish authentication, authorization, rate limiting or an explicit API-wide request-size limit. These capabilities should not be assumed to exist.
12. The API health endpoint does not independently validate PostgreSQL connectivity.

---

# 9. Source Files Reviewed

The API documentation was prepared using the following project files:

- `api/main.py`
- `api/schemas/prediction.py`
- `api/routers/predict.py`
- `api/routers/metadata.py`
- `api/routers/explainability.py`
- `src/services/batch_prediction_service.py`

Live API behaviour was also checked through requests to the running Dockerized application.

---

# 10. Conclusion

The Vantara Customer Behaviour Prediction API provides a modular interface for individual and batch customer churn prediction, CSV-based inference, database-backed prediction workflows and SHAP-based explainability.

The API successfully executed seven live functional test cases covering service health, model metadata, individual prediction, JSON batch prediction, CSV batch prediction and individual and global SHAP explanations.

PostgreSQL stored-customer prediction remains pending further validation.

This API documentation describes the implemented functionality and the observed test results of the Vantara Retail Customer Behaviour Prediction application.