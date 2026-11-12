# Vantara – API Testing Report

**Project:** Retail Customer Behaviour Prediction  
**API:** Vantara Customer Behaviour Prediction API  
**Model:** `xgboost_tuned_v2`  
**Environment:** Docker Compose – Local  
**Testing Status:** 7 Passed, 1 Pending

## 1. Testing Environment

| Component | Details |
|---|---|
| API | FastAPI |
| Containerization | Docker Compose |
| Database | PostgreSQL 18 |
| Model | xgboost_tuned_v2 |
| API Base URL | http://localhost:8000 |
| API Documentation | http://localhost:8000/docs |

## 2. API Test Results

| Test ID | Test Description | Result |
|---|---|---|
| TC-01 | Docker services status | PASS |
| TC-02 | Health and metadata endpoints | PASS |
| TC-03 | Single customer prediction | PASS |
| TC-04 | JSON batch prediction | PASS |
| TC-05 | CSV batch prediction | PASS |
| TC-06 | Individual SHAP explanation | PASS |
| TC-07 | Global SHAP explanation | PASS |
| TC-08 | PostgreSQL stored-customer prediction | PENDING |

## 3. Detailed Test Observations

### TC-01: Docker Services

Verified API, dashboard and PostgreSQL containers using `docker compose ps`.

- API container: Running
- Dashboard container: Running
- PostgreSQL container: Running and healthy

**Result: PASS**

### TC-02: Health and Metadata

Verified:
- `GET /health`
- `GET /metadata`

Observed:
- API status: healthy
- Model name: `xgboost_tuned_v2`
- Model type: classification
- Number of features: 27

**Result: PASS**

### TC-03: Single Customer Prediction

Endpoint: `POST /predict`

Submitted a synthetic customer feature record containing all 27 model features.

Observed:
- Predicted class: 1
- Churn probability: 0.5665479898452759
- Model: `xgboost_tuned_v2`

**Result: PASS**

### TC-04: JSON Batch Prediction

Endpoint: `POST /predict/batch`

Submitted three synthetic customer profiles.

Observed:
- Total customers processed: 3
- All three predictions returned successfully.
- Predicted class for all three: 1.

Churn probabilities:
- Customer 1: 0.6180948
- Customer 2: 0.5884315
- Customer 3: 0.5708888

**Result: PASS**

### TC-05: CSV Batch Prediction

Endpoint: `POST /predict/batch/csv`

Uploaded a CSV containing three synthetic customer records.

Observed:
- HTTP status: 200
- Response content type: `text/csv`
- Output CSV successfully downloaded.
- Original feature columns preserved.
- Three output columns appended:
  - `Predicted_Class`
  - `Churn_Probability`
  - `Model_Name`
- CSV predictions matched the JSON batch predictions for the same records.

**Result: PASS**

### TC-06: Individual SHAP Explanation

Endpoint: `POST /explain/shap/customer`

Observed:
- Individual customer explanation returned successfully.
- 27 feature contributions returned.
- Base value and SHAP values returned.

Top absolute SHAP contributions:
1. Engagement_Score
2. Recency
3. Seasonal_Purchase_Concentration

**Result: PASS**

### TC-07: Global SHAP Explanation

Endpoint: `POST /explain/shap/global`

Submitted three synthetic customer profiles with `max_features=10`.

Observed:
- Three customers analysed.
- Global feature importance returned.
- Top-ranked features included Engagement_Score, Recency and Seasonal_Purchase_Concentration.

**Result: PASS**

### TC-08: PostgreSQL Stored-Customer Prediction

Endpoint: `POST /predict/customer/{customer_id}`

Status: PENDING.

Database inspection confirmed:
- Database: `vantara_db`
- Schema: `vantara`
- Tables:
  - `customers`
  - `customer_features`
  - `churn_predictions`

The table structures and available customer records have not yet been inspected, and the endpoint has not yet been tested.

**Result: PENDING – Not included in passed test count.**

## 4. Overall Test Summary

| Metric | Result |
|---|---:|
| Total planned API tests | 8 |
| Passed | 7 |
| Failed | 0 |
| Pending | 1 |
| Pass rate of executed tests | 100% |

## 5. Important Testing Notes

- Synthetic customer profiles were used for prediction endpoint validation.
- Successful predictions demonstrate API functionality, not real-world model accuracy.
- SHAP values represent model-output contributions and should not be interpreted as direct percentage-point changes or causal effects.
- PostgreSQL stored-customer integration remains pending.
- Negative API scenarios have not been included in the passed test count.

## 6. Conclusion

Seven API test cases were successfully executed against the running Dockerized Vantara application.

The health, metadata, single prediction, batch prediction, CSV prediction and SHAP explainability endpoints responded successfully.

PostgreSQL stored-customer prediction remains pending further database validation.