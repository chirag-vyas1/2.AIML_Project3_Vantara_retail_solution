# Vantara — Retail Customer Behaviour Prediction

An end-to-end machine learning project for retail customer analytics, churn prediction, customer segmentation, customer lifetime value (CLV) analysis, anomaly detection, and explainable AI.

## 1. Project Overview

**Vantara Retail Customer Behaviour Prediction** is a machine-learning solution designed to help retail businesses understand customer behaviour, identify customers at risk of churn, segment customers into meaningful groups, and support data-driven business decisions.

The project brings together data preparation, feature engineering, machine-learning experimentation, model artifacts, explainability, API-based predictions, an interactive dashboard, and database integration.

The application is designed to run locally using Docker Compose, with separate services for the prediction API, interactive dashboard, and PostgreSQL database.

## 2. Key Features

- **Customer churn prediction:** Predict customer churn using trained classification models.
- **Customer segmentation:** Group customers into behavioural segments using clustering techniques.
- **Customer lifetime value analysis:** Support customer value analysis using CLV-related datasets and regression models.
- **Anomaly detection:** Identify potentially unusual customer behaviour using an autoencoder-based model.
- **Explainable AI:** Generate model explanations and feature-importance insights using SHAP, LIME, and partial dependence analysis.
- **REST API:** Serve predictions, model metadata, and explainability functionality through FastAPI.
- **Interactive dashboard:** Explore customer behaviour and model outputs through Streamlit.
- **Database integration:** Store and retrieve customer-related data and prediction records using PostgreSQL.
- **Containerized deployment:** Run the application services using Docker and Docker Compose.
- **Modular codebase:** Separate data processing, feature engineering, modelling, explainability, segmentation, and service utilities.

## 3. Technology Stack

| Category | Technologies |
|---|---|
| Programming language | Python 3.11 |
| Data analysis | Pandas, NumPy |
| Machine learning | Scikit-learn, XGBoost, LightGBM |
| Deep learning | TensorFlow/Keras |
| Explainable AI | SHAP, LIME, partial dependence analysis |
| API development | FastAPI, Uvicorn |
| Dashboard | Streamlit |
| Database | PostgreSQL |
| Database connectivity | Psycopg, SQLAlchemy |
| Data visualization | Matplotlib, Plotly |
| Model persistence | Joblib, Keras model artifacts |
| Containerization | Docker, Docker Compose |
| Configuration | YAML |

## 4. System Architecture

The application consists of three primary runtime services.

```text
                    Retail Customer Data
                             |
                             v
                 Data Processing and Features
                             |
                             v
                  Trained ML Model Artifacts
                             |
               +-------------+-------------+
               |                           |
               v                           v
        FastAPI Prediction API       Explainability
               |                           |
               +-------------+-------------+
                             |
               +-------------+-------------+
               |                           |
               v                           v
        Streamlit Dashboard          PostgreSQL
               |
               v
       Customer Insights and
       Prediction Results
```

**Runtime services**

- **API:** Exposes model-related functionality over HTTP.
- **Dashboard:** Provides an interactive interface for users.
- **Database:** Stores customer features and churn prediction records.

Docker Compose coordinates the services and configures the database dependency.

## 5. Project Structure

```text
customer-behaviour-prediction/
│
├── api/
│   ├── main.py
│   ├── routers/
│   │   ├── explainability.py
│   │   ├── metadata.py
│   │   └── predict.py
│   └── schemas/
│       └── prediction.py
│
├── config/
│   └── config.yaml
│
├── data/
│   ├── processed/
│   ├── interim/
│   └── raw/
│
├── frontend/
│   └── dashboard.py
│
├── model_artifacts/
│   ├── explainability/
│   ├── models/
│   └── preprocessing/
│
├── notebook/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_experiments.ipynb
│   ├── 04_customer_segmentation.ipynb
│   └── 05_explainibility.ipynb
│
├── src/
│   ├── data/
│   ├── explainability/
│   ├── features/
│   ├── models/
│   ├── segmentation/
│   ├── services/
│   └── utils/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-docker.txt
├── .dockerignore
└── .gitignore
```

The structure above highlights the main application components. Some local data, configuration, and generated files may be excluded from version control.

## 6. Prerequisites

Install the following before running the application:

- Windows, Linux, or macOS
- Git
- Docker Desktop on Windows or Docker Engine with the Docker Compose plugin on Linux
- Visual Studio Code (recommended)

Docker must be running before starting the application.

## 7. Installation and Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/chirag-vyas1/2.AIML_Project3_Vantara_retail_solution.git
cd 2.AIML_Project3_Vantara_retail_solution/customer-behaviour-prediction
```

### Step 2: Configure environment variables

Create a `.env` file in the application directory.

```dotenv
DB_HOST=db
DB_PORT=5432
DB_NAME=vantara_db
DB_USER=postgres
DB_PASSWORD=replace_with_a_strong_local_password
```

Replace the example password with a strong password for your environment.

**Security:** Never commit `.env` or real credentials to GitHub. The repository is configured to ignore environment files.

### Step 3: Start the application

From the directory containing `docker-compose.yml`, run:

```bash
docker compose up -d --build
```

This builds the application image and starts the database, API, and dashboard services.

### Step 4: Check service status

```bash
docker compose ps
```

Wait until the database reports healthy and the application services are running.

### Step 5: Open the application

| Component | Local URL |
|---|---|
| Streamlit dashboard | http://localhost:8501 |
| FastAPI API | http://localhost:8000 |
| Interactive API documentation | http://localhost:8000/docs |
| API metadata | http://localhost:8000/metadata |

The endpoints above reflect the current local Docker configuration. Individual API operations depend on the application's implemented routes.

## 8. API Functionality

The API is organized into separate routers.

| Router | Purpose |
|---|---|
| `api/routers/predict.py` | Customer prediction functionality |
| `api/routers/metadata.py` | Model metadata |
| `api/routers/explainability.py` | Customer-level and global SHAP explanations |

Use the interactive Swagger interface at `http://localhost:8000/docs` to inspect the available routes, request schemas, and responses.

Example metadata request:

```bash
curl http://localhost:8000/metadata
```

The running application has returned metadata identifying the active model as `xgboost_tuned_v2`, with 27 features.

## 9. Machine-Learning Models and Artifacts

The repository includes trained artifacts for multiple machine-learning tasks.

### Classification

- XGBoost
- Random Forest
- LightGBM
- Support Vector Machine (SVM)

### Deep learning and anomaly detection

- Churn artificial neural network
- Customer anomaly-detection autoencoder

### Regression

- Linear Regression
- Ridge Regression

### Customer segmentation

- Gaussian Mixture Model (GMM)

The `model_artifacts/` directory also contains preprocessing artifacts, model metadata, explainability outputs, and selected visualization files.

Model performance metrics and benchmark comparisons should be added after verifying the final evaluation results.

## 10. Explainable AI

Explainability is included to help interpret model behaviour rather than treating predictions as unexplained outputs.

- **SHAP:** Quantifies feature contributions to individual predictions and global feature importance.
- **LIME:** Provides local explanations for individual predictions.
- **Partial dependence analysis:** Helps explore how model predictions change with selected features.
- **Feature importance:** Supports interpretation of influential variables.

The explainability endpoints and saved artifacts are available within the API and `model_artifacts/explainability/` directories.

## 11. Data and Database

The project includes processed datasets and database integration utilities.

The PostgreSQL schema used by the current deployment includes these tables:

- `vantara.customers`
- `vantara.customer_features`
- `vantara.churn_predictions`

The local Docker deployment uses a persistent named volume for PostgreSQL data.

### Data handling

- Raw source datasets are excluded from version control.
- The large intermediate file `data/interim/cleaned_transactions.csv` is excluded from Git.
- Processed datasets required by the project are included selectively.
- Database dump files and local environment files are excluded from Git.

**Important:** A fresh clone does not automatically contain ignored raw data, the ignored intermediate dataset, or a database backup. Supply any required data through your authorized data preparation workflow. Restore a database backup only when one is available and appropriate for your environment.

## 12. Useful Docker Commands

Start or rebuild the application:

```bash
docker compose up -d --build
```

View service status:

```bash
docker compose ps
```

View all service logs:

```bash
docker compose logs
```

View API logs:

```bash
docker compose logs api
```

View dashboard logs:

```bash
docker compose logs dashboard
```

Restart the API:

```bash
docker compose restart api
```

Stop the services without deleting the persistent database volume:

```bash
docker compose down
```

**Warning:** Avoid `docker compose down -v` unless you intentionally want to remove the Compose-managed volumes and understand the data-loss implications.

## 13. Development Workflow

The project follows a modular machine-learning workflow:

1. Inspect and validate the input data.
2. Clean the data and handle data-quality issues.
3. Perform exploratory data analysis.
4. Engineer customer-level features.
5. Train and evaluate candidate models.
6. Save the selected models and preprocessing artifacts.
7. Generate explainability outputs.
8. Expose model functionality through FastAPI.
9. Access application functionality through Streamlit.
10. Persist relevant customer and prediction information in PostgreSQL.

The notebooks support experimentation, while the Python modules organize reusable functionality.

## 14. Current Project Status

The following components have been exercised in the local development environment:

- Docker image build and container startup
- PostgreSQL database restoration and persistence
- API startup and metadata endpoint
- API-to-PostgreSQL connectivity
- Streamlit dashboard access

These checks describe the local development environment and do not imply that the application has been deployed to a public production server.

## 15. Future Enhancements

Potential future improvements include:

- Automated data ingestion and retraining pipelines
- Model performance and drift monitoring
- Expanded API integration and automated testing
- CI/CD automation
- Cloud deployment
- Authentication and role-based access control
- Model evaluation reports and dashboard screenshots
- Comprehensive setup instructions for obtaining or generating required datasets

## 16. Author

**Chirag Vyas**

AI/ML Engineering | Data Science | Machine Learning | Explainable AI | Model Deployment

GitHub: [@chirag-vyas1](https://github.com/chirag-vyas1)

---
**Fresh-clone setup note:** This repository does not include all local datasets or a database backup. Before running the dashboard and database-dependent features in a fresh environment, provide any required input or intermediate datasets and initialize or restore the PostgreSQL database using the project's supported workflow. The local database contents from the development environment are not automatically included when cloning this repository.

*This project is intended for learning, experimentation, and retail analytics development. Predictions and customer insights should be validated before being used to make consequential business decisions.*
