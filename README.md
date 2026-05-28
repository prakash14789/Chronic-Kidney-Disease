# 🧬 CKD Clinical Intelligence Portal

[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0+-red.svg)](https://streamlit.io/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()

An enterprise-ready **Chronic Kidney Disease (CKD) Prediction and Intelligence System** designed for clinicians, medical researchers, and hospital IT systems. It combines clinical analytics, 10 machine learning models (with data-leakage safeguards), local explainable AI (SHAP), a secure patient database with audit logging, and an HL7 FHIR-compliant REST API.

---

## 📖 Table of Contents
1. [Core Features](#-core-features)
2. [Project Architecture](#-project-architecture)
3. [Technology Stack](#-technology-stack)
4. [Installation & Setup](#%EF%B8%8F-installation--setup)
5. [How to Run the Application](#-how-to-run-the-application)
6. [FastAPI Endpoints & EMR Integration](#-fastapi-endpoints--emr-integration)
7. [Security & HIPAA Compliance](#-security--hipaa-compliance)
8. [License & Disclaimer](#-license--disclaimer)

---

## ✨ Core Features

### 🖥️ 1. Interactive Streamlit Dashboard
Divided into 4 core sections using a premium, dark-mode glassmorphic interface:
* **📊 Data Audit:** Evaluates class imbalances (92/8 split), visualizes clinical feature distributions, and provides a correlation heatmap.
* **🚀 Model Analytics:** Trains and compares 10 models (Logistic Regression, RF, XGBoost, LightGBM, SVM, KNN, NB, Decision Trees, Extra Trees, Gradient Boosting) across two experiment paradigms:
  * **Experiment 1 (Full):** Baseline testing utilizing all clinical variables.
  * **Experiment 2 (No-Leakage):** Strict clinical testing excluding proxies like GFR, Serum Creatinine, BUN, Protein in Urine, and ACR to prevent data leakage.
  * **Tuning & Registry:** Optuna-based Bayesian parameter search and an interactive **Model Registry** listing saved model versions and metrics.
* **🧠 Interpretability (SHAP):** Local waterfall plots, global summary dot plots, and risk group contribution analysis (Lifestyle, Demographic, Clinical, etc.).
* **🏥 Clinical Tools:** Real-time patient diagnosis with dynamic slider interventions (What-If analysis), batch patient prediction (via CSV upload), KDIGO stage classifier (stages 1-5), and a 5-year progression timeline simulator.

### 🔌 2. REST API & EHR/EMR Integration
FastAPI server featuring:
* **OAuth2 Authentication:** JWT tokens for secure practitioner sessions.
* **FHIR Compliance:** Predicts and returns an HL7 FHIR-compliant `RiskAssessment` JSON Bundle for direct integration with EMRs (Epic, Cerner).
* **Asynchronous Tasks:** Celery/Redis queue for generating complex clinical PDF reports in the background.

### 🗃️ 3. Secure Patient Database
SQLAlchemy backend using an SQLite database with:
* **Encryption at Rest:** Sensitive Patient Identifiable Information (PII) like clinical actions and raw features are encrypted using a Fernet engine key.
* **Comprehensive Audit Trail:** Logs all practitioner actions (authentication, patient searches, PDF exports, risk predictions).

---

## 📂 Project Architecture

```
CKD/
├── Home.py                       # 🏠 Dashboard Main Entrypoint
├── api.py                        # 🔌 FastAPI Server (REST API)
├── app_tabs.py                   # 📊 Streamlit Render Functions
├── core_state.py                 # ⚙️ Sidebar config, CSS/JS injectors, & Run pipelines
├── data_processor.py             # 📦 Stratification, train-test splitting & encoders
├── model_trainer.py              # 🤖 Model pipelines (SMOTE, Scaling), SHAP, & Optuna Tuning
├── stage_predictor.py            # 🏗️ KDIGO multi-class stage predictor
├── visualizer.py                 # 📈 Plotly/Matplotlib charts and custom CSS templates
├── database.py                   # 🗃️ SQLite secure database, encryption, & audit logs
├── tasks.py                      # 📦 Celery asynchronous worker tasks
├── extrapolate_data.py           # 📊 Synthetic data generator (50k rows) with quality audits
├── report_generator.py           # 📄 Clinical PDF report generator
├── requirements.txt              # 📋 Python software dependencies
├── Dockerfile                    # 🐳 Container config
└── docker-compose.yml            # 🐳 Orchestration for FastAPI + Celery + Redis
```

---

## 🛠️ Technology Stack

* **Front-end:** Streamlit, HTML5, Vanilla CSS (Glassmorphic Theme), driver.js, particles.js, vanilla-tilt.js.
* **ML Engines:** Scikit-Learn, Imbalanced-Learn (SMOTE), XGBoost, LightGBM, Optuna.
* **XAI (Explainability):** SHAP (TreeExplainer & Explainer with predict_proba).
* **Backend API:** FastAPI, Uvicorn, Pydantic.
* **Database & Security:** SQLAlchemy, SQLAlchemy-Utils (Fernet Cryptography), SQLite, Passlib (Bcrypt).
* **Distributed System:** Celery, Redis.
* **Reporting:** FPDF2.
* **Testing:** Pytest.

---

## ⚙️ Installation & Setup

### Prerequisites
* Python 3.8 to 3.13 installed.
* SQLite (built-in).
* Redis server (optional, required for Celery background tasks).

### 1. Clone & Initialize Environment
```bash
git clone https://github.com/prakash14789/Chronic-Kidney-Disease.git
cd Chronic-Kidney-Disease
```

### 2. Set Up Virtual Environment
```bash
# Create venv
python -m venv venv

# Activate venv (Windows)
venv\Scripts\activate

# Activate venv (Mac/Linux)
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Application

### 🖥️ Running the Streamlit Dashboard
```bash
streamlit run Home.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.
* **Username:** `admin` | **Password:** `admin` (Administrator Role)
* **Username:** `client` | **Password:** `client` (Practitioner Role)

### 🔌 Running the FastAPI Server
```bash
uvicorn api:app --reload --port 8000
```
Open **[http://localhost:8000/docs](http://localhost:8000/docs)** to view the interactive Swagger API documentation.

### 🐳 Running via Docker Compose (Includes Redis + Celery)
To launch the complete enterprise pipeline (API, database, background workers, and message broker):
```bash
docker-compose up --build
```

### 🧪 Running Unit Tests
```bash
pytest
```

---

## 🔌 FastAPI Endpoints & EMR Integration

The API is fully authenticated and documented via Swagger. Core endpoints include:

* **`POST /token`**: Generates a JWT token using OAuth2 username/password flow.
* **`POST /predict`**: Takes patient clinical fields and returns the prediction score, risk level, assessment color, and clinical guidelines.
* **`POST /predict/fhir`**: Takes patient inputs and returns an HL7 FHIR R4 Bundle containing a `Patient` resource and a `RiskAssessment` resource.
* **`POST /predict/batch`**: Evaluates risk profiles for multiple patients concurrently.
* **`POST /report/async`**: Places a PDF report generation task in the Celery/Redis queue and returns a task tracker ID.

---

## 🛡️ Security & HIPAA Compliance

The codebase integrates modern standard security controls matching clinical software requirements:
1. **PII Encryption at Rest:** Patient names, addresses, or identifiers are not collected in raw form. Saved clinical recommendations and raw patient feature JSON dictionaries are encrypted using AES-256 Fernet keys before insertion into SQLite.
2. **Audit Logging:** Every critical model transaction, EMR export, or PDF generation is logged in a separate read-append `audit_logs` SQL table tracking practitioner IDs and execution details.
3. **Role-Based Access Control (RBAC):** Restricts interface accessibility based on user profile privileges (`admin` vs. `client`).

---

## ⚖️ License & Disclaimer

This project is licensed under the MIT License.

> **⚠️ CLINICAL DISCLAIMER:** This software is an intelligence portal built for clinical research, educational purposes, and workflow acceleration. It is not an FDA-approved diagnostic tool and must not be used as a replacement for professional medical diagnosis, advice, or treatment by a licensed healthcare practitioner.
