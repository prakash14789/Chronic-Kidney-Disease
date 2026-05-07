# 🧬 CKD Clinical Intelligence — Complete Project Documentation

> **What is this project?**  
> This is a **Chronic Kidney Disease (CKD) Prediction System**. It takes a patient's medical data (like age, blood pressure, blood sugar, etc.) and predicts whether they are at risk of having Chronic Kidney Disease. It also explains *why* the prediction was made, so doctors can trust and act on it.

---

## 📖 Table of Contents

1. [The Big Picture — What Does This Project Do?](#1-the-big-picture)
2. [The Dataset — Where Does the Data Come From?](#2-the-dataset)
3. [Project Files — What Does Each File Do?](#3-project-files)
4. [Tools & Libraries Used — What Software Powers This?](#4-tools--libraries-used)
5. [The Complete Flow — Step by Step](#5-the-complete-flow)
6. [How the Files Are Connected](#6-how-the-files-are-connected)
7. [Key Concepts Explained Simply](#7-key-concepts-explained-simply)
8. [The Dashboard — What the User Sees](#8-the-dashboard)
9. [How to Run the Project](#9-how-to-run-the-project)

---

## 1. The Big Picture

Imagine you go to a hospital and the doctor has your blood test results, your age, your lifestyle habits, etc. This project takes all that information and answers one question:

> **"Is this patient likely to have Chronic Kidney Disease?"**

But it doesn't just say "Yes" or "No". It also:
- Gives a **risk percentage** (e.g., "72% chance of CKD")
- Shows a **color-coded risk gauge** (green = safe, yellow = caution, red = danger)
- Explains **which factors** are pushing the risk up or down (e.g., "High blood pressure is increasing risk, but good diet is reducing it")
- Shows **what changes could reduce the risk** (e.g., "If blood pressure drops by 20, risk reduces by 5%")
- Finds **similar patients** from the dataset and shows their outcomes

All of this is displayed on a beautiful, interactive **web dashboard** that a doctor or researcher can use in their browser.

---

## 2. The Dataset

| Detail | Value |
|--------|-------|
| **File Name** | `Chronickidneydiseases.csv` |
| **Total Patients** | ~83,672 rows |
| **Total Features (columns)** | 55 columns |
| **Target Column** | `Diagnosis` (0 = No CKD, 1 = Has CKD) |

### What kind of information is in the data?

| Category | Example Columns | What They Mean |
|----------|----------------|----------------|
| **Patient Info** | Age, Gender, BMI | Basic details about the patient |
| **Blood Tests** | HbA1c, FastingBloodSugar, HemoglobinLevels | Results from blood work |
| **Kidney Markers** | GFR, SerumCreatinine, BUNLevels, ProteinInUrine, ACR | Direct kidney health indicators |
| **Blood Pressure** | SystolicBP, DiastolicBP | How hard the heart is pumping |
| **Lifestyle** | Smoking, PhysicalActivity, DietQuality, SleepQuality | Daily habits |
| **Family History** | FamilyHistoryKidneyDisease, FamilyHistoryHypertension, FamilyHistoryDiabetes | Genetic risk factors |
| **Other** | Edema (swelling), FatigueLevels, QualityOfLifeScore, Adherence (do they take medicines regularly?) | Additional health indicators |

### Why only 5,000 rows are used for training

The full dataset has ~83,672 patients. Training machine learning models on all of them would take a very long time. So we take a **smart sample of 5,000 patients** that keeps the same ratio of CKD vs non-CKD patients (this is called **stratified sampling** — like picking a smaller group that still represents the whole).

---

## 3. Project Files

Here's every file in the project and what it does:

```
CKD/
├── app.py                        ← 🖥️  The Dashboard (what users see)
├── data_processor.py             ← 📦  Loads & prepares the data
├── model_trainer.py              ← 🤖  Trains & evaluates AI models
├── visualizer.py                 ← 📊  Creates all charts & graphs
├── report_generator.py           ← 📄  Generates professional clinical PDF reports
├── chronic_kidney_disease.py     ← 📓  Original research script (standalone version)
├── Chronickidneydiseases.csv     ← 📄  The patient data file
├── requirements.txt              ← 📋  List of software needed
├── .gitignore                    ← 🚫  Files to not upload to GitHub
└── venv/                         ← 📂  Virtual environment (installed software)
```

### Detailed breakdown:

### 📦 `data_processor.py` — The Data Chef
**Role:** Prepares raw patient data for the AI models.

What it does step by step:
1. **Loads the CSV file** — Reads the 83,672-patient dataset
2. **Takes a stratified sample** — Picks 5,000 patients while keeping the CKD ratio balanced
3. **Removes useless columns** — Drops `PatientID` (just an ID number) and `RecommendedVisitsPerMonth` (not useful for prediction)
4. **Encodes text into numbers** — The "Adherence" column has text values like "Adherent" / "Non-Adherent". Computers need numbers, so it converts them (e.g., Adherent → 0, Non-Adherent → 1)
5. **Splits into train/test** — 80% of data is used to teach the model, 20% is kept aside to test how well it learned (like studying from a textbook but being tested on new questions)

### 🤖 `model_trainer.py` — The Brain
**Role:** Trains 10 different AI models and picks the best one.

What it does:
1. **Builds smart pipelines** — Each model is wrapped in a pipeline that handles data balancing (SMOTE) and scaling automatically
2. **Trains 10 models** — Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, KNN, Naive Bayes, Extra Trees, XGBoost, LightGBM
3. **Evaluates each model** — Measures accuracy, precision, recall, F1-score, ROC-AUC, and more
4. **Picks the best model** — Ranks models by "Balanced Accuracy" (a fairer metric than raw accuracy)
5. **Tunes the decision threshold** — Instead of just "above 50% = CKD", it finds the optimal cutoff point
6. **Generates SHAP explanations** — Explains WHY each prediction was made
7. **Finds similar patients** — For any new patient, finds the 5 most similar patients in the training data
8. **Runs what-if analysis** — Shows how changing one health factor would change the risk

### 📊 `visualizer.py` — The Artist
**Role:** Creates all the beautiful, interactive charts shown on the dashboard.

Charts it creates:
- Pie chart of CKD vs Non-CKD distribution
- ROC curves (how well each model separates CKD from non-CKD)
- Precision-Recall charts
- Correlation heatmap (which features are related)
- SHAP bar and summary plots (which features matter most)
- Risk gauge (speedometer-style risk display)
- Confusion matrix (where the model gets it right/wrong)
- Calibration curve (are the probabilities trustworthy?)
- What-if counterfactual chart
- Population risk distribution
- And more...

### 🖥️ `app.py` — The Dashboard
**Role:** The main file that runs the web application. It ties everything together.

It creates 8 tabs:
1. **📊 Data Audit** — Shows class distribution, correlation, and clinical distributions
2. **🚀 Experiment 1 (Full)** — Results using ALL features (including kidney markers)
3. **🛡️ Experiment 2 (No-Leakage)** — Results WITHOUT kidney markers (the honest test)
4. **📉 Comparison** — Side-by-side comparison showing the performance drop
5. **🎯 Threshold Tuning** — Interactive slider to adjust the decision boundary
6. **🧠 SHAP Interpretation** — Which features matter most globally
7. **🔬 Deep Analysis** — Calibration, error patterns, stability checks
8. **🏥 Patient Diagnosis** — Enter a patient's details and get a prediction with full explanation

### 📓 `chronic_kidney_disease.py` — The Original Script
**Role:** The standalone research script that was originally written in Google Colab. It does everything in one file (data loading, preprocessing, training, evaluation, plotting). The modular files above (`data_processor.py`, `model_trainer.py`, `visualizer.py`) are the cleaned-up, production-ready versions of this script.

---

## 4. Tools & Libraries Used

### Core Language
| Tool | What It Is | Why We Use It |
|------|-----------|---------------|
| **Python** | Programming language | The #1 language for data science and AI |

### Data Handling
| Library | What It Does | Simple Analogy |
|---------|-------------|----------------|
| **pandas** | Works with tables of data | Like a super-powered Excel |
| **numpy** | Does math on large arrays of numbers | A high-speed calculator |

### Machine Learning (The AI Brain)
| Library | What It Does | Simple Analogy |
|---------|-------------|----------------|
| **scikit-learn** | Provides 7 of the 10 AI models, plus tools for splitting data, measuring accuracy, scaling numbers | The Swiss Army knife of AI |
| **xgboost** | A very powerful tree-based model (often wins competitions) | A specialist doctor who's really good |
| **lightgbm** | Another powerful tree model, faster than XGBoost | An equally good specialist who works faster |
| **imbalanced-learn** | Handles the problem where CKD patients are fewer than non-CKD | Makes sure the AI doesn't ignore the minority group |

### Explainability
| Library | What It Does | Simple Analogy |
|---------|-------------|----------------|
| **shap** | Explains WHY the AI made a specific prediction | Like the AI showing its work on a math test |

### Visualization (Charts & Graphs)
| Library | What It Does | Simple Analogy |
|---------|-------------|----------------|
| **plotly** | Creates interactive, clickable charts | Charts you can hover over and zoom into |
| **matplotlib** | Creates static charts (used for SHAP plots) | Traditional printed charts |
| **seaborn** | Makes matplotlib charts prettier | A filter that makes charts look professional |

### Web Dashboard
| Library | What It Does | Simple Analogy |
|---------|-------------|----------------|
| **streamlit** | Turns Python code into a web application | Converts code into a website with buttons, sliders, and charts |

---

## 5. The Complete Flow — Step by Step

Here's exactly what happens when someone opens the dashboard:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        THE COMPLETE FLOW                            │
└─────────────────────────────────────────────────────────────────────┘

    STEP 1: LOAD DATA
    ─────────────────
    📄 Chronickidneydiseases.csv (83,672 patients)
                │
                ▼
    ┌───────────────────────┐
    │   data_processor.py   │  ← Loads the CSV file
    └───────────┬───────────┘
                │
                ▼
    STEP 2: PREPARE DATA
    ────────────────────
    • Take a stratified sample of 5,000 patients
    • Remove PatientID and RecommendedVisitsPerMonth
    • Convert "Adherence" text → numbers
    • Split into 80% Training + 20% Testing
                │
                ▼
    STEP 3: CREATE TWO VERSIONS OF THE DATA
    ────────────────────────────────────────
    ┌─────────────────────┐    ┌──────────────────────────────┐
    │   Full Features     │    │   No-Leakage Features        │
    │   (ALL 50+ columns) │    │   (removed GFR, Creatinine,  │
    │                     │    │    BUN, ProteinInUrine, ACR)  │
    └────────┬────────────┘    └──────────────┬───────────────┘
             │                                │
             ▼                                ▼
    STEP 4: TRAIN 10 AI MODELS (twice — once for each version)
    ──────────────────────────────────────────────────────────
    ┌──────────────────────────────────────────────┐
    │              model_trainer.py                 │
    │                                              │
    │  For EACH of the 10 models:                  │
    │  1. Balance the data (SMOTE)                 │
    │  2. Scale numbers if needed                  │
    │  3. Train the model on 80% data              │
    │  4. Test on the remaining 20%                │
    │  5. Measure how good it is                   │
    │  6. Pick the best one                        │
    └──────────────────┬───────────────────────────┘
                       │
                       ▼
    STEP 5: GENERATE EXPLANATIONS
    ─────────────────────────────
    • SHAP values → Which features matter most?
    • Threshold tuning → What's the best cutoff?
    • Error analysis → Where does it make mistakes?
                       │
                       ▼
    STEP 6: BUILD THE DASHBOARD
    ───────────────────────────
    ┌──────────────────────────────────────────────┐
    │                  app.py                       │
    │  + visualizer.py (creates all charts)         │
    │                                              │
    │  8 Interactive Tabs:                          │
    │  📊 Data Audit                               │
    │  🚀 Experiment 1 (Full Features)             │
    │  🛡️ Experiment 2 (No-Leakage)               │
    │  📉 Comparison                               │
    │  🎯 Threshold Tuning                         │
    │  🧠 SHAP Interpretation                      │
    │  🔬 Deep Analysis                            │
    │  🏥 Patient Diagnosis (Enter a patient!)     │
    └──────────────────────────────────────────────┘
                       │
                       ▼
    STEP 7: PATIENT DIAGNOSIS (Interactive)
    ───────────────────────────────────────
    Doctor enters patient details →
      → AI calculates risk probability →
        → Shows risk gauge, explanation, similar patients,
          what-if analysis, and protective/risk factors
```

---

## 6. How the Files Are Connected

```
                    ┌──────────────────────┐
                    │       app.py         │  ← The main dashboard file
                    │   (Streamlit App)    │
                    └──────┬───┬───┬───────┘
                           │   │   │
              ┌────────────┘   │   └────────────┐
              ▼                ▼                 ▼
   ┌──────────────────┐ ┌───────────────┐ ┌────────────────┐
   │ data_processor.py│ │model_trainer.py│ │ visualizer.py  │
   │                  │ │               │ │                │
   │ • Loads CSV      │ │ • 10 AI models│ │ • All charts   │
   │ • Samples data   │ │ • SMOTE       │ │ • Dark theme   │
   │ • Encodes text   │ │ • Evaluation  │ │ • Gauge meter  │
   │ • Train/Test     │ │ • SHAP        │ │ • SHAP plots   │
   │   split          │ │ • Threshold   │ │ • Heatmaps     │
   └────────┬─────────┘ │ • What-if     │ └────────────────┘
            │           │ • Similar pts │
            ▼           └───────┬───────┘
   ┌─────────────────┐          │
   │ CSV Data File   │          ▼
   │ (83,672 rows)   │  ┌──────────────┐
   └─────────────────┘  │ scikit-learn │
                        │ xgboost     │
                        │ lightgbm    │
                        │ shap        │
                        │ imbalanced- │
                        │ learn       │
                        └─────────────┘
```

### The connection in plain English:

1. **`app.py`** is the boss — it calls the other three files
2. **`data_processor.py`** is called first to load and prepare the data
3. **`model_trainer.py`** is called next to train models and get predictions
4. **`visualizer.py`** is called throughout to create charts from the results
5. The **CSV file** is the raw input that `data_processor.py` reads

---

## 7. Key Concepts Explained Simply

### 🔴 What is "Data Leakage"?
Imagine you're taking an exam, and someone accidentally shows you the answer key before the test. You'd score 99%, but you didn't actually learn anything. That's data leakage.

In our case, columns like **GFR** (Glomerular Filtration Rate) and **SerumCreatinine** are *direct indicators* of kidney disease. If the AI sees these, it's basically cheating — it's looking at the answer while making the prediction.

**What we do:** We run TWO experiments:
- **Experiment 1 (Full):** Uses all features including the "cheat" columns → Very high accuracy (but misleading)
- **Experiment 2 (No-Leakage):** Removes those 5 columns → Lower accuracy, but honest and trustworthy

### 🔵 What is SMOTE?
If 90% of patients are healthy and only 10% have CKD, the AI might just say "everyone is healthy" and still be 90% accurate. That's useless!

**SMOTE** (Synthetic Minority Over-sampling Technique) creates artificial copies of CKD patients so the AI sees a more balanced dataset. Think of it as giving the AI more examples of sick patients to learn from.

**Important:** SMOTE is applied ONLY to the training data (inside the pipeline), never to the test data. This prevents another form of cheating.

### 🟢 What is SHAP?
After the AI makes a prediction, SHAP answers: **"Why did you predict that?"**

For example, if the AI says "75% risk of CKD", SHAP might explain:
- High blood pressure → pushed risk UP by 15%
- Age 70 → pushed risk UP by 10%
- Good diet quality → pushed risk DOWN by 5%
- Regular exercise → pushed risk DOWN by 3%

This is crucial for doctors — they won't trust a "black box" that just says yes or no without explaining itself.

### 🟡 What is a Pipeline?
Instead of doing each step separately (balance data → scale numbers → train model), a **pipeline** packages all steps together like an assembly line. This ensures:
- The same steps happen every time
- Test data never "leaks" into training steps
- Everything is reproducible

### 🟠 What is Threshold Tuning?
By default, if the AI says "probability > 50%", we classify as CKD. But maybe 40% is a better cutoff (catching more sick patients) or 60% (fewer false alarms). **Threshold tuning** finds the sweet spot that maximizes overall performance.

### 🟣 What are the 10 AI Models?

| # | Model | How It Works (Simple) |
|---|-------|----------------------|
| 1 | **Logistic Regression** | Draws a straight line to separate CKD from non-CKD |
| 2 | **Decision Tree** | Asks yes/no questions like a flowchart |
| 3 | **Random Forest** | 100 decision trees vote together (wisdom of the crowd) |
| 4 | **Gradient Boosting** | Trees that learn from each other's mistakes |
| 5 | **SVM** | Finds the widest possible gap between the two groups |
| 6 | **KNN** | Looks at the 7 nearest patients and copies their diagnosis |
| 7 | **Naive Bayes** | Uses probability math (assumes features are independent) |
| 8 | **Extra Trees** | Like Random Forest but with more randomness |
| 9 | **XGBoost** | An advanced, competition-winning version of Gradient Boosting |
| 10 | **LightGBM** | A faster version of XGBoost |

---

## 8. The Dashboard — What the User Sees

The dashboard has **8 tabs**, each serving a specific purpose:

### Tab 1: 📊 Data Audit
- **Pie chart** showing CKD vs Non-CKD split
- **Checklist** confirming data hygiene (encoding done right, no leakage, SMOTE inside pipeline)
- **Correlation heatmap** showing which features are related
- **Box plots** of key clinical features

### Tab 2: 🚀 Experiment 1 (Full Features)
- Table of all 10 models with their scores
- ROC curves showing how well each model performs
- *Note: These results include leakage features, so accuracy is inflated*

### Tab 3: 🛡️ Experiment 2 (No-Leakage)
- Same as Tab 2, but without the "cheat" columns
- Confusion matrix showing exact right/wrong predictions
- *These are the trustworthy, research-grade results*

### Tab 4: 📉 Comparison
- Shows the **accuracy drop** when leakage features are removed
- Demonstrates why raw accuracy is misleading
- Precision/Recall/F1 chart for honest evaluation

### Tab 5: 🎯 Threshold Tuning
- Interactive slider to adjust the decision boundary
- Live metrics that update as you move the slider
- Sanity check button (shuffles labels to verify model isn't just memorizing)

### Tab 6: 🧠 SHAP Interpretation
- **Bar chart** of most important features globally
- **Dot plot** showing how each feature pushes predictions up or down
- **Grouped chart** showing contribution by category (Lifestyle, Clinical, etc.)

### Tab 7: 🔬 Deep Analysis
- **Feature direction chart** — what's different between CKD and non-CKD patients
- **Calibration curve** — are the predicted probabilities trustworthy?
- **Error analysis** — detailed breakdown of where the model fails
- **Population risk distribution** — histogram of risk scores
- **Stability check** — does the model give consistent results across different data splits?

### Tab 8: 🏥 Patient Diagnosis
This is the **star feature**. A doctor can:
1. **Enter patient details** (age, BMI, blood pressure, blood sugar, etc.)
2. Click "Generate Precision Diagnosis"
3. Get:
   - A **risk gauge** (0-100%)
   - A **color-coded assessment** (Low / Moderate / High Risk)
   - **Clinical recommendations** (e.g., "Schedule follow-up" or "Urgent referral")
   - **What-if analysis** (how changing each factor affects risk)
   - **Population comparison** (where this patient ranks among all patients)
   - **5 most similar patients** from the dataset
   - **Protective vs Risk factors** explained visually
   - **SHAP waterfall** showing exact feature contributions
   - **📄 Clinical PDF Report** — A downloadable, professional report for clinical records

---

## 9. How to Run the Project

### Prerequisites
- Python 3.8 or higher installed on your computer
- The dataset file `Chronickidneydiseases.csv` in the project folder

### Steps

1. **Open a terminal/command prompt** in the project folder

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

4. **Install all required libraries**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the dashboard**:
   ```bash
   streamlit run app.py
   ```

6. **Open your browser** — Streamlit will show a URL (usually `http://localhost:8501`). Open it and you'll see the dashboard!

### What happens when you run it:
1. The app loads the dataset (~83K patients)
2. Takes a sample of 5,000 patients
3. Trains 10 AI models twice (with and without leakage features)
4. This takes about 1-3 minutes on a standard computer
5. Results are cached, so it's instant on subsequent visits (unless you change the sample size)

---

> **Built with ❤️ for clinical research. This tool is for research/educational purposes and should not replace professional medical judgment.**
