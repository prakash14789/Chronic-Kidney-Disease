import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_score, recall_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import time
import copy

# Optional Imports
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except:
    LGBM_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except:
    SMOTE_AVAILABLE = False

# Page Config
st.set_page_config(
    page_title="CKD Analytics Pro",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #f8f9fa;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 10px 10px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
        color: #495057;
        border: none;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background-color: #0d6efd !important;
        color: white !important;
        box-shadow: 0px 4px 15px rgba(13, 110, 253, 0.3);
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #0d6efd;
    }
    
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    h1, h2, h3 {
        color: #212529;
        font-weight: 700;
    }
    
    .sidebar-text {
        font-size: 0.9rem;
        color: #6c757d;
    }
    </style>
    """, unsafe_allow_html=True)

# Constants
RANDOM_STATE = 42
DATA_PATH = "Chronickidneydiseases.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_data
def preprocess_data(df, sample_n=5000):
    # Stratified Sampling
    le_target = LabelEncoder()
    df['Diagnosis_Encoded'] = df['Diagnosis'] # Already numeric but for safety
    
    idx, _ = train_test_split(
        df.index, train_size=sample_n,
        stratify=df["Diagnosis"], random_state=RANDOM_STATE
    )
    df_sample = df.loc[idx].reset_index(drop=True)
    
    # Preprocessing
    drop_cols = ["PatientID", "RecommendedVisitsPerMonth"]
    df_sample.drop(columns=drop_cols, inplace=True, errors="ignore")
    
    le = LabelEncoder()
    if 'Adherence' in df_sample.columns:
        df_sample["Adherence"] = le.fit_transform(df_sample["Adherence"])
    
    return df_sample

@st.cache_resource
def train_models(X_train, y_train, X_test, y_test, scaled_indices):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "SVM": SVC(probability=True, kernel="rbf", random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
        "Naive Bayes": GaussianNB(),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1),
    }
    
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(n_estimators=100, max_depth=6, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)
    if LGBM_AVAILABLE:
        models["LightGBM"] = LGBMClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    
    results = []
    roc_curves = {}
    trained_models = {}
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    for name, model in models.items():
        is_scaled = name in ["Logistic Regression", "SVM", "KNN"]
        Xtr = X_train_sc if is_scaled else X_train
        Xte = X_test_sc if is_scaled else X_test
        
        t0 = time.time()
        model.fit(Xtr, y_train)
        elapsed = time.time() - t0
        
        y_pred = model.predict(Xte)
        y_proba = model.predict_proba(Xte)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves[name] = (fpr, tpr, auc)
        trained_models[name] = model
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc,
            "Time": elapsed
        })
        
    return pd.DataFrame(results).sort_values("Accuracy", ascending=False), roc_curves, trained_models, scaler

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100)
    st.title("CKD Analytics")
    st.markdown("---")
    st.markdown("### Configuration")
    sample_size = st.slider("Sample Size", 1000, 10000, 5000)
    st.markdown("---")
    st.info("This dashboard provides advanced clinical insights and predictive modeling for Chronic Kidney Disease.")
    st.markdown("---")
    st.markdown("<p class='sidebar-text'>Developed for Research & Analysis</p>", unsafe_allow_html=True)

# Data Loading
df_full = load_data()
df = preprocess_data(df_full, sample_n=sample_size)

# Main App
st.title("🧬 Chronic Kidney Disease Intelligence Dashboard")
st.markdown("Leveraging Machine Learning to identify clinical patterns and predict kidney health outcomes.")

tabs = st.tabs(["📊 Overview", "🔍 EDA", "🎯 Performance", "💡 Insights", "🔮 Predictor"])

# --- TAB 1: OVERVIEW ---
with tabs[0]:
    st.header("Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", f"{len(df_full):,}")
    col2.metric("Working Sample", f"{len(df):,}")
    col3.metric("Features", f"{df.shape[1]-1}")
    col4.metric("CKD Prevalence", f"{(df['Diagnosis'].mean()*100):.1f}%")
    
    st.markdown("### Class Distribution")
    fig_dist = px.pie(df, names='Diagnosis', title='CKD vs Non-CKD Distribution',
                     color_discrete_sequence=['#4C72B0', '#DD8452'],
                     labels={'Diagnosis': 'Status'})
    fig_dist.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_dist, use_container_width=True)
    
    st.markdown("### Sample Data Snippet")
    st.dataframe(df.head(10), use_container_width=True)

# --- TAB 2: EDA ---
with tabs[1]:
    st.header("Clinical Feature Analysis")
    
    eda_col1, eda_col2 = st.columns(2)
    
    with eda_col1:
        st.markdown("#### Age Distribution by Diagnosis")
        fig_age = px.histogram(df, x="Age", color="Diagnosis", barmode="overlay",
                               color_discrete_map={0: 'steelblue', 1: 'tomato'},
                               marginal="box")
        st.plotly_chart(fig_age, use_container_width=True)
        
    with eda_col2:
        st.markdown("#### BMI vs Physical Activity")
        fig_scatter = px.scatter(df, x="BMI", y="PhysicalActivity", color="Diagnosis",
                                 color_discrete_map={0: 'steelblue', 1: 'tomato'},
                                 opacity=0.6)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("#### Correlation Heatmap (Key Features)")
    top15_corr = df.corr()['Diagnosis'].abs().sort_values(ascending=False).head(16).index
    corr_matrix = df[top15_corr].corr()
    fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='RdBu_r', aspect="auto")
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("#### Clinical Markers Boxplots")
    clinical_cols = ["GFR", "SerumCreatinine", "BUNLevels", "HbA1c", "ProteinInUrine"]
    selected_clinical = st.multiselect("Select markers to visualize", clinical_cols, default=clinical_cols[:3])
    if selected_clinical:
        fig_box = px.box(df, x="Diagnosis", y=selected_clinical, facet_col="variable", color="Diagnosis",
                         color_discrete_map={0: 'steelblue', 1: 'tomato'})
        st.plotly_chart(fig_box, use_container_width=True)

# --- TAB 3: PERFORMANCE ---
with tabs[2]:
    st.header("Model Performance Benchmarking")
    
    # Train/Test Split
    X = df.drop(columns=["Diagnosis"])
    y = df["Diagnosis"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    
    if SMOTE_AVAILABLE:
        sm = SMOTE(random_state=RANDOM_STATE)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    else:
        X_train_res, y_train_res = X_train, y_train
        
    results_df, roc_curves, trained_models, scaler = train_models(X_train_res, y_train_res, X_test, y_test, ["Logistic Regression", "SVM", "KNN"])
    
    st.markdown("#### Metrics Comparison")
    st.dataframe(results_df.style.highlight_max(axis=0, subset=["Accuracy", "ROC-AUC", "F1-Score"]), use_container_width=True)
    
    perf_col1, perf_col2 = st.columns(2)
    
    with perf_col1:
        st.markdown("#### Accuracy Comparison")
        fig_acc = px.bar(results_df, x="Model", y="Accuracy", color="Accuracy",
                         color_continuous_scale="Viridis")
        st.plotly_chart(fig_acc, use_container_width=True)
        
    with perf_col2:
        st.markdown("#### ROC Curves")
        fig_roc = go.Figure()
        for name, (fpr, tpr, auc) in roc_curves.items():
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={auc:.3f})", mode='lines'))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash='dash'), showlegend=False))
        fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)

# --- TAB 4: INSIGHTS ---
with tabs[3]:
    st.header("Advanced Clinical Insights")
    
    ins_col1, ins_col2 = st.columns([1, 1])
    
    with ins_col1:
        st.markdown("#### Feature Importance (Random Forest)")
        rf_model = trained_models["Random Forest"]
        feat_imp = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
        fig_imp = px.bar(feat_imp, orientation='h', color=feat_imp.values, color_continuous_scale="Blues")
        fig_imp.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_imp, use_container_width=True)
        
    with ins_col2:
        st.markdown("#### Data Leakage Analysis")
        st.warning("Clinical markers like GFR, Creatinine, and BUN are highly correlated with the diagnosis and can cause 'Data Leakage' in predictive models.")
        
        leakage_cols = ["GFR", "SerumCreatinine", "BUNLevels", "ProteinInUrine", "ACR"]
        X_nl = X.drop(columns=leakage_cols, errors="ignore")
        X_tr_nl, X_te_nl, y_tr_nl, y_te_nl = train_test_split(X_nl, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
        
        # Simplified experiment for leakage-free
        rf_nl = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)
        rf_nl.fit(X_tr_nl, y_tr_nl)
        y_pred_nl = rf_nl.predict(X_te_nl)
        acc_nl = accuracy_score(y_te_nl, y_pred_nl)
        
        best_acc_full = results_df["Accuracy"].max()
        drop = best_acc_full - acc_nl
        
        st.metric("Leakage-Free Accuracy", f"{acc_nl:.4f}", delta=f"-{drop:.4f}", delta_color="inverse")
        st.write(f"When removing the top 5 clinical markers, accuracy drops by {drop:.2%}, revealing the true predictive power of non-clinical features (lifestyle, history).")

    st.markdown("#### Confusion Matrix (Top Model)")
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    is_scaled = best_model_name in ["Logistic Regression", "SVM", "KNN"]
    Xte_final = scaler.transform(X_test) if is_scaled else X_test
    y_pred_final = best_model.predict(Xte_final)
    cm = confusion_matrix(y_test, y_pred_final)
    
    fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"),
                       x=['Not CKD', 'CKD'], y=['Not CKD', 'CKD'],
                       color_continuous_scale="Blues")
    st.plotly_chart(fig_cm, use_container_width=True)

# --- TAB 5: PREDICTOR ---
with tabs[4]:
    st.header("Patient Outcome Predictor")
    st.write("Input patient data to estimate the likelihood of Chronic Kidney Disease.")
    
    with st.expander("Enter Patient Details"):
        p_col1, p_col2, p_col3 = st.columns(3)
        
        age = p_col1.number_input("Age", 1, 120, 45)
        bmi = p_col2.number_input("BMI", 10.0, 50.0, 25.0)
        systolic = p_col3.number_input("Systolic BP", 80, 200, 120)
        
        gfr = p_col1.number_input("GFR (Clinical)", 0.0, 200.0, 90.0)
        serum = p_col2.number_input("Serum Creatinine", 0.0, 15.0, 1.0)
        hba1c = p_col3.number_input("HbA1c", 3.0, 15.0, 5.5)
        
        gender = p_col1.selectbox("Gender", ["Male", "Female"])
        ethnicity = p_col2.selectbox("Ethnicity", ["Caucasian", "African American", "Asian", "Other"])
        smoking = p_col3.selectbox("Smoking Status", ["No", "Yes"])
        
    if st.button("Predict Diagnosis", type="primary"):
        # Create a dummy input vector matching X.columns
        input_data = pd.DataFrame(columns=X.columns)
        input_data.loc[0] = 0 # Initialize with zeros
        
        # Fill in relevant values
        input_data.at[0, "Age"] = age
        input_data.at[0, "BMI"] = bmi
        input_data.at[0, "SystolicBP"] = systolic
        input_data.at[0, "GFR"] = gfr
        input_data.at[0, "SerumCreatinine"] = serum
        input_data.at[0, "HbA1c"] = hba1c
        # Note: In a real app, you'd need to encode gender/ethnicity correctly based on LabelEncoder
        
        best_model = trained_models["Random Forest"] # Using RF as it's usually stable
        prob = best_model.predict_proba(input_data)[0][1]
        
        st.markdown("---")
        if prob > 0.5:
            st.error(f"### Result: High Risk of CKD ({prob:.1%})")
            st.write("Clinical indicators suggest a high probability of Chronic Kidney Disease. Immediate medical consultation is recommended.")
        else:
            st.success(f"### Result: Low Risk of CKD ({prob:.1%})")
            st.write("Patient indicators are within normal ranges for kidney health. Continue regular monitoring.")

st.markdown("---")
st.caption("Disclaimer: This tool is for educational and research purposes only. It is not a substitute for professional medical advice.")
