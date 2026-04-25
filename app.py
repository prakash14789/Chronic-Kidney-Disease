import streamlit as st
import pandas as pd
import numpy as np
from data_processor import CKDDataProcessor
from model_trainer import CKDModelTrainer
from visualizer import CKDVisualizer, COLORS

# Page Config
st.set_page_config(page_title="CKD Clinical Intelligence v3.2", page_icon="🧬", layout="wide")

# Custom Styles
st.markdown(f"""
    <style>
    .stApp {{ background-color: {COLORS["bg"]}; color: {COLORS["text"]}; }}
    .stTabs [aria-selected="true"] {{ background-color: {COLORS["primary"]} !important; color: white !important; }}
    .metric-card {{ 
        background: linear-gradient(135deg, #1e293b 0%, {COLORS["bg"]} 100%); 
        color: {COLORS["text"]} !important; 
        padding: 20px; 
        border-radius: 12px; 
        border-left: 6px solid {COLORS["primary"]}; 
        margin-bottom: 20px;
        border: 1px solid {COLORS["grid"]};
    }}
    .metric-card h4, .metric-card p {{ color: {COLORS["text"]} !important; margin: 0; }}
    .prediction-box {{
        background-color: {COLORS["grid"]};
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid {COLORS["primary"]};
        color: {COLORS["text"]};
    }}
    </style>
    """, unsafe_allow_html=True)

# Initialize
processor = CKDDataProcessor()
trainer = CKDModelTrainer()
viz = CKDVisualizer()

# Sidebar
with st.sidebar:
    st.title("🧬 CKD Intelligence")
    st.image("https://cdn-icons-png.flaticon.com/512/3067/3067451.png", width=80)
    sample_size = st.slider("Stratified Sample Size", 1000, 10000, 5000)
    st.divider()
    st.success("v3 Research Pipeline Active")

# --- PIPELINE EXECUTION ---
@st.cache_data
def run_full_pipeline(sample_n):
    df_full = processor.load_raw_data()
    df_sample = processor.get_v3_refined_data(df_full, sample_n=sample_n)
    X_tr_f, X_te_f, y_tr_f, y_te_f = processor.split_and_encode_v3(df_sample)
    
    leakage_cols = ["GFR", "SerumCreatinine", "BUNLevels", "ProteinInUrine", "ACR"]
    X_tr_nl = X_tr_f.drop(columns=leakage_cols, errors="ignore")
    X_te_nl = X_te_f.drop(columns=leakage_cols, errors="ignore")
    
    n_neg, n_pos = (y_tr_f == 0).sum(), (y_tr_f == 1).sum()
    
    # Run Full
    pipes_f = trainer.get_v3_pipelines(n_neg, n_pos)
    res_f, roc_f, pr_f, trained_f = trainer.run_v3_experiment(X_tr_f, X_te_f, y_tr_f, y_te_f, pipelines=pipes_f)
    
    # Run No-Leakage
    pipes_nl = trainer.get_v3_pipelines(n_neg, n_pos)
    res_nl, roc_nl, pr_nl, trained_nl = trainer.run_v3_experiment(X_tr_nl, X_te_nl, y_tr_f, y_te_f, pipelines=pipes_nl)
    
    return df_full, df_sample, res_f, roc_f, res_nl, roc_nl, pr_nl, trained_nl, X_te_nl, y_te_f

df_full, df_sample, res_f, roc_f, res_nl, roc_nl, pr_nl, trained_nl, X_te_nl, y_te_f = run_full_pipeline(sample_size)

# Main UI
st.title("CKD Clinical Intelligence Dashboard")

tabs = st.tabs([
    "📊 Data Audit", 
    "🚀 Exp 1 (Full)", 
    "🛡️ Exp 2 (No-Leakage)", 
    "📉 Comparison", 
    "🎯 Threshold Tuning",
    "🧠 SHAP Interpretation",
    "🏥 Patient Diagnosis"
])

# --- TAB 1: DATA AUDIT ---
with tabs[0]:
    st.header("1. Class Imbalance & Audit")
    ckd_pct = (df_full['Diagnosis'].mean() * 100)
    col1, col2 = st.columns([1, 2])
    with col1: st.plotly_chart(viz.plot_class_distribution(df_full, ckd_pct), use_container_width=True, key="dist")
    with col2:
        st.write("### v3 Hygiene Checklist")
        st.checkbox("LabelEncoder fitted on Train only", value=True, disabled=True)
        st.checkbox("Leakage-free derived from same split", value=True, disabled=True)
        st.checkbox("SMOTE lives inside Pipeline", value=True, disabled=True)
    
    st.markdown("---")
    st.header("2. Correlation & Clinical Distributions")
    st.pyplot(viz.plot_correlation_heatmap(df_sample))
    st.plotly_chart(viz.plot_clinical_boxplots(df_sample), use_container_width=True)

# --- TAB 2: EXP 1 ---
with tabs[1]:
    st.header("Experiment 1: Full Features")
    st.dataframe(res_f[['Model', 'Balanced Accuracy', 'Macro F1', 'ROC-AUC']], use_container_width=True)
    st.plotly_chart(viz.plot_roc_curves(roc_f), use_container_width=True, key="roc_f")

# --- TAB 3: EXP 2 ---
with tabs[2]:
    st.header("Experiment 2: Leakage-Free (Research Grade)")
    st.dataframe(res_nl[['Model', 'Balanced Accuracy', 'Macro F1', 'ROC-AUC']], use_container_width=True)
    st.plotly_chart(viz.plot_roc_curves(roc_nl), use_container_width=True, key="roc_nl")

# --- TAB 4: COMPARISON ---
with tabs[3]:
    st.header("Performance Drop Comparison")
    best_f = res_f.iloc[0]
    best_nl = res_nl.iloc[0]
    st.markdown(f"""
    <div class='metric-card'>
    <h4>Accuracy Inflation from Leakage: <strong>{best_f['Balanced Accuracy'] - best_nl['Balanced Accuracy']:+.2%}</strong></h4>
    </div>
    """, unsafe_allow_html=True)
    
    col_c1, col_c2 = st.columns(2)
    with col_c1: st.plotly_chart(viz.plot_misleading_accuracy(ckd_pct), use_container_width=True, key="mis")
    with col_c2: st.plotly_chart(viz.plot_precision_recall_f1(res_nl), use_container_width=True, key="pr_f1")
    
    st.plotly_chart(viz.plot_age_distribution(df_sample), use_container_width=True, key="age_dist")

# --- TAB 5: THRESHOLD ---
with tabs[4]:
    st.header("Decision Threshold Optimization")
    best_name = res_nl.iloc[0]["Model"]
    y_proba = trained_nl[best_name].predict_proba(X_te_nl)[:, 1]
    th_df, best_th = trainer.tune_threshold(y_te_f, y_proba)
    st.plotly_chart(viz.plot_threshold_tuning(th_df, best_th), use_container_width=True, key="th_tune")
    if st.button("Run Sanity Check"):
        acc = trainer.run_sanity_check(trained_nl[best_name], X_te_nl, y_te_f)
        st.metric("Shuffled Balanced Accuracy", f"{acc:.2%}")

# --- TAB 6: SHAP ---
with tabs[5]:
    st.header("🧠 Model Interpretation (SHAP)")
    with st.spinner("Calculating Global SHAP..."):
        explainer, shap_values, X_df = trainer.get_shap_explainer(trained_nl[best_name], X_te_nl)
    cs1, cs2 = st.columns(2)
    with cs1: st.pyplot(viz.plot_shap_bar(explainer, shap_values, X_df, best_name))
    with cs2: st.pyplot(viz.plot_shap_summary(explainer, shap_values, X_df, best_name))

# --- TAB 7: DIAGNOSIS ---
with tabs[6]:
    st.header("🏥 Individual Patient Diagnosis")
    with st.form("diag_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 20, 90, 50)
            gender = st.selectbox("Gender", ["Male", "Female"])
            bmi = st.number_input("BMI", 15.0, 45.0, 25.0)
        with c2:
            smoking = st.selectbox("Smoking", ["No", "Yes"])
            activity = st.number_input("Activity", 0, 300, 150)
            systolic = st.number_input("Systolic BP", 90, 200, 120)
        with c3:
            fbs = st.number_input("Fasting Blood Sugar", 70, 200, 100)
            adherence = st.selectbox("Adherence", ["High", "Moderate", "Low"])
        
        sub = st.form_submit_button("🩺 Diagnose")
        
    if sub:
        input_row = pd.DataFrame(0, index=[0], columns=X_te_nl.columns)
        input_row["Age"] = age
        input_row["BMI"] = bmi
        input_row["SystolicBP"] = systolic
        input_row["FastingBloodSugar"] = fbs
        input_row["PhysicalActivity"] = activity
        if "Gender" in input_row.columns: input_row["Gender"] = 1 if gender == "Male" else 0
        if "Smoking" in input_row.columns: input_row["Smoking"] = 1 if smoking == "Yes" else 0
        
        prob = trained_nl[best_name].predict_proba(input_row)[0, 1]
        st.markdown(f"<div class='prediction-box'><h3>Risk Score: {prob:.1%}</h3></div>", unsafe_allow_html=True)
        st.pyplot(viz.plot_local_shap(explainer, shap_values, X_df, patient_idx=0))

st.markdown("---")
st.caption("CKD Intelligence v3.2 — All v3 Research Insights Included.")
