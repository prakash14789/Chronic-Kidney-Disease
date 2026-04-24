import streamlit as st
import pandas as pd
import numpy as np
from data_processor import CKDDataProcessor
from model_trainer import CKDModelTrainer
from visualizer import CKDVisualizer

# Page Config
st.set_page_config(page_title="CKD Intelligence Dashboard", page_icon="🧬", layout="wide")

# Custom Styles
st.markdown("""
    <style>
    .stTabs [aria-selected="true"] { background-color: #0d6efd !important; color: white !important; }
    .metric-card { 
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
        color: white !important; 
        padding: 20px; 
        border-radius: 12px; 
        border-left: 6px solid #3b82f6; 
        margin-bottom: 20px;
    }
    .metric-card h4, .metric-card p { color: white !important; margin: 0; }
    .section-header { border-bottom: 2px solid #3b82f6; padding-bottom: 10px; margin-top: 30px; margin-bottom: 20px; }
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
    st.info("Full Research Pipeline with SHAP Interpretation.")

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
    pipes_nl = trainer.get_v3_pipelines(n_neg, n_pos)
    res_nl, roc_nl, pr_nl, trained_nl = trainer.run_v3_experiment(X_tr_nl, X_te_nl, y_tr_f, y_te_f, pipelines=pipes_nl)
    
    return df_full, df_sample, res_nl, roc_nl, pr_nl, trained_nl, X_te_nl, y_te_f

df_full, df_sample, res_nl, roc_nl, pr_nl, trained_nl, X_te_nl, y_te_f = run_full_pipeline(sample_size)

# Main UI
st.title("CKD Clinical Intelligence Dashboard")

tabs = st.tabs(["📊 Performance Command Center", "🧠 Model Interpretation (SHAP)", "🎯 Threshold Tuning"])

# --- TAB 1: COMMAND CENTER ---
with tabs[0]:
    st.markdown("<h2 class='section-header'>1. Dataset Overview</h2>", unsafe_allow_html=True)
    ckd_pct = (df_full['Diagnosis'].mean() * 100)
    col1, col2, col3 = st.columns(3)
    with col1: st.plotly_chart(viz.plot_class_distribution(df_full, ckd_pct), use_container_width=True, key="dist")
    with col2: st.plotly_chart(viz.plot_misleading_accuracy(ckd_pct), use_container_width=True, key="mis")
    with col3: st.plotly_chart(viz.plot_age_distribution(df_sample), use_container_width=True, key="age")

    st.markdown("<h2 class='section-header'>2. Model Performance Analysis</h2>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("Model Comparison (Detailed Metrics)")
        st.dataframe(res_nl[['Model', 'Balanced Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1']], use_container_width=True)
    with col_b:
        st.plotly_chart(viz.plot_precision_recall_f1(res_nl), use_container_width=True, key="metrics_bar")

    col_c, col_d = st.columns(2)
    with col_c: st.plotly_chart(viz.plot_roc_curves(roc_nl), use_container_width=True, key="roc")
    with col_d: st.plotly_chart(viz.plot_pr_curves(pr_nl), use_container_width=True, key="pr")

# --- TAB 2: SHAP INTERPRETATION ---
with tabs[1]:
    st.header("🧠 Model Interpretation with SHAP")
    best_name = res_nl.iloc[0]["Model"]
    st.write(f"Analyzing the top model: **{best_name}**")
    
    with st.spinner("Calculating SHAP values (this may take a moment)..."):
        explainer, shap_values, X_df = trainer.get_shap_explainer(trained_nl[best_name], X_te_nl)
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.subheader("Feature Importance (Global)")
        st.pyplot(viz.plot_shap_bar(explainer, shap_values, X_df, best_name))
    with col_s2:
        st.subheader("Feature Impact (Bee Swarm)")
        st.pyplot(viz.plot_shap_summary(explainer, shap_values, X_df, best_name))

    st.info("SHAP (SHapley Additive exPlanations) values show how much each feature contributes to the prediction. Red dots indicate high feature values, blue dots indicate low feature values.")

# --- TAB 3: THRESHOLD ---
with tabs[2]:
    st.header("Decision Threshold Optimization")
    y_proba = trained_nl[best_name].predict_proba(X_te_nl)[:, 1]
    th_df, best_th = trainer.tune_threshold(y_te_f, y_proba)
    st.plotly_chart(viz.plot_threshold_tuning(th_df, best_th), use_container_width=True, key="th")
    
    col_metric, col_sanity = st.columns(2)
    with col_metric:
        st.markdown(f"<div class='metric-card'><h4>Optimal Threshold: {best_th}</h4><p>Balanced detection strategy.</p></div>", unsafe_allow_html=True)
    with col_sanity:
        if st.button("Run Sanity Check"):
            acc = trainer.run_sanity_check(trained_nl[best_name], X_te_nl, y_te_f)
            st.metric("Shuffled Balanced Accuracy", f"{acc:.2%}")

st.markdown("---")
st.caption("Consolidated Research Dashboard with SHAP and Multi-Metric Analysis.")
