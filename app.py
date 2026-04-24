import streamlit as st
import pandas as pd
import numpy as np
from data_processor import CKDDataProcessor
from model_trainer import CKDModelTrainer
from visualizer import CKDVisualizer

# Page Config
st.set_page_config(page_title="CKD Dual Experiment Pipeline", page_icon="🧬", layout="wide")

# Custom Styles
st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    .stTabs [aria-selected="true"] { background-color: #0d6efd !important; color: white !important; }
    .metric-card { background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #0d6efd; }
    </style>
    """, unsafe_allow_html=True)

# Initialize Modules
processor = CKDDataProcessor()
trainer = CKDModelTrainer()
viz = CKDVisualizer()

# Sidebar
with st.sidebar:
    st.title("🧬 Research Pipeline")
    st.image("https://cdn-icons-png.flaticon.com/512/3067/3067451.png", width=80)
    sample_size = st.slider("Select Stratified Sample Size", 1000, 10000, 5000)
    st.divider()
    st.info("This app follows your 'Dual Experiment' logic: Full Features vs Leakage-Free.")

# ── 1. LOAD & PREPARE DATA ──
@st.cache_data
def load_all_data(sample_n):
    # Load raw
    raw_df = processor.load_and_clean_data(sample_n=sample_n)
    
    # Experiment 1: Full Features (including leakage ones for comparison)
    # We temporarily bypass the removal of leakage features to show the 'Baseline'
    X_full = raw_df.drop(columns=["Diagnosis"])
    y_full = raw_df["Diagnosis"]
    X_tr_f, X_te_f, y_tr_f, y_te_f = processor.prepare_train_test(raw_df)
    
    # Experiment 2: Leakage-Free
    leakage_cols = ["GFR", "SerumCreatinine", "BUNLevels", "ProteinInUrine", "ACR"]
    df_nl = raw_df.drop(columns=leakage_cols, errors="ignore")
    X_tr_nl, X_te_nl, y_tr_nl, y_te_nl = processor.prepare_train_test(df_nl)
    
    return raw_df, (X_tr_f, X_te_f, y_tr_f, y_te_f), (X_tr_nl, X_te_nl, y_tr_nl, y_te_nl)

raw_df, exp_full, exp_nl = load_all_data(sample_size)
X_tr_f, X_te_f, y_tr_f, y_te_f = exp_full
X_tr_nl, X_te_nl, y_tr_nl, y_te_nl = exp_nl

# Main Interface
st.title("Chronic Kidney Disease (CKD) Classification")
st.markdown("### Dual Experiment Analysis: Full Features vs Leakage-Free")

tabs = st.tabs(["📊 Dataset Overview", "🚀 Experiment 1 (Full)", "🛡️ Experiment 2 (Leakage-Free)", "📉 Comparison & Insights"])

# --- TAB 1: OVERVIEW ---
with tabs[0]:
    st.header("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Sample Size", f"{len(raw_df):,}")
    col2.metric("CKD Cases", raw_df['Diagnosis'].sum())
    col3.metric("Non-CKD", len(raw_df) - raw_df['Diagnosis'].sum())
    
    st.plotly_chart(viz.plot_class_distribution(raw_df), use_container_width=True)
    st.dataframe(raw_df.head(10), use_container_width=True)

# --- TAB 2: EXPERIMENT 1 ---
with tabs[1]:
    st.header("Experiment 1: Full Features (Baseline)")
    st.warning("Note: This experiment includes clinical proxy markers (GFR, etc.) which usually result in near-perfect scores.")
    
    with st.spinner("Training models for Experiment 1..."):
        res_full, roc_full, pipes_full = trainer.run_experiment(X_tr_f, X_te_f, y_tr_f, y_te_f, label="FULL")
    
    st.dataframe(res_full.style.highlight_max(axis=0, subset=["Test Accuracy", "ROC-AUC"]), use_container_width=True)
    st.plotly_chart(viz.plot_roc_curves(roc_full), use_container_width=True)

# --- TAB 3: EXPERIMENT 2 ---
with tabs[2]:
    st.header("Experiment 2: Leakage-Free Features")
    st.success("Removed clinical markers: GFR, SerumCreatinine, BUNLevels, ProteinInUrine, ACR.")
    
    with st.spinner("Training models for Experiment 2..."):
        res_nl, roc_nl, pipes_nl = trainer.run_experiment(X_tr_nl, X_te_nl, y_tr_nl, y_te_nl, label="NO-LEAKAGE")
    
    st.dataframe(res_nl.style.highlight_max(axis=0, subset=["Test Accuracy", "ROC-AUC"]), use_container_width=True)
    st.plotly_chart(viz.plot_roc_curves(roc_nl), use_container_width=True)

# --- TAB 4: COMPARISON ---
with tabs[3]:
    st.header("Performance Drop Comparison")
    
    best_f = res_full.iloc[0]
    best_nl = res_nl.iloc[0]
    
    acc_drop = best_f["Test Accuracy"] - best_nl["Test Accuracy"]
    
    st.markdown(f"""
    <div class='metric-card'>
    <h4>Accuracy Inflation from Leakage: <strong>{acc_drop:+.2%}</strong></h4>
    <p>Removing the clinical markers shows the true predictive power of demographics and lifestyle factors.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Comparison Table (Exact like your code)
    comp_data = {
        "Metric": ["Best Model", "Test Accuracy", "Recall", "ROC-AUC"],
        "Full Features": [best_f["Model"], best_f["Test Accuracy"], best_f["Recall"], best_f["ROC-AUC"]],
        "No-Leakage": [best_nl["Model"], best_nl["Test Accuracy"], best_nl["Recall"], best_nl["ROC-AUC"]],
        "Drop": ["-", f"{best_f['Test Accuracy'] - best_nl['Test Accuracy']:+.4f}", 
                 f"{best_f['Recall'] - best_nl['Recall']:+.4f}", 
                 f"{best_f['ROC-AUC'] - best_nl['ROC-AUC']:+.4f}"]
    }
    st.table(pd.DataFrame(comp_data))

    st.plotly_chart(viz.plot_age_distribution(raw_df), use_container_width=True)
    
    st.subheader("Sanity Check")
    if st.button("Run Sanity Check on Best No-Leakage Model"):
        acc_shuffled = trainer.run_sanity_check(pipes_nl[best_nl["Model"]], X_te_nl, y_te_nl)
        st.metric("Shuffled Accuracy", f"{acc_shuffled:.2%}")
        if acc_shuffled < 0.6:
            st.success("Pass: Model is learning real patterns, not structural biases.")

st.markdown("---")
st.caption("Structured Modular Pipeline using your exact Research Source Code.")
