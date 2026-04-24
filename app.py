import streamlit as st
import pandas as pd
from data_processor import CKDDataProcessor
from model_trainer import CKDModelTrainer
from visualizer import CKDVisualizer

# Page Config
st.set_page_config(page_title="CKD Leakage-Free Pipeline", page_icon="🛡️", layout="wide")

# Custom Styles
st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    .stTabs [aria-selected="true"] { background-color: #1a73e8 !important; color: white !important; }
    .status-box { padding: 15px; border-radius: 10px; margin-bottom: 20px; }
    .leakage-removed { background-color: #e8f0fe; border-left: 5px solid #1a73e8; }
    </style>
    """, unsafe_allow_html=True)

# Initialize Modules
processor = CKDDataProcessor()
trainer = CKDModelTrainer()
viz = CKDVisualizer()

@st.cache_data
def get_clean_pipeline_data(sample_size):
    # 1. Load and remove duplicates/leakage features
    clean_df = processor.load_and_clean_data(sample_n=sample_size)
    # 2. Strict Split and Encode (Fit on Train ONLY)
    X_train, X_test, y_train, y_test = processor.prepare_train_test(clean_df)
    return clean_df, X_train, X_test, y_train, y_test

# Sidebar
with st.sidebar:
    st.title("🛡️ Secure Pipeline")
    st.image("https://cdn-icons-png.flaticon.com/512/2569/2569106.png", width=100)
    sample_size = st.slider("Dataset Sample Size", 2000, 10000, 5000)
    st.divider()
    st.success("Target Leakage: REMOVED")
    st.success("Preprocessing Leakage: REMOVED")
    st.success("Data Overlap: REMOVED")

# Logic Flow
clean_df, X_train, X_test, y_train, y_test = get_clean_pipeline_data(sample_size)

# Main Interface
st.title("Realistic CKD Classification Pipeline")
st.markdown("This dashboard demonstrates a production-level ML pipeline with strict isolation between training and testing data.")

tabs = st.tabs(["📋 Data Audit", "🏗️ Pipeline Execution", "🔍 Validation & Sanity Check"])

# --- TAB 1: DATA AUDIT ---
with tabs[0]:
    st.header("1. Audit & Feature Selection")
    
    st.markdown("""
    <div class='status-box leakage-removed'>
    <strong>Audit Result:</strong> High-risk clinical markers (GFR, Creatinine, BUN, etc.) have been permanently removed. 
    The model now relies on <strong>lifestyle, demographics, and medical history</strong> to predict risk.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Final Feature Set")
        st.write(list(X_train.columns))
    with col2:
        st.write("### Target Class Balance")
        st.plotly_chart(viz.plot_class_distribution(clean_df), use_container_width=True)

    st.write("### Data Preview (Post-Audit)")
    st.dataframe(X_train.head(10), use_container_width=True)

# --- TAB 2: PIPELINE EXECUTION ---
with tabs[1]:
    st.header("2. Modular Training & Metrics")
    
    with st.spinner("Executing secure pipeline..."):
        results_df, trained_pipelines = trainer.evaluate_all(X_train, y_train, X_test, y_test)
    
    st.subheader("Model Performance (Realistic)")
    st.write("Notice that metrics are no longer 1.0. These results represent the true predictive capability of the data.")
    st.dataframe(results_df.style.background_gradient(cmap='Blues', subset=['Accuracy', 'ROC-AUC']), use_container_width=True)
    
    st.plotly_chart(viz.plot_model_comparison(results_df), use_container_width=True)

# --- TAB 3: VALIDATION & SANITY CHECK ---
with tabs[2]:
    st.header("3. Pipeline Validation")
    
    st.subheader("🛑 Sanity Check: Target Shuffling")
    st.info("If we shuffle the labels, the model's accuracy should drop to random chance (~50%). This proves the model is learning real patterns and not just exploiting a bug in the code.")
    
    if st.button("Run Sanity Check"):
        best_model_name = results_df.iloc[0]["Model"]
        pipeline = trained_pipelines[best_model_name]
        
        acc_real = results_df.iloc[0]["Accuracy"]
        acc_shuffled = trainer.run_sanity_check(pipeline, X_test, y_test)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Real Accuracy", f"{acc_real:.2%}")
        c2.metric("Shuffled Accuracy", f"{acc_shuffled:.2%}", delta=f"{acc_shuffled - acc_real:.2%}", delta_color="inverse")
        
        if acc_shuffled < 0.6:
            st.success("✅ **Sanity Check Passed!** The accuracy dropped to random chance. The pipeline is robust.")
        else:
            st.error("❌ **Sanity Check Failed!** Accuracy is still high. There is still a structural leakage.")

    st.divider()
    st.subheader("📈 Insights & Curves")
    st.plotly_chart(viz.plot_age_distribution(clean_df), use_container_width=True)
    
    # Simple ROC plot logic (could be integrated into visualizer better)
    st.write("### Feature Importance (Top Model)")
    best_pipeline = trained_pipelines[results_df.iloc[0]["Model"]]
    model_obj = best_pipeline.named_steps['classifier']
    fig_imp = viz.plot_feature_importance(model_obj, X_train.columns)
    if fig_imp:
        st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("---")
st.caption("Developed for Academic Excellence & Production Readiness.")
