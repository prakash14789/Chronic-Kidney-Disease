import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_processor import CKDDataProcessor
from model_trainer import CKDModelTrainer
from visualizer import CKDVisualizer

# Page Config
st.set_page_config(page_title="CKD Analytics Pro", page_icon="🧬", layout="wide")

# Custom Styles
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stTabs [aria-selected="true"] { background-color: #0d6efd !important; color: white !important; }
    div[data-testid="stMetricValue"] { color: #0d6efd; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

# Initialize Modules
processor = CKDDataProcessor()
trainer = CKDModelTrainer()
viz = CKDVisualizer()
scaler = StandardScaler()

@st.cache_data
def get_processed_data(sample_size):
    raw_df = processor.load_raw_data()
    sample_df = processor.get_stratified_sample(raw_df, sample_n=sample_size)
    refined_df = processor.refine_data(sample_df)
    return raw_df, refined_df

# Sidebar
with st.sidebar:
    st.title("🧬 CKD Analytics")
    sample_size = st.slider("Select Sample Size", 1000, 10000, 5000)
    st.info("Structure: Data Refining → Model Training → Insight Generation")

# Logic Flow
raw_df, df = get_processed_data(sample_size)
X_train, X_test, y_train, y_test = processor.split_data(df)

# Main Interface
st.title("Chronic Kidney Disease Analysis Pipeline")

tabs = st.tabs(["📁 Data Refining", "⚙️ Model Training", "📈 Insights & Graphs"])

# --- TAB 1: DATA REFINING ---
with tabs[0]:
    st.header("1. Data Refinement & Preprocessing")
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Raw Data Overview")
        st.write(f"Original Dataset: **{len(raw_df):,} rows**")
        st.dataframe(raw_df.head(5), use_container_width=True)
    with col2:
        st.write("### Refined Sample")
        st.write(f"Processed Sample: **{len(df):,} rows**")
        st.dataframe(df.head(5), use_container_width=True)
    
    st.markdown("---")
    st.write("### Dataset Characteristics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Missing Values", df.isnull().sum().sum())
    c2.metric("Feature Count", len(df.columns) - 1)
    c3.metric("Target Balance", f"{df['Diagnosis'].mean():.1%}")

# --- TAB 2: MODEL TRAINING ---
with tabs[1]:
    st.header("2. Model Training & Evaluation")
    
    with st.spinner("Training models across the pipeline..."):
        results_df, roc_data, trained_instances = trainer.train_and_evaluate(X_train, y_train, X_test, y_test, scaler)
    
    st.success("All models trained successfully!")
    st.dataframe(results_df.style.highlight_max(axis=0, subset=["Accuracy", "ROC-AUC"]), use_container_width=True)
    
    st.plotly_chart(viz.plot_model_comparison(results_df), use_container_width=True)

# --- TAB 3: INSIGHTS & GRAPHS ---
with tabs[2]:
    st.header("3. Clinical Insights & Visual Overview")
    
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.plotly_chart(viz.plot_class_distribution(df), use_container_width=True)
    with row1_col2:
        st.plotly_chart(viz.plot_age_distribution(df), use_container_width=True)
    
    st.plotly_chart(viz.plot_correlation_heatmap(df), use_container_width=True)
    
    st.markdown("---")
    st.write("### Model Specific Insights")
    selected_model = st.selectbox("Select Model for Feature Importance", results_df["Model"].tolist())
    
    model_obj = trained_instances[selected_model]
    fig_imp = viz.plot_feature_importance(model_obj, X_train.columns)
    if fig_imp:
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Feature importance is not available for this model type.")
    
    st.plotly_chart(viz.plot_roc_curves(roc_data), use_container_width=True)

st.markdown("---")
st.caption("Structured Pipeline: data_processor.py → model_trainer.py → visualizer.py → app.py")
