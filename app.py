import streamlit as st
import pandas as pd
import numpy as np
import datetime
from data_processor import CKDDataProcessor
from model_trainer import CKDModelTrainer
from visualizer import CKDVisualizer, COLORS
from report_generator import CKDReportGenerator
from stage_predictor import CKDStagePredictor

# Page Config
st.set_page_config(page_title="CKD Clinical Intelligence v3.2", page_icon="🧬", layout="wide")

# Custom Styles with Theme Toggle
theme = st.sidebar.selectbox("🎨 Theme", ["Premium Dark (Glass)", "Clinical Light"], index=0)

if theme == "Clinical Light":
    COLORS["bg"] = "#F8FAFC"
    COLORS["text"] = "#0F172A"
    COLORS["grid"] = "#E2E8F0"
    COLORS["primary"] = "#2563EB"
    COLORS["secondary"] = "#1D4ED8"
    COLORS["accent"] = "#1E40AF"
    
    bg_color = COLORS["bg"]
    text_color = COLORS["text"]
    card_bg = "rgba(255, 255, 255, 0.9)"
    border_color = "rgba(0, 0, 0, 0.1)"
    accent_color = COLORS["primary"]
else:
    bg_color = COLORS["bg"]
    text_color = COLORS["text"]
    card_bg = "rgba(30, 41, 59, 0.7)"
    border_color = "rgba(255, 255, 255, 0.08)"
    accent_color = COLORS["primary"]

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    html, body, [class*="css"], .stApp {{
        font-family: 'Outfit', sans-serif !important;
    }}

    .stApp {{ 
        background: radial-gradient(circle at top left, #1e293b, {bg_color}); 
        color: {text_color}; 
    }}
    
    .stTabs [aria-selected="true"] {{ 
        background: linear-gradient(135deg, {accent_color} 0%, #1D4ED8 100%) !important; 
        color: white !important; 
        border-radius: 8px !important;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px 8px 0 0 !important;
        margin-right: 4px !important;
    }}

    .metric-card {{ 
        background: {card_bg}; 
        backdrop-filter: blur(16px) saturate(180%);
        -webkit-backdrop-filter: blur(16px) saturate(180%);
        color: {text_color} !important; 
        padding: 24px; 
        border-radius: 16px; 
        border-left: 5px solid {accent_color}; 
        margin-bottom: 20px;
        border-top: 1px solid {border_color};
        border-right: 1px solid {border_color};
        border-bottom: 1px solid {border_color};
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
        border-color: rgba(67, 97, 238, 0.3);
    }}

    .metric-card h4, .metric-card p {{ color: {text_color} !important; margin: 0; }}
    
    .prediction-box {{
        background-color: {card_bg}; 
        padding: 24px; 
        border-radius: 16px;
        text-align: center; 
        border: 2px solid {accent_color}; 
        color: {text_color};
        box-shadow: 0 0 15px rgba(67, 97, 238, 0.2);
    }}
    
    .glass-card {{
        background: {card_bg}; 
        backdrop-filter: blur(16px) saturate(180%);
        -webkit-backdrop-filter: blur(16px) saturate(180%);
        border: 1px solid {border_color}; 
        border-radius: 16px;
        padding: 24px; 
        margin-bottom: 16px;
        color: {text_color};
        transition: all 0.3s ease;
    }}
    
    .glass-card:hover {{
        border-color: rgba(67, 97, 238, 0.3);
        box-shadow: 0 8px 12px -1px rgba(0, 0, 0, 0.1);
    }}

    .kpi-row {{ display: flex; gap: 16px; margin-bottom: 24px; }}
    
    .kpi-box {{
        flex: 1; 
        background: {card_bg};
        backdrop-filter: blur(16px) saturate(180%);
        -webkit-backdrop-filter: blur(16px) saturate(180%);
        border-radius: 16px; 
        padding: 24px; 
        text-align: center;
        border: 1px solid {border_color};
        color: {text_color};
        transition: all 0.3s ease;
    }}
    
    .kpi-box:hover {{
        transform: translateY(-3px);
        border-color: rgba(67, 97, 238, 0.3);
    }}

    .kpi-box h2 {{ color: {accent_color}; margin: 0; font-size: 2.2rem; font-weight: 700; }}
    .kpi-box p {{ color: {text_color}; margin: 6px 0 0 0; font-size: 0.9rem; opacity: 0.8; }}
    
    .stButton>button {{
        background: linear-gradient(135deg, {accent_color} 0%, #1D4ED8 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }}
    
    .stButton>button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(67, 97, 238, 0.3) !important;
    }}
    
    [data-testid="stSidebar"] {{
        background-color: #0f172a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- LOGIN SYSTEM ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['role'] = None

def login_page():
    # Center the login form
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("""
            <div style='background: rgba(30, 41, 59, 0.5); backdrop-filter: blur(16px); padding: 30px; border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.1); margin-top: 50px; margin-bottom: 20px; text-align: center;'>
                <span style='font-size: 3rem;'>🧬</span>
                <h2 style='color: #f8fafc; margin-top: 10px; font-weight: 700; letter-spacing: -0.05em;'>Clinical Portal</h2>
                <p style='color: #94a3b8; font-size: 0.9rem;'>Chronic Kidney Disease Intelligence</p>
            </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                if username == "admin" and password == "admin":
                    st.session_state['logged_in'] = True
                    st.session_state['role'] = "admin"
                    st.success("Admin Access Granted!")
                    st.rerun()
                elif username == "client" and password == "client":
                    st.session_state['logged_in'] = True
                    st.session_state['role'] = "client"
                    st.success("Client Access Granted!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

if not st.session_state['logged_in']:
    login_page()
    st.stop()

# Initialize
processor = CKDDataProcessor()
trainer = CKDModelTrainer()
viz = CKDVisualizer()
reporter = CKDReportGenerator()
stage_predictor = CKDStagePredictor()

@st.cache_resource
def get_cached_shap(_model, X_test):
    return trainer.get_shap_explainer(_model, X_test)

# Sidebar
with st.sidebar:
    st.title("🧬 CKD Intelligence")
    st.image("https://cdn-icons-png.flaticon.com/512/3067/3067451.png", width=80)
    sample_size = st.slider("Stratified Sample Size", 1000, 10000, 5000)
    use_cv = st.checkbox("Enable Cross-Validation (Slower)", value=False)
    st.divider()
    st.markdown("[🔗 View Source on GitHub](https://github.com/prakash14789/Chronic-Kidney-Disease)")
    st.success("v3.3 Research Pipeline Active")
    st.divider()
    # Model saving for API
    if st.session_state.get('role') == 'admin':
        if st.button("💾 Save Model for API", use_container_width=True):
            try:
                path = trainer.save_model_for_api(
                    trained_nl[best_name] if 'trained_nl' in dir() else None,
                    X_te_nl.columns if 'X_te_nl' in dir() else [],
                    best_name if 'best_name' in dir() else 'unknown',
                    best_th if 'best_th' in dir() else 0.5
                )
                st.success(f"Model saved to {path}")
            except Exception as e:
                st.error(f"Save failed: {e}")
    st.divider()
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state['logged_in'] = False
        st.session_state['role'] = None
        st.rerun()

# --- PIPELINE EXECUTION ---
@st.cache_data
def run_full_pipeline(sample_n, use_cv=False):
    df_full = processor.load_raw_data()
    df_sample = processor.get_v3_refined_data(df_full, sample_n=sample_n)
    X_tr_f, X_te_f, y_tr_f, y_te_f = processor.split_and_encode_v3(df_sample)
    
    leakage_cols = ["GFR", "SerumCreatinine", "BUNLevels", "ProteinInUrine", "ACR"]
    X_tr_nl = X_tr_f.drop(columns=leakage_cols, errors="ignore")
    X_te_nl = X_te_f.drop(columns=leakage_cols, errors="ignore")
    X_te_nl = X_te_nl[X_tr_nl.columns]
    
    n_neg, n_pos = (y_tr_f == 0).sum(), (y_tr_f == 1).sum()
    
    # 1. Experiment 1: Full Features (With SMOTE)
    pipes_f = trainer.get_v3_pipelines(n_neg, n_pos, use_smote=True)
    res_f, roc_f, pr_f, trained_f = trainer.run_v3_experiment(X_tr_f, X_te_f, y_tr_f, y_te_f, pipelines=pipes_f, use_cv=use_cv)
    
    # 2. Experiment 2: No-Leakage (With SMOTE)
    pipes_nl = trainer.get_v3_pipelines(n_neg, n_pos, use_smote=True)
    res_nl, roc_nl, pr_nl, trained_nl = trainer.run_v3_experiment(X_tr_nl, X_te_nl, y_tr_f, y_te_f, pipelines=pipes_nl, use_cv=use_cv)
    
    # 3. Experiment 3: No-Leakage (WITHOUT SMOTE) for comparison
    pipes_no_smote = trainer.get_v3_pipelines(n_neg, n_pos, use_smote=False)
    res_no_smote, _, _, _ = trainer.run_v3_experiment(X_tr_nl, X_te_nl, y_tr_f, y_te_f, pipelines=pipes_no_smote, use_cv=use_cv)
    
    best_name = res_nl.iloc[0]["Model"]
    y_proba = trained_nl[best_name].predict_proba(X_te_nl)[:, 1]
    _, best_th = trainer.tune_threshold(y_te_f, y_proba)
    
    return (df_full, df_sample, res_f, roc_f, res_nl, roc_nl, pr_nl,
            trained_nl, X_tr_nl, X_te_nl, y_tr_f, y_te_f, best_th, res_no_smote)

(df_full, df_sample, res_f, roc_f, res_nl, roc_nl, pr_nl,
 trained_nl, X_tr_nl, X_te_nl, y_tr_f, y_te_f, best_th, res_no_smote) = run_full_pipeline(sample_size, use_cv=use_cv)

# Derived values
best_name = res_nl.iloc[0]["Model"]
y_proba_all = trained_nl[best_name].predict_proba(X_te_nl)[:, 1]
ckd_pct = df_full['Diagnosis'].mean() * 100

# --- KPI CARDS ---
st.markdown("<h1 style='text-align: center; margin-bottom: 30px; font-weight: 800; letter-spacing: -0.05em;'>🧬 CKD Clinical Intelligence</h1>", unsafe_allow_html=True)

st.markdown(f"""
<div class='kpi-row'>
    <div class='kpi-box'>
        <p>🏆 Best Model</p>
        <h2>{best_name}</h2>
    </div>
    <div class='kpi-box'>
        <p>🎯 Balanced Accuracy</p>
        <h2>{res_nl.iloc[0]['Balanced Accuracy']:.2%}</h2>
    </div>
    <div class='kpi-box'>
        <p>📈 ROC-AUC</p>
        <h2>{res_nl.iloc[0]['ROC-AUC']:.4f}</h2>
    </div>
    <div class='kpi-box'>
        <p>📊 Dataset Size</p>
        <h2>{len(df_full):,}</h2>
        <p style='font-size: 0.8rem; opacity: 0.6;'>Sampled to {sample_size}</p>
    </div>
</div>
""", unsafe_allow_html=True)

if st.session_state['role'] == "admin":
    tabs = st.tabs([
        "📊 Data Audit", "🚀 Exp 1 (Full)", "🛡️ Exp 2 (No-Leakage)", "📉 Comparison",
        "🧬 SMOTE Insights", "🎯 Threshold Tuning", "🧠 SHAP Interpretation", "🔬 Deep Analysis",
        "🏥 Patient Diagnosis", "📂 Batch Diagnosis",
        "🏗️ CKD Staging", "⚡ Optuna Tuning", "📈 Risk Timeline"
    ])
    (t_audit, t_exp1, t_exp2, t_comp, t_smote, t_th, t_shap, t_deep,
     t_diag, t_batch, t_stage, t_optuna, t_timeline) = tabs
else:
    tabs = st.tabs(["🏥 Patient Diagnosis", "📂 Batch Diagnosis"])
    t_diag, t_batch = tabs
    t_audit = t_exp1 = t_exp2 = t_comp = t_smote = t_th = t_shap = t_deep = None
    t_stage = t_optuna = t_timeline = None

# --- TABS RENDERING ---
import app_tabs

if t_audit:
    with t_audit:
        app_tabs.render_data_audit(df_full, df_sample, viz, ckd_pct)

if t_exp1:
    with t_exp1:
        app_tabs.render_exp1(res_f, roc_f, viz)

if t_exp2:
    with t_exp2:
        app_tabs.render_exp2(res_nl, roc_nl, trained_nl, best_name, X_te_nl, y_te_f, trainer, viz)

if t_comp:
    with t_comp:
        app_tabs.render_comparison(res_f, res_nl, ckd_pct, df_sample, viz)

if t_smote:
    with t_smote:
        app_tabs.render_smote_insights(res_nl, res_no_smote)

if t_th:
    with t_th:
        app_tabs.render_threshold_tuning(y_te_f, y_proba_all, best_th, trainer, trained_nl, best_name, X_te_nl, viz)

if t_shap:
    with t_shap:
        app_tabs.render_shap_interpretation(trained_nl, best_name, X_te_nl, get_cached_shap, trainer, viz)

if t_deep:
    with t_deep:
        app_tabs.render_deep_analysis(df_sample, y_te_f, y_proba_all, trained_nl, best_name, X_te_nl, y_tr_f, trainer, viz)

if t_diag:
    with t_diag:
        app_tabs.render_patient_diagnosis(X_te_nl, trained_nl, best_name, best_th, trainer, viz, reporter, get_cached_shap, y_proba_all, X_tr_nl, y_tr_f)

if t_batch:
    with t_batch:
        app_tabs.render_batch_diagnosis(X_te_nl, trained_nl, best_name, best_th, trainer)

if t_stage:
    with t_stage:
        @st.cache_data
        def run_stage_pipeline(_df_full, sample_n):
            sp = CKDStagePredictor()
            results = sp.train(_df_full, sample_n=sample_n)
            return sp, results

        with st.spinner("Training multi-class stage predictor..."):
            stage_model, stage_results = run_stage_pipeline(df_full, sample_size)

        app_tabs.render_stage_prediction(stage_results, viz)

        # Individual stage prediction
        st.markdown("---")
        st.subheader("🏥 Predict Stage for a Patient")
        if st.button("🔬 Predict CKD Stage (Using Last Diagnosis Input)", key="stage_predict_btn"):
            try:
                input_row_stage = X_te_nl.mean().to_frame().T.copy()
                for feat in stage_model.feature_names:
                    if feat not in input_row_stage.columns:
                        input_row_stage[feat] = 0
                input_row_stage = input_row_stage[stage_model.feature_names]

                result = stage_model.predict_stage(input_row_stage)
                info = result['stage_info']

                st.plotly_chart(viz.plot_stage_prediction_bar(
                    result['stage_probabilities'], result['predicted_stage']
                ), use_container_width=True, key="stage_pred_bar")

                st.markdown(f"""
                <div class='metric-card' style='border-left: 6px solid {info["color"]}; text-align: center;'>
                    <h3>{info["icon"]} Predicted: {info["name"]}</h3>
                    <p style='font-size: 1.1rem;'>{info["description"]}</p>
                    <p style='font-size: 0.95rem; color: #94a3b8;'>GFR Range: {info["gfr_range"]}</p>
                    <p style='margin-top: 12px;'><strong>Action:</strong> {info["action"]}</p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Stage prediction error: {e}")

if t_optuna:
    with t_optuna:
        app_tabs.render_optuna_tuning(X_tr_nl, y_tr_f, X_te_nl, y_te_f, trainer, viz, res_nl)

if t_timeline:
    with t_timeline:
# Removed redundant auto-save to improve performance. Use the sidebar button instead.

st.markdown("---")
st.caption("CKD Intelligence v3.3 — Precision Research Dashboard with Stage Prediction, Optuna Tuning & Risk Timeline.")

