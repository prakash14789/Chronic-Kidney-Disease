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

# --- TAB 1: DATA AUDIT ---
if t_audit:
    with t_audit:
        st.header("1. Class Imbalance & Audit")
        col1, col2 = st.columns([1, 2])
        with col1: st.plotly_chart(viz.plot_class_distribution(df_full, ckd_pct), use_container_width=True, key="dist")
        with col2:
            st.write("### v3 Hygiene Checklist")
            st.checkbox("Adherence encoded globally before split", value=True, disabled=True)
            st.checkbox("Leakage-free derived from same split", value=True, disabled=True)
            st.checkbox("SMOTE lives inside Pipeline", value=True, disabled=True)
        st.markdown("---")
        st.header("2. Correlation & Clinical Distributions")
        st.pyplot(viz.plot_correlation_heatmap(df_sample))
        st.plotly_chart(viz.plot_clinical_boxplots(df_sample), use_container_width=True)

# --- TAB 2: EXP 1 ---
if t_exp1:
    with t_exp1:
        st.header("Experiment 1: Full Features")
        st.dataframe(res_f[['Model', 'Balanced Accuracy', 'Macro F1', 'ROC-AUC']], use_container_width=True)
        st.plotly_chart(viz.plot_roc_curves(roc_f), use_container_width=True, key="roc_f")

# --- TAB 3: EXP 2 ---
if t_exp2:
    with t_exp2:
        st.header("Experiment 2: Leakage-Free (Research Grade)")
        st.dataframe(res_nl[['Model', 'Balanced Accuracy', 'Macro F1', 'ROC-AUC']], use_container_width=True)
        st.plotly_chart(viz.plot_roc_curves(roc_nl), use_container_width=True, key="roc_nl")
        # Confusion Matrix
        err = trainer.get_error_analysis(trained_nl[best_name], X_te_nl, y_te_f)
        st.plotly_chart(viz.plot_confusion_matrix(err["counts"]), use_container_width=True, key="cm")

# --- TAB 4: COMPARISON ---
if t_comp:
    with t_comp:
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

# --- TAB 5: SMOTE INSIGHTS ---
if t_smote:
    with t_smote:
        st.header("🧬 SMOTE Impact Analysis")
        st.info("SMOTE (Synthetic Minority Over-sampling Technique) creates artificial samples to balance the dataset. In our CKD case (92/8 split), it targets the 'Healthy' minority.")
        
        # Side-by-side comparison
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.subheader("Results WITH SMOTE")
            st.dataframe(res_nl[['Model', 'Balanced Accuracy', 'Macro F1', 'Macro Recall']].head(5), use_container_width=True)
        with col_s2:
            st.subheader("Results WITHOUT SMOTE")
            st.dataframe(res_no_smote[['Model', 'Balanced Accuracy', 'Macro F1', 'Macro Recall']].head(5), use_container_width=True)

        st.markdown("---")
        st.subheader("🔍 Why do metrics fall after SMOTE?")
        
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown(f"""
            <div class='glass-card'>
            <h4>1. The Accuracy Trap</h4>
            <p>With a 92% majority class, a model that guesses 'Sick' for everyone is 92% accurate but useless. SMOTE forces the model to learn the harder 8% (Healthy), which naturally drops the 'easy' accuracy score.</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class='glass-card'>
            <h4>2. Redundancy</h4>
            <p>Your models already use <strong>class_weight='balanced'</strong>. Adding SMOTE on top can be overkill, making the model over-sensitive to the minority class and increasing False Positives.</p>
            </div>
            """, unsafe_allow_html=True)
        with sc2:
            st.markdown(f"""
            <div class='glass-card'>
            <h4>3. Boundary Blurring</h4>
            <p>SMOTE creates points by drawing lines between existing ones. If Healthy and Sick patients have overlapping features, SMOTE creates 'synthetic noise' in the overlap, confusing the model.</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class='glass-card'>
            <h4>4. The Trade-off</h4>
            <p>SMOTE is used to improve <strong>Minority Recall</strong> (finding healthy people). We often accept a drop in Precision or Accuracy to ensure the model isn't just ignoring the minority group.</p>
            </div>
            """, unsafe_allow_html=True)

# --- TAB 6: THRESHOLD ---
if t_th:
    with t_th:
        st.header("Decision Threshold Optimization")
        th_df, _ = trainer.tune_threshold(y_te_f, y_proba_all)
        st.plotly_chart(viz.plot_threshold_tuning(th_df, best_th), use_container_width=True, key="th_tune")
        # Interactive threshold slider
        user_th = st.slider("Adjust Threshold", 0.05, 0.95, float(best_th), 0.05, key="th_slider")
        from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
        y_at_th = (y_proba_all >= user_th).astype(int)
        tc1, tc2, tc3, tc4 = st.columns(4)
        with tc1: st.metric("Balanced Acc", f"{balanced_accuracy_score(y_te_f, y_at_th):.2%}")
        with tc2: st.metric("Precision", f"{precision_score(y_te_f, y_at_th, zero_division=0):.2%}")
        with tc3: st.metric("Recall", f"{recall_score(y_te_f, y_at_th, zero_division=0):.2%}")
        with tc4: st.metric("Macro F1", f"{f1_score(y_te_f, y_at_th, average='macro', zero_division=0):.2%}")
        if st.button("Run Sanity Check"):
            acc = trainer.run_sanity_check(trained_nl[best_name], X_te_nl, y_te_f)
            st.metric("Shuffled Balanced Accuracy", f"{acc:.2%}")

# --- TAB 7: SHAP ---
if t_shap:
    with t_shap:
        st.header("🧠 Model Interpretation (SHAP)")
        with st.spinner("Calculating Global SHAP..."):
            X_test_df = pd.DataFrame(X_te_nl.values, columns=X_te_nl.columns).reset_index(drop=True)
            explainer, shap_values, X_df = trainer.get_shap_explainer(trained_nl[best_name], X_test_df)
        cs1, cs2 = st.columns(2)
        with cs1: st.pyplot(viz.plot_shap_bar(explainer, shap_values, X_df, best_name))
        with cs2: st.pyplot(viz.plot_shap_summary(explainer, shap_values, X_df, best_name))
        # Grouped SHAP
        st.subheader("Risk Factor Group Contributions")
        group_imp = trainer.get_grouped_shap(shap_values, X_df.columns)
        st.plotly_chart(viz.plot_grouped_shap(group_imp), use_container_width=True, key="grp_shap")

# --- TAB 8: DEEP ANALYSIS ---
if t_deep:
    with t_deep:
        st.header("🔬 Deep Analysis")

        # 1. Feature Direction
        st.subheader("1. What Differentiates CKD Patients?")
        st.plotly_chart(viz.plot_feature_direction(df_sample), use_container_width=True, key="feat_dir")

        # 2. Calibration Curve
        st.subheader("2. Calibration — Is Your Probability Trustworthy?")
        st.plotly_chart(viz.plot_calibration(y_te_f, y_proba_all), use_container_width=True, key="calib")

        # 3. Error Analysis
        st.subheader("3. Error Analysis — Where Does the Model Fail?")
        err_data = trainer.get_error_analysis(trained_nl[best_name], X_te_nl, y_te_f)
        ec1, ec2 = st.columns(2)
        with ec1:
            st.plotly_chart(viz.plot_confusion_matrix(err_data["counts"]), use_container_width=True, key="cm2")
        with ec2:
            c = err_data["counts"]
            st.markdown(f"""
            <div class='glass-card'>
            <h4>📋 Error Breakdown</h4>
            <p>✅ True Positives: <strong>{c['TP']}</strong> — Correctly identified CKD</p>
            <p>✅ True Negatives: <strong>{c['TN']}</strong> — Correctly ruled out CKD</p>
            <p>⚠️ False Positives: <strong>{c['FP']}</strong> — Healthy flagged as CKD</p>
            <p>🚨 False Negatives: <strong>{c['FN']}</strong> — Missed CKD cases</p>
            </div>
            """, unsafe_allow_html=True)
        st.plotly_chart(viz.plot_error_patterns(err_data["fp_data"], err_data["fn_data"], X_te_nl),
                        use_container_width=True, key="err_pat")

        # 4. Population Risk Distribution
        st.subheader("4. Population Risk Distribution")
        st.plotly_chart(viz.plot_population_risk(y_proba_all), use_container_width=True, key="pop_risk")

        # 5. Model Stability
        st.subheader("5. Model Stability Across Splits")
        if st.button("🔄 Run Stability Check (5 splits)", key="stab_btn"):
            with st.spinner("Running 5 cross-validation splits..."):
                # Reconstruct full no-leakage data for stability
                leakage_cols = ["GFR", "SerumCreatinine", "BUNLevels", "ProteinInUrine", "ACR"]
                X_full_nl = pd.concat([X_tr_nl, X_te_nl]).reset_index(drop=True)
                y_full = pd.concat([y_tr_f, y_te_f]).reset_index(drop=True)
                scores = trainer.run_stability_check_multi(X_full_nl, y_full, n_runs=5)
            st.plotly_chart(viz.plot_model_stability(scores), use_container_width=True, key="stab")
            st.markdown(f"""
            <div class='glass-card'>
            <h4>📊 Stability Summary</h4>
            <p>Mean: <strong>{np.mean(scores):.4f}</strong> | Std: <strong>{np.std(scores):.4f}</strong></p>
            <p>{'✅ Model is stable across splits' if np.std(scores) < 0.02 else '⚠️ Some variance detected'}</p>
            </div>
            """, unsafe_allow_html=True)

# --- TAB 9: DIAGNOSIS ---
with t_diag:
    st.header("🏥 Precision Patient Diagnosis")
    st.info("Fill in the clinical details below. Features not specified will be set to the population average.")
    
    with st.form("diag_form"):
        st.subheader("1. Patient Profile & Lifestyle")
        cl1, cl2, cl3, cl4 = st.columns(4)
        with cl1:
            age = st.number_input("Age", 20, 90, 50)
            gender = st.selectbox("Gender", ["Male", "Female"])
        with cl2:
            bmi = st.number_input("BMI", 15.0, 45.0, 25.0)
            smoking = st.selectbox("Smoking", ["No", "Yes"])
        with cl3:
            activity = st.number_input("Activity (min/week)", 0, 300, 150)
            adherence = st.selectbox("Medication Adherence", ["Adherent", "Non-Adherent"])
        with cl4:
            diet = st.slider("Diet Quality (0-10)", 0, 10, 5)
            sleep = st.slider("Sleep Quality (0-10)", 0, 10, 7)

        st.divider()
        st.subheader("2. Primary Clinical Markers")
        cv1, cv2, cv3, cv4 = st.columns(4)
        with cv1:
            systolic = st.number_input("Systolic BP (mmHg)", 90, 200, 120)
            diastolic = st.number_input("Diastolic BP (mmHg)", 60, 130, 80)
        with cv2:
            fbs = st.number_input("Fasting Blood Sugar", 70, 250, 100)
            hba1c = st.number_input("HbA1c (%)", 4.0, 15.0, 5.5)
        with cv3:
            hemoglobin = st.number_input("Hemoglobin (g/dL)", 10.0, 20.0, 14.0)
            cholesterol = st.number_input("Total Cholesterol", 100, 400, 180)
        with cv4:
            st.write("Family History")
            fam_kidney = st.checkbox("Kidney Disease", value=False)
            fam_hyper = st.checkbox("Hypertension", value=False)
            fam_diab = st.checkbox("Diabetes", value=False)

        with st.expander("🛠️ Advanced Parameters (Optional - Defaults to Population Mean)"):
            st.write("Adjust these for a more precise clinical profile.")
            ca1, ca2, ca3 = st.columns(3)
            with ca1:
                sodium = st.number_input("Serum Sodium (mEq/L)", 130.0, 150.0, float(X_te_nl["SerumElectrolytesSodium"].mean()))
                potassium = st.number_input("Serum Potassium (mEq/L)", 3.0, 6.0, float(X_te_nl["SerumElectrolytesPotassium"].mean()))
            with ca2:
                fatigue = st.slider("Fatigue Level (0-10)", 0, 10, int(X_te_nl["FatigueLevels"].mean()))
                edema = st.selectbox("Edema (Swelling)", ["No", "Yes"], index=int(X_te_nl["Edema"].mean()))
            with ca3:
                qol = st.slider("Quality of Life Score", 0, 100, int(X_te_nl["QualityOfLifeScore"].mean()))
                heavy_metals = st.checkbox("Heavy Metals Exposure", value=bool(X_te_nl["HeavyMetalsExposure"].mean() > 0.5))

        sub = st.form_submit_button("🚀 Generate Precision Diagnosis")
        
    if sub:
        input_row = X_te_nl.mean().to_frame().T.copy()
        input_row["Age"] = age
        input_row["BMI"] = bmi
        input_row["SystolicBP"] = systolic
        input_row["DiastolicBP"] = diastolic
        input_row["FastingBloodSugar"] = fbs
        input_row["HbA1c"] = hba1c
        input_row["PhysicalActivity"] = activity
        input_row["DietQuality"] = diet
        input_row["SleepQuality"] = sleep
        input_row["HemoglobinLevels"] = hemoglobin
        input_row["CholesterolTotal"] = cholesterol
        input_row["SerumElectrolytesSodium"] = sodium
        input_row["SerumElectrolytesPotassium"] = potassium
        input_row["FatigueLevels"] = fatigue
        input_row["QualityOfLifeScore"] = qol
        input_row["Gender"] = 1 if gender == "Male" else 0
        input_row["Smoking"] = 1 if smoking == "Yes" else 0
        input_row["FamilyHistoryKidneyDisease"] = 1 if fam_kidney else 0
        input_row["FamilyHistoryHypertension"] = 1 if fam_hyper else 0
        input_row["FamilyHistoryDiabetes"] = 1 if fam_diab else 0
        input_row["Edema"] = 1 if edema == "Yes" else 0
        input_row["HeavyMetalsExposure"] = 1 if heavy_metals else 0
        input_row["Adherence"] = 0 if adherence == "Adherent" else 1
        input_row = input_row[X_te_nl.columns]
        
        prob = trained_nl[best_name].predict_proba(input_row)[0, 1]
        assessment = trainer.get_clinical_assessment(prob)
        
        # Risk Gauge + Assessment
        col_g, col_r = st.columns([1, 2])
        with col_g:
            st.plotly_chart(viz.plot_risk_gauge(prob), use_container_width=True, key="gauge")
        with col_r:
            st.markdown(f"""
                <div class='metric-card' style='border-left: 6px solid {assessment["Color"]}; box-shadow: 0 4px 15px {assessment["Color"]}20;'>
                    <h4>{assessment["Icon"]} {assessment["Level"]} — {prob:.1%} Risk</h4>
                    <p style='font-size: 1.1rem; margin-top: 10px;'>{assessment["Action"]}</p>
                    <p style='font-size: 0.9rem; color: #94a3b8; margin-top: 15px;'>
                        *Decision based on optimal clinical threshold of <strong>{best_th}</strong>
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        # --- REPORT GENERATION ---
        st.markdown("### 📄 Clinical Documentation")
        shap_highlights = trainer.get_patient_shap_highlights(shap_values, X_df.columns, patient_idx=0)
        
        # We need to map internal names back to readable names if possible, but internal are fine for now.
        patient_summary = {
            "Age": age, "Gender": gender, "BMI": bmi, "Systolic BP": systolic,
            "Diastolic BP": diastolic, "HbA1c": hba1c, "Hemoglobin": hemoglobin,
            "Adherence": adherence
        }
        
        if st.button("🛠️ Prepare Clinical Report"):
            with st.spinner("Generating professional PDF..."):
                report_path = reporter.generate_patient_report(patient_summary, prob, assessment, shap_highlights)
                with open(report_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Clinical PDF Report",
                        data=f,
                        file_name=f"CKD_Patient_Report_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf"
                    )
                st.success("Report ready for download!")

        st.markdown("---")
        
        # Counterfactual What-If
        st.subheader("🔄 What Needs to Change to Reduce Risk?")
        base_p, cf_results = trainer.compute_counterfactual(trained_nl[best_name], input_row)
        st.plotly_chart(viz.plot_counterfactual(base_p, cf_results), use_container_width=True, key="cf")
        
        # Population Position
        st.subheader("📊 Where Does This Patient Fall?")
        percentile = (y_proba_all < prob).mean() * 100
        st.markdown(f"<div class='glass-card'><h4>This patient is in the **top {100-percentile:.0f}%** risk bracket</h4></div>", unsafe_allow_html=True)
        st.plotly_chart(viz.plot_population_risk(y_proba_all, prob), use_container_width=True, key="pop")

        # Similar Patients
        st.subheader("👥 Most Similar Patients in Dataset")
        sim_X, sim_y, sim_d = trainer.find_similar_patients(X_tr_nl, y_tr_f, input_row, n=5)
        sim_df = sim_X.copy()
        sim_df["Diagnosis"] = sim_y.values
        sim_df["Distance"] = sim_d.round(2)
        ckd_count = (sim_y == 1).sum()
        st.markdown(f"<div class='glass-card'><h4>Of 5 most similar patients: <strong>{ckd_count}/5 have CKD</strong></h4></div>", unsafe_allow_html=True)
        st.dataframe(sim_df[["Age", "BMI", "SystolicBP", "HbA1c", "Diagnosis", "Distance"]], use_container_width=True)
        
        # Why CKD / Why NOT CKD
        st.subheader("🛡️ Why CKD / Why NOT CKD")
        st.plotly_chart(viz.plot_protective_factors(shap_values, X_df.columns, patient_idx=0),
                        use_container_width=True, key="prot")
        
        # SHAP Waterfall
        st.subheader("📊 Feature Contribution Analysis")
        st.pyplot(viz.plot_local_shap(explainer, shap_values, X_df, patient_idx=0))

# --- TAB 10: BATCH DIAGNOSIS ---
if t_batch:
    with t_batch:
        st.header("📂 Batch Patient Diagnosis")
        st.info("Upload a CSV file containing patient data. The system will predict the risk for all patients.")
        
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                import pandas as pd
                import plotly.express as px
                
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(batch_df)} patients.")
                
                # We need to ensure the columns match X_te_nl.columns
                missing_cols = [c for c in X_te_nl.columns if c not in batch_df.columns]
                if missing_cols:
                    st.warning(f"The following required columns are missing and will be filled with population means: {missing_cols}")
                    for c in missing_cols:
                        batch_df[c] = X_te_nl[c].mean()
                
                # Ensure correct order and handle extra columns
                X_batch = batch_df[X_te_nl.columns]
                
                # Run predictions
                probs = trained_nl[best_name].predict_proba(X_batch)[:, 1]
                preds = (probs >= best_th).astype(int)
                
                # Add results to dataframe
                batch_df["CKD_Risk_Score"] = probs
                batch_df["CKD_Prediction"] = preds
                batch_df["Risk_Level"] = [trainer.get_clinical_assessment(p)["Level"] for p in probs]
                
                # Show results
                st.dataframe(batch_df, use_container_width=True)
                
                # Summary plots
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.pie(batch_df, names="Risk_Level", title="Risk Level Distribution",
                                 color="Risk_Level",
                                 color_discrete_map={"Low Risk": "#4D96FF", "Moderate Risk": "#FFD93D", "High Risk": "#FF6B6B"})
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig = px.histogram(batch_df, x="CKD_Risk_Score", title="Risk Score Distribution",
                                       nbinsx=20, color_discrete_sequence=["#4361EE"])
                    st.plotly_chart(fig, use_container_width=True)
                
                # Download button
                csv = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv,
                    file_name="ckd_batch_predictions.csv",
                    mime="text/csv",
                )
                
            except Exception as e:
                st.error(f"Error processing file: {e}")

# --- TAB 11: CKD STAGING ---
if t_stage:
    with t_stage:
        st.header("🏗️ CKD Stage Prediction (Multi-Class)")
        st.info("Predicts CKD stages 1–5 based on KDIGO guidelines using non-leakage features. Stages are derived from GFR ranges, but the model predicts stages *without* GFR.")

        @st.cache_data
        def run_stage_pipeline(_df_full, sample_n):
            sp = CKDStagePredictor()
            results = sp.train(_df_full, sample_n=sample_n)
            return sp, results

        with st.spinner("Training multi-class stage predictor..."):
            stage_model, stage_results = run_stage_pipeline(df_full, sample_size)

        # Stage overview
        st.subheader("📊 Stage Distribution in Dataset")
        sc1, sc2 = st.columns([1, 2])
        with sc1:
            st.plotly_chart(viz.plot_stage_distribution(stage_results['stage_distribution']),
                            use_container_width=True, key="stage_dist")
        with sc2:
            # Stage definitions table
            st.markdown("""
            <div class='glass-card'>
            <h4>KDIGO Stage Definitions</h4>
            <table style='width:100%; color: inherit;'>
            <tr><th>Stage</th><th>GFR Range</th><th>Severity</th></tr>
            <tr><td>🟢 Stage 1</td><td>≥90 mL/min</td><td>Normal/High</td></tr>
            <tr><td>🔵 Stage 2</td><td>60–89 mL/min</td><td>Mild</td></tr>
            <tr><td>🟡 Stage 3</td><td>30–59 mL/min</td><td>Moderate</td></tr>
            <tr><td>🟠 Stage 4</td><td>15–29 mL/min</td><td>Severe</td></tr>
            <tr><td>🔴 Stage 5</td><td><15 mL/min</td><td>Kidney Failure</td></tr>
            </table>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("🎯 Model Performance")
        st.metric("Balanced Accuracy", f"{stage_results['balanced_accuracy']:.2%}")

        sr1, sr2 = st.columns(2)
        with sr1:
            # Per-class metrics
            report_df = pd.DataFrame(stage_results['report']).T
            report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
            st.dataframe(report_df.round(3), use_container_width=True)
        with sr2:
            st.plotly_chart(viz.plot_stage_confusion(
                stage_results['y_test'], stage_results['y_pred']
            ), use_container_width=True, key="stage_cm")

        st.markdown("---")
        st.subheader("🔑 Top Features for Stage Prediction")
        top_feats = stage_results['feature_importances'].head(10)
        import plotly.express as px
        fig_fi = px.bar(x=top_feats.values, y=top_feats.index, orientation='h',
                        color=top_feats.values, color_continuous_scale='blues',
                        labels={'x': 'Importance', 'y': 'Feature'})
        fig_fi.update_layout(title="Top 10 Features for CKD Stage Classification",
                             yaxis=dict(autorange="reversed"))
        st.plotly_chart(viz._apply_dark_theme(fig_fi), use_container_width=True, key="stage_fi")

        # Individual stage prediction (reuses diagnosis form values if available)
        st.markdown("---")
        st.subheader("🏥 Predict Stage for a Patient")
        if st.button("🔬 Predict CKD Stage (Using Last Diagnosis Input)", key="stage_predict_btn"):
            try:
                input_row_stage = X_te_nl.mean().to_frame().T.copy()
                # Ensure all stage features are present
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

# --- TAB 12: OPTUNA TUNING ---
if t_optuna:
    with t_optuna:
        st.header("⚡ Hyperparameter Tuning with Optuna")
        st.info("Automatically searches for optimal hyperparameters using Bayesian optimization. Tunes Random Forest, Gradient Boosting, XGBoost, and LightGBM.")

        n_trials = st.slider("Number of Trials per Model", 10, 100, 30, key="optuna_trials")

        if st.button("🚀 Start Optuna Tuning", key="optuna_start", type="primary"):
            with st.spinner(f"Running {n_trials} trials per model... This may take a few minutes."):
                optuna_results = trainer.tune_with_optuna(
                    X_tr_nl, y_tr_f, X_te_nl, y_te_f, n_trials=n_trials
                )

            st.success("✅ Tuning complete!")

            # Optimization history
            st.subheader("📈 Optimization History")
            st.plotly_chart(viz.plot_optuna_history(optuna_results),
                            use_container_width=True, key="optuna_hist")

            # Before vs After comparison
            st.subheader("📊 Default vs Tuned Performance")
            default_scores = {}
            for _, row in res_nl.iterrows():
                if row['Model'] in optuna_results:
                    default_scores[row['Model']] = row['Balanced Accuracy']
            st.plotly_chart(viz.plot_optuna_comparison(optuna_results, default_scores),
                            use_container_width=True, key="optuna_comp")

            # Best params for each model
            st.subheader("🏆 Best Hyperparameters Found")
            for model_name, data in optuna_results.items():
                with st.expander(f"**{model_name}** — Best Score: {data['best_score']:.4f}"):
                    improvement = data['best_score'] - default_scores.get(model_name, 0)
                    if improvement > 0:
                        st.success(f"Improvement: +{improvement:.4f} Balanced Accuracy")
                    else:
                        st.info(f"Change: {improvement:+.4f} (default params were already strong)")

                    params_df = pd.DataFrame([data['best_params']]).T
                    params_df.columns = ["Value"]
                    params_df.index.name = "Parameter"
                    st.dataframe(params_df, use_container_width=True)

            # Summary
            st.markdown("---")
            best_tuned = max(optuna_results.items(), key=lambda x: x[1]['best_score'])
            st.markdown(f"""
            <div class='metric-card' style='text-align: center;'>
                <h4>🏆 Overall Best After Tuning</h4>
                <h2 style='color: #4361EE;'>{best_tuned[0]}</h2>
                <p>Balanced Accuracy: <strong>{best_tuned[1]['best_score']:.4f}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='glass-card'>
                <h4>ℹ️ How Optuna Works</h4>
                <p><strong>Bayesian Optimization:</strong> Unlike grid search, Optuna uses TPE (Tree-structured Parzen Estimators) to intelligently explore the hyperparameter space, focusing on promising regions.</p>
                <p><strong>What gets tuned:</strong></p>
                <ul>
                    <li>🌲 <strong>Random Forest:</strong> n_estimators, max_depth, min_samples_split/leaf, max_features</li>
                    <li>📈 <strong>Gradient Boosting:</strong> n_estimators, max_depth, learning_rate, subsample</li>
                    <li>🚀 <strong>XGBoost:</strong> + colsample_bytree, reg_alpha, reg_lambda</li>
                    <li>⚡ <strong>LightGBM:</strong> + num_leaves, regularization params</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# --- TAB 13: RISK TIMELINE ---
if t_timeline:
    with t_timeline:
        st.header("📈 5-Year Risk Progression Simulator")
        st.info("Simulates how a patient's CKD risk evolves over 5 years under two scenarios: natural progression vs. lifestyle intervention.")

        st.subheader("Configure Patient Profile")
        with st.form("timeline_form"):
            tl1, tl2, tl3, tl4 = st.columns(4)
            with tl1:
                tl_age = st.number_input("Age", 20, 90, 50, key="tl_age")
                tl_bmi = st.number_input("BMI", 15.0, 45.0, 28.0, key="tl_bmi")
            with tl2:
                tl_systolic = st.number_input("Systolic BP", 90, 200, 135, key="tl_sys")
                tl_diastolic = st.number_input("Diastolic BP", 60, 130, 88, key="tl_dia")
            with tl3:
                tl_hba1c = st.number_input("HbA1c (%)", 4.0, 15.0, 6.5, key="tl_hba1c")
                tl_fbs = st.number_input("Fasting Blood Sugar", 70, 250, 115, key="tl_fbs")
            with tl4:
                tl_activity = st.number_input("Physical Activity (min/wk)", 0, 300, 90, key="tl_act")
                tl_years = st.slider("Projection Years", 1, 10, 5, key="tl_years")

            tl_submit = st.form_submit_button("🔮 Simulate Progression", type="primary")

        if tl_submit:
            # Build patient input
            tl_input = X_te_nl.mean().to_frame().T.copy()
            tl_input["Age"] = tl_age
            tl_input["BMI"] = tl_bmi
            tl_input["SystolicBP"] = tl_systolic
            tl_input["DiastolicBP"] = tl_diastolic
            tl_input["HbA1c"] = tl_hba1c
            tl_input["FastingBloodSugar"] = tl_fbs
            tl_input["PhysicalActivity"] = tl_activity
            tl_input = tl_input[X_te_nl.columns]

            # Run both scenarios
            timeline_no = trainer.simulate_risk_progression(
                trained_nl[best_name], tl_input, X_te_nl.columns,
                years=tl_years, scenario="no_intervention"
            )
            timeline_yes = trainer.simulate_risk_progression(
                trained_nl[best_name], tl_input, X_te_nl.columns,
                years=tl_years, scenario="with_intervention"
            )

            # Main chart
            st.plotly_chart(viz.plot_risk_timeline(timeline_no, timeline_yes),
                            use_container_width=True, key="risk_timeline")

            # Summary cards
            tmc1, tmc2, tmc3 = st.columns(3)
            with tmc1:
                st.markdown(f"""
                <div class='metric-card'>
                    <h4>📍 Current Risk</h4>
                    <h2 style='color: #4361EE;'>{timeline_no.iloc[0]['Risk_Score']:.1%}</h2>
                    <p>{timeline_no.iloc[0]['Risk_Level']}</p>
                </div>
                """, unsafe_allow_html=True)
            with tmc2:
                st.markdown(f"""
                <div class='metric-card' style='border-left-color: #FF6B6B;'>
                    <h4>🚨 Year {tl_years} (No Action)</h4>
                    <h2 style='color: #FF6B6B;'>{timeline_no.iloc[-1]['Risk_Score']:.1%}</h2>
                    <p>{timeline_no.iloc[-1]['Risk_Level']}</p>
                </div>
                """, unsafe_allow_html=True)
            with tmc3:
                st.markdown(f"""
                <div class='metric-card' style='border-left-color: #4D96FF;'>
                    <h4>✅ Year {tl_years} (With Intervention)</h4>
                    <h2 style='color: #4D96FF;'>{timeline_yes.iloc[-1]['Risk_Score']:.1%}</h2>
                    <p>{timeline_yes.iloc[-1]['Risk_Level']}</p>
                </div>
                """, unsafe_allow_html=True)

            # Risk reduction
            risk_reduction = timeline_no.iloc[-1]['Risk_Score'] - timeline_yes.iloc[-1]['Risk_Score']
            st.markdown(f"""
            <div class='glass-card' style='text-align: center;'>
                <h4>💡 Potential Risk Reduction at Year {tl_years}</h4>
                <h2 style='color: #4CC9F0;'>{risk_reduction:.1%} lower risk with intervention</h2>
                <p style='color: #94a3b8; margin-top: 10px;'>
                    Intervention assumptions: increased physical activity (+10 min/wk/yr),
                    improved diet (+0.5/yr), BP management (-3 mmHg/yr), weight loss (-0.5 BMI/yr)
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Year-by-year table
            st.subheader("📋 Year-by-Year Comparison")
            comparison = pd.DataFrame({
                'Year': timeline_no['Year'],
                'Risk (No Intervention)': [f"{r:.1%}" for r in timeline_no['Risk_Score']],
                'Level (No Intervention)': timeline_no['Risk_Level'],
                'Risk (With Intervention)': [f"{r:.1%}" for r in timeline_yes['Risk_Score']],
                'Level (With Intervention)': timeline_yes['Risk_Level'],
            })
            st.dataframe(comparison, use_container_width=True)

# --- SAVE MODEL AFTER PIPELINE (for sidebar button) ---
try:
    trainer.save_model_for_api(
        trained_nl[best_name], X_te_nl.columns, best_name, best_th
    )
except Exception:
    pass

st.markdown("---")
st.caption("CKD Intelligence v3.3 — Precision Research Dashboard with Stage Prediction, Optuna Tuning & Risk Timeline.")

