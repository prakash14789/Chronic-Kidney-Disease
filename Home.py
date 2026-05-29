import streamlit as st
from ckd_app.dashboard import tabs as app_tabs
from ckd_app.dashboard.state import inject_custom_css, check_login, setup_sidebar, run_full_pipeline, viz, load_lottie_file, save_best_model_state
from streamlit_lottie import st_lottie

# Must be the very first Streamlit command
st.set_page_config(page_title="CKD Intelligence Dashboard", page_icon="🏠", layout="wide")

inject_custom_css()
check_login()

sample_size, use_cv = setup_sidebar("Home")

# Run the cached pipeline
(df_full, df_sample, res_f, roc_f, res_nl, roc_nl, pr_nl,
 trained_nl, X_tr_nl, X_te_nl, y_tr_f, y_te_f, best_th, res_no_smote) = run_full_pipeline(sample_size, use_cv=use_cv)
save_best_model_state(res_nl, trained_nl, best_th, X_te_nl)

best_name = res_nl.iloc[0]["Model"]
ckd_pct = df_full['Diagnosis'].mean() * 100


st.markdown(f"""
<div class='welcome-card'>
    <h1 style='margin: 0; font-weight: 800; font-family: "Space Grotesk", sans-serif; font-size: 2.8rem;'>🧬 Chronic Kidney Disease Clinical Intelligence</h1>
    <p style='color: #94a3b8; font-size: 1.15rem; margin-top: 10px; margin-bottom: 0;'>
        Predictive clinical intelligence portal powered by machine learning. High-fidelity risk analysis, feature impact explainability, and treatment simulation.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class='kpi-row'>
    <div class='kpi-box'>
        <p>🏆 Best Model</p>
        <h2>{best_name}</h2>
    </div>
    <div class='kpi-box'>
        <p>🎯 Balanced Accuracy</p>
        <div class="animate-number" data-target="{res_nl.iloc[0]['Balanced Accuracy'] * 100}" data-percent="true">0%</div>
    </div>
    <div class='kpi-box'>
        <p>📈 ROC-AUC</p>
        <div class="animate-number" data-target="{res_nl.iloc[0]['ROC-AUC']}" data-decimal="true">0.0000</div>
    </div>
    <div class='kpi-box'>
        <p>📊 Dataset Size</p>
        <div class="animate-number" data-target="{len(df_full)}">0</div>
        <p style='font-size: 0.8rem; opacity: 0.6;'>Sampled to {sample_size}</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
# Only render Data Audit here. Other functionalities are on other pages.
app_tabs.render_data_audit(df_full, df_sample, viz, ckd_pct)
