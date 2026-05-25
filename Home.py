import streamlit as st
import app_tabs
from core_state import inject_custom_css, check_login, setup_sidebar, run_full_pipeline, viz

# Must be the very first Streamlit command
st.set_page_config(page_title="CKD Intelligence Dashboard", page_icon="🏠", layout="wide")

inject_custom_css()
check_login()

sample_size, use_cv = setup_sidebar("Home")

# Run the cached pipeline
(df_full, df_sample, res_f, roc_f, res_nl, roc_nl, pr_nl,
 trained_nl, X_tr_nl, X_te_nl, y_tr_f, y_te_f, best_th, res_no_smote) = run_full_pipeline(sample_size, use_cv=use_cv)

best_name = res_nl.iloc[0]["Model"]
ckd_pct = df_full['Diagnosis'].mean() * 100

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

st.markdown("---")
# Only render Data Audit here. Other functionalities are on other pages.
app_tabs.render_data_audit(df_full, df_sample, viz, ckd_pct)
