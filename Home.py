import streamlit as st
import app_tabs
from core_state import inject_custom_css, check_login, setup_sidebar, run_full_pipeline, viz, load_lottieurl
from streamlit_lottie import st_lottie

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

hc1, hc2, hc3 = st.columns([1, 2, 1])
with hc2:
    lottie_dna = load_lottieurl("https://lottie.host/7900b8bf-dc7d-4581-91a6-733c0d7ff3eb/uU59NlWv7Z.json")
    if lottie_dna:
        st_lottie(lottie_dna, height=120, key="home_lottie")
    else:
        st.markdown("<div style='text-align: center;'><span style='font-size: 3rem;'>🧬</span></div>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; margin-top: -15px; margin-bottom: 30px; font-weight: 800; letter-spacing: -0.05em;'>CKD Clinical Intelligence</h1>", unsafe_allow_html=True)

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
