import streamlit as st
import numpy as np
from ckd_app.dashboard import tabs as app_tabs
from ckd_app.dashboard.state import inject_custom_css, check_login, setup_sidebar, run_full_pipeline, get_cached_shap, viz, trainer, save_best_model_state

st.set_page_config(page_title="Interpretability", page_icon="🧠", layout="wide")

inject_custom_css()
check_login()
sample_size, use_cv = setup_sidebar("Interpretability")

# Fetch cached data
(df_full, df_sample, res_f, roc_f, res_nl, roc_nl, pr_nl,
 trained_nl, X_tr_nl, X_te_nl, y_tr_f, y_te_f, best_th, res_no_smote) = run_full_pipeline(sample_size, use_cv=use_cv)
save_best_model_state(res_nl, trained_nl, best_th, X_te_nl)

best_name = res_nl.iloc[0]["Model"]
y_proba_all = trained_nl[best_name].predict_proba(X_te_nl)[:, 1]

st.title("🧠 AI Interpretability & Analysis")
st.markdown("Understand *why* the models make their decisions and where they are prone to errors.")

t1, t2, t3 = st.tabs(
    ["🎯 Threshold Tuning", "🧠 Global SHAP", "🔬 Deep Analysis"],
    key="active_tab_interpretability",
    on_change="rerun"
)

with t1:
    app_tabs.render_threshold_tuning(y_te_f, y_proba_all, best_th, trainer, trained_nl, best_name, X_te_nl, viz)

with t2:
    app_tabs.render_shap_interpretation(trained_nl, best_name, X_te_nl, get_cached_shap, trainer, viz)

with t3:
    app_tabs.render_deep_analysis(df_sample, y_te_f, y_proba_all, trained_nl, best_name, X_te_nl, X_tr_nl, y_tr_f, trainer, viz)
