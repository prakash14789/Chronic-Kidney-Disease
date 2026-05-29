import streamlit as st
from ckd_app.dashboard import tabs as app_tabs
from ckd_app.dashboard.state import inject_custom_css, check_login, setup_sidebar, run_full_pipeline, viz, trainer, save_best_model_state

st.set_page_config(page_title="Model Analytics", page_icon="🚀", layout="wide")

inject_custom_css()
check_login()
sample_size, use_cv = setup_sidebar("Model Analytics")

# Fetch cached data
(df_full, df_sample, res_f, roc_f, res_nl, roc_nl, pr_nl,
 trained_nl, X_tr_nl, X_te_nl, y_tr_f, y_te_f, best_th, res_no_smote) = run_full_pipeline(sample_size, use_cv=use_cv)
save_best_model_state(res_nl, trained_nl, best_th, X_te_nl)

best_name = res_nl.iloc[0]["Model"]
ckd_pct = df_full['Diagnosis'].mean() * 100

st.title("🚀 Model Analytics")
st.markdown("Explore detailed machine learning experiments, performance comparisons, and hyperparameter optimization.")

t1, t2, t3, t4, t5, t6 = st.tabs(
    ["🚀 Exp 1 (Full Features)", "🛡️ Exp 2 (No-Leakage)", "📉 Leakage Comparison", "🧬 SMOTE Insights", "⚡ Optuna Tuning", "📋 Model Registry"],
    key="active_tab_model_analytics",
    on_change="rerun"
)

with t1:
    app_tabs.render_exp1(res_f, roc_f, viz)

with t2:
    app_tabs.render_exp2(res_nl, roc_nl, trained_nl, best_name, X_te_nl, y_te_f, trainer, viz)

with t3:
    app_tabs.render_comparison(res_f, res_nl, ckd_pct, df_sample, viz)

with t4:
    app_tabs.render_smote_insights(res_nl, res_no_smote)

with t5:
    app_tabs.render_optuna_tuning(X_tr_nl, y_tr_f, X_te_nl, y_te_f, trainer, viz, res_nl)

with t6:
    app_tabs.render_model_registry(viz)
