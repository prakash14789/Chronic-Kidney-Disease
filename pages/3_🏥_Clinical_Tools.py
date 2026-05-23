import streamlit as st
import app_tabs
from core_state import (
    inject_custom_css, check_login, setup_sidebar, run_full_pipeline,
    get_cached_shap, viz, trainer, reporter
)

st.set_page_config(page_title="Clinical Tools", page_icon="🏥", layout="wide")

inject_custom_css()
check_login()
_, use_cv = setup_sidebar()

# Fetch cached data
(df_full, df_sample, res_f, roc_f, res_nl, roc_nl, pr_nl,
 trained_nl, X_tr_nl, X_te_nl, y_tr_f, y_te_f, best_th, res_no_smote) = run_full_pipeline(sample_size, use_cv=use_cv)

best_name = res_nl.iloc[0]["Model"]
y_proba_all = trained_nl[best_name].predict_proba(X_te_nl)[:, 1]

st.title("🏥 Clinical Tools & Diagnostics")
st.markdown("Precision tools for patient diagnosis, batch risk assessment, and longitudinal tracking.")

t1, t2, t3, t4, t5 = st.tabs([
    "🏥 Patient Diagnosis", "📂 Batch Diagnosis", "🏗️ CKD Staging",
    "📈 Risk Timeline", "📚 Patient History"
])

with t1:
    app_tabs.render_patient_diagnosis(X_te_nl, trained_nl, best_name, best_th, trainer, viz, reporter, get_cached_shap, y_proba_all, X_tr_nl, y_tr_f)

with t2:
    app_tabs.render_batch_diagnosis(X_te_nl, trained_nl, best_name, best_th, trainer)

with t3:
    from stage_predictor import CKDStagePredictor
    
    @st.cache_data
    def run_stage_pipeline(_df_full, sample_n):
        sp = CKDStagePredictor()
        results = sp.train(_df_full, sample_n=sample_n)
        return sp, results

    with st.spinner("Training multi-class stage predictor..."):
        stage_model, stage_results = run_stage_pipeline(df_full, sample_size)

    app_tabs.render_stage_prediction(stage_results, viz)

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

with t4:
    app_tabs.render_risk_timeline(X_te_nl, trained_nl, best_name, trainer, viz)

with t5:
    app_tabs.render_patient_history(viz)
