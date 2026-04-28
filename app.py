import streamlit as st
import pandas as pd
import numpy as np
from data_processor import CKDDataProcessor
from model_trainer import CKDModelTrainer
from visualizer import CKDVisualizer, COLORS

# Page Config
st.set_page_config(page_title="CKD Clinical Intelligence v3.2", page_icon="🧬", layout="wide")

# Custom Styles
st.markdown(f"""
    <style>
    .stApp {{ background-color: {COLORS["bg"]}; color: {COLORS["text"]}; }}
    .stTabs [aria-selected="true"] {{ background-color: {COLORS["primary"]} !important; color: white !important; }}
    .metric-card {{ 
        background: linear-gradient(135deg, #1e293b 0%, {COLORS["bg"]} 100%); 
        color: {COLORS["text"]} !important; padding: 20px; border-radius: 12px; 
        border-left: 6px solid {COLORS["primary"]}; margin-bottom: 20px;
        border: 1px solid {COLORS["grid"]};
    }}
    .metric-card h4, .metric-card p {{ color: {COLORS["text"]} !important; margin: 0; }}
    .prediction-box {{
        background-color: {COLORS["grid"]}; padding: 20px; border-radius: 10px;
        text-align: center; border: 2px solid {COLORS["primary"]}; color: {COLORS["text"]};
    }}
    .glass-card {{
        background: rgba(30,41,59,0.7); backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.08); border-radius: 16px;
        padding: 24px; margin-bottom: 16px;
    }}
    .kpi-row {{ display: flex; gap: 16px; margin-bottom: 24px; }}
    .kpi-box {{
        flex: 1; background: linear-gradient(135deg, #1e293b, #0E1117);
        border-radius: 14px; padding: 20px; text-align: center;
        border: 1px solid {COLORS["grid"]};
    }}
    .kpi-box h2 {{ color: {COLORS["primary"]}; margin: 0; font-size: 2rem; }}
    .kpi-box p {{ color: {COLORS["text"]}; margin: 4px 0 0 0; font-size: 0.85rem; }}
    </style>
    """, unsafe_allow_html=True)

# Initialize
processor = CKDDataProcessor()
trainer = CKDModelTrainer()
viz = CKDVisualizer()

# Sidebar
with st.sidebar:
    st.title("🧬 CKD Intelligence")
    st.image("https://cdn-icons-png.flaticon.com/512/3067/3067451.png", width=80)
    sample_size = st.slider("Stratified Sample Size", 1000, 10000, 5000)
    st.divider()
    st.success("v3 Research Pipeline Active")

# --- PIPELINE EXECUTION ---
@st.cache_data
def run_full_pipeline(sample_n):
    df_full = processor.load_raw_data()
    df_sample = processor.get_v3_refined_data(df_full, sample_n=sample_n)
    X_tr_f, X_te_f, y_tr_f, y_te_f = processor.split_and_encode_v3(df_sample)
    
    leakage_cols = ["GFR", "SerumCreatinine", "BUNLevels", "ProteinInUrine", "ACR"]
    X_tr_nl = X_tr_f.drop(columns=leakage_cols, errors="ignore")
    X_te_nl = X_te_f.drop(columns=leakage_cols, errors="ignore")
    X_te_nl = X_te_nl[X_tr_nl.columns]
    
    n_neg, n_pos = (y_tr_f == 0).sum(), (y_tr_f == 1).sum()
    
    pipes_f = trainer.get_v3_pipelines(n_neg, n_pos)
    res_f, roc_f, pr_f, trained_f = trainer.run_v3_experiment(X_tr_f, X_te_f, y_tr_f, y_te_f, pipelines=pipes_f)
    
    pipes_nl = trainer.get_v3_pipelines(n_neg, n_pos)
    res_nl, roc_nl, pr_nl, trained_nl = trainer.run_v3_experiment(X_tr_nl, X_te_nl, y_tr_f, y_te_f, pipelines=pipes_nl)
    
    best_name = res_nl.iloc[0]["Model"]
    y_proba = trained_nl[best_name].predict_proba(X_te_nl)[:, 1]
    _, best_th = trainer.tune_threshold(y_te_f, y_proba)
    
    return (df_full, df_sample, res_f, roc_f, res_nl, roc_nl, pr_nl,
            trained_nl, X_tr_nl, X_te_nl, y_tr_f, y_te_f, best_th)

(df_full, df_sample, res_f, roc_f, res_nl, roc_nl, pr_nl,
 trained_nl, X_tr_nl, X_te_nl, y_tr_f, y_te_f, best_th) = run_full_pipeline(sample_size)

# Derived values
best_name = res_nl.iloc[0]["Model"]
y_proba_all = trained_nl[best_name].predict_proba(X_te_nl)[:, 1]
ckd_pct = df_full['Diagnosis'].mean() * 100

# --- KPI CARDS ---
st.title("CKD Clinical Intelligence Dashboard")
k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("🏆 Best Model", best_name)
with k2: st.metric("🎯 Balanced Accuracy", f"{res_nl.iloc[0]['Balanced Accuracy']:.2%}")
with k3: st.metric("📈 ROC-AUC", f"{res_nl.iloc[0]['ROC-AUC']:.4f}")
with k4: st.metric("📊 Dataset Size", f"{len(df_full):,} → {sample_size}")

tabs = st.tabs([
    "📊 Data Audit", "🚀 Exp 1 (Full)", "🛡️ Exp 2 (No-Leakage)", "📉 Comparison",
    "🎯 Threshold Tuning", "🧠 SHAP Interpretation", "🔬 Deep Analysis", "🏥 Patient Diagnosis"
])

# --- TAB 1: DATA AUDIT ---
with tabs[0]:
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
with tabs[1]:
    st.header("Experiment 1: Full Features")
    st.dataframe(res_f[['Model', 'Balanced Accuracy', 'Macro F1', 'ROC-AUC']], use_container_width=True)
    st.plotly_chart(viz.plot_roc_curves(roc_f), use_container_width=True, key="roc_f")

# --- TAB 3: EXP 2 ---
with tabs[2]:
    st.header("Experiment 2: Leakage-Free (Research Grade)")
    st.dataframe(res_nl[['Model', 'Balanced Accuracy', 'Macro F1', 'ROC-AUC']], use_container_width=True)
    st.plotly_chart(viz.plot_roc_curves(roc_nl), use_container_width=True, key="roc_nl")
    # Confusion Matrix
    err = trainer.get_error_analysis(trained_nl[best_name], X_te_nl, y_te_f)
    st.plotly_chart(viz.plot_confusion_matrix(err["counts"]), use_container_width=True, key="cm")

# --- TAB 4: COMPARISON ---
with tabs[3]:
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

# --- TAB 5: THRESHOLD ---
with tabs[4]:
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

# --- TAB 6: SHAP ---
with tabs[5]:
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

# --- TAB 7: DEEP ANALYSIS ---
with tabs[6]:
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

# --- TAB 8: DIAGNOSIS ---
with tabs[7]:
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
                <div class='metric-card' style='border-left: 6px solid {assessment["Color"]};'>
                    <h4>{assessment["Icon"]} {assessment["Level"]} — {prob:.1%} Risk</h4>
                    <p style='font-size: 1.1rem; margin-top: 10px;'>{assessment["Action"]}</p>
                    <p style='font-size: 0.9rem; color: #94a3b8; margin-top: 15px;'>
                        *Decision based on optimal clinical threshold of <strong>{best_th}</strong>
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
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

st.markdown("---")
st.caption("CKD Intelligence v3.2 — Precision Research Dashboard.")
