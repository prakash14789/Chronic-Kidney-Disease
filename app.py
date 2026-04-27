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
        color: {COLORS["text"]} !important; 
        padding: 20px; 
        border-radius: 12px; 
        border-left: 6px solid {COLORS["primary"]}; 
        margin-bottom: 20px;
        border: 1px solid {COLORS["grid"]};
    }}
    .metric-card h4, .metric-card p {{ color: {COLORS["text"]} !important; margin: 0; }}
    .prediction-box {{
        background-color: {COLORS["grid"]};
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid {COLORS["primary"]};
        color: {COLORS["text"]};
    }}
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
    
    n_neg, n_pos = (y_tr_f == 0).sum(), (y_tr_f == 1).sum()
    
    # Run Full
    pipes_f = trainer.get_v3_pipelines(n_neg, n_pos)
    res_f, roc_f, pr_f, trained_f = trainer.run_v3_experiment(X_tr_f, X_te_f, y_tr_f, y_te_f, pipelines=pipes_f)
    
    # Run No-Leakage
    pipes_nl = trainer.get_v3_pipelines(n_neg, n_pos)
    res_nl, roc_nl, pr_nl, trained_nl = trainer.run_v3_experiment(X_tr_nl, X_te_nl, y_tr_f, y_te_f, pipelines=pipes_nl)
    
    # Calculate Optimal Threshold for Best Model
    best_name = res_nl.iloc[0]["Model"]
    y_proba = trained_nl[best_name]["calibrated"].predict_proba(X_te_nl)[:, 1]
    _, best_th = trainer.tune_threshold(y_te_f, y_proba)
    
    # Audit: Print Adherence Mapping for verification
    print(f"DEBUG: Adherence Classes: {processor.le_adherence.classes_}")
    
    return processor, df_full, df_sample, res_f, roc_f, res_nl, roc_nl, pr_nl, trained_nl, X_te_nl, y_te_f, best_th

processor, df_full, df_sample, res_f, roc_f, res_nl, roc_nl, pr_nl, trained_nl, X_te_nl, y_te_f, best_th = run_full_pipeline(sample_size)

# Main UI
st.title("CKD Clinical Intelligence Dashboard")

tabs = st.tabs([
    "📊 Data Audit", 
    "🚀 Exp 1 (Full)", 
    "🛡️ Exp 2 (No-Leakage)", 
    "📉 Comparison", 
    "🎯 Threshold Tuning",
    "🧠 SHAP Interpretation",
    "🏥 Patient Diagnosis"
])

# --- TAB 1: DATA AUDIT ---
with tabs[0]:
    st.header("1. Class Imbalance & Audit")
    ckd_pct = (df_full['Diagnosis'].mean() * 100)
    col1, col2 = st.columns([1, 2])
    with col1: st.plotly_chart(viz.plot_class_distribution(df_full, ckd_pct), use_container_width=True, key="dist")
    with col2:
        st.write("### v3 Hygiene Checklist")
        st.checkbox("LabelEncoder fitted on Train only", value=True, disabled=True)
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
    best_name = res_nl.iloc[0]["Model"]
    y_proba = trained_nl[best_name]["calibrated"].predict_proba(X_te_nl)[:, 1]
    th_df, _ = trainer.tune_threshold(y_te_f, y_proba) # Use pre-calculated best_th
    st.plotly_chart(viz.plot_threshold_tuning(th_df, best_th), use_container_width=True, key="th_tune")
    if st.button("Run Sanity Check"):
        acc = trainer.run_sanity_check(trained_nl[best_name]["calibrated"], X_te_nl, y_te_f)
        st.metric("Shuffled Balanced Accuracy", f"{acc:.2%}")

# --- TAB 6: SHAP ---
with tabs[5]:
    st.header("🧠 Model Interpretation (SHAP)")
    try:
        with st.spinner("Calculating Global SHAP..."):
            explainer, shap_values, X_df = trainer.get_shap_explainer(trained_nl[best_name], X_te_nl)
        st.write(f"✅ SHAP values type: `{type(shap_values).__name__}` | shape: `{np.array(shap_values).shape}`")
        cs1, cs2 = st.columns(2)
        with cs1: st.pyplot(viz.plot_shap_bar(explainer, shap_values, X_df, best_name))
        with cs2: st.pyplot(viz.plot_shap_summary(explainer, shap_values, X_df, best_name))
    except Exception as e:
        st.error(f"SHAP calculation failed: {e}")
        import traceback
        st.code(traceback.format_exc())

# --- TAB 7: DIAGNOSIS ---
with tabs[6]:
    st.header("🏥 Precision Patient Diagnosis")
    st.warning("⚠️ Prediction is based on partial clinical input. Missing features are estimated using population statistics (Median).")
    st.info("Fill in the clinical details below. Features not specified will be set to the population median for the research model.")
    
    with st.form("diag_form"):
        # Section 1: Demographics & Lifestyle
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
        
        # Section 2: Core Clinical Vitals
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

        # Section 3: Advanced Parameters (Optional)
        with st.expander("🛠️ Advanced Parameters (Optional - Defaults to Population Mean)"):
            st.write("Adjust these for a more precise clinical profile.")
            ca1, ca2, ca3 = st.columns(3)
            with ca1:
                sodium = st.number_input("Serum Sodium (mEq/L)", 130.0, 150.0, float(X_te_nl["SerumElectrolytesSodium"].median()))
                potassium = st.number_input("Serum Potassium (mEq/L)", 3.0, 6.0, float(X_te_nl["SerumElectrolytesPotassium"].median()))
            with ca2:
                fatigue = st.slider("Fatigue Level (0-10)", 0, 10, int(X_te_nl["FatigueLevels"].median()))
                edema = st.selectbox("Edema (Swelling)", ["No", "Yes"], index=int(X_te_nl["Edema"].median()))
            with ca3:
                qol = st.slider("Quality of Life Score", 0, 100, int(X_te_nl["QualityOfLifeScore"].median()))
                heavy_metals = st.checkbox("Heavy Metals Exposure", value=bool(X_te_nl["HeavyMetalsExposure"].median() > 0.5))

        sub = st.form_submit_button("🚀 Generate Precision Diagnosis")
        
    if sub:
        # 1. Use MEDIAN values as baseline for ALL features (more robust than mean)
        input_row = X_te_nl.median().to_frame().T.copy()
        
        # 2. Overwrite with user input (mapping names to CSV columns)
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
        
        # Binary / Categorical Mappings
        input_row["Gender"] = 1 if gender == "Male" else 0
        input_row["Smoking"] = 1 if smoking == "Yes" else 0
        input_row["FamilyHistoryKidneyDisease"] = 1 if fam_kidney else 0
        input_row["FamilyHistoryHypertension"] = 1 if fam_hyper else 0
        input_row["FamilyHistoryDiabetes"] = 1 if fam_diab else 0
        input_row["Edema"] = 1 if edema == "Yes" else 0
        input_row["HeavyMetalsExposure"] = 1 if heavy_metals else 0
        
        # Adherence mapping (Dynamic via LabelEncoder)
        input_row["Adherence"] = processor.le_adherence.transform([adherence])[0]

        # --- Intelligent Feature Alignment ---
        # If BP is high -> adjust related features
        if systolic > 140:
            input_row["FatigueLevels"] = max(input_row["FatigueLevels"].iloc[0], 6)
            input_row["QualityOfLifeScore"] = min(input_row["QualityOfLifeScore"].iloc[0], 50)

        # If sugar is high -> adjust HbA1c
        if fbs > 126:
            input_row["HbA1c"] = max(input_row["HbA1c"].iloc[0], 6.5)

        # If BMI high -> adjust cholesterol
        if bmi > 30:
            input_row["CholesterolTotal"] = max(input_row["CholesterolTotal"].iloc[0], 220)

        # If smoking -> worsen health indicators
        if smoking == "Yes":
            input_row["HemoglobinLevels"] = min(input_row["HemoglobinLevels"].iloc[0], 13)

        # 4. Ensure exact column order and alignment
        input_row = input_row[X_te_nl.columns]
        
        # 5. Prediction
        prob = trained_nl[best_name]["calibrated"].predict_proba(input_row)[0, 1]
        
        # DEBUG: Verify Adherence encoding (Temporary)
        st.write("Adherence Encoding Check:", X_te_nl["Adherence"].unique())
        
        assessment = trainer.get_clinical_assessment(prob, best_th)
        
        # Display Results
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            st.markdown(f"""
                <div class='prediction-box'>
                    <p style='color: {COLORS["text"]}; margin-bottom: 5px;'>Predicted Risk Probability</p>
                    <h2 style='color: {assessment["Color"]};'>{prob:.1%}</h2>
                    <hr style='border: 1px solid {COLORS["grid"]};'>
                    <p style='font-size: 1.2rem; font-weight: bold; color: {assessment["Color"]};'>
                        {assessment["Icon"]} {assessment["Level"]}
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            with st.expander("🔍 Model Confidence Details"):
                st.write(f"**Probability:** {prob:.4f}")
                st.write(f"**Optimal Threshold:** {best_th:.2f}")
                st.write(f"**Clinical Prediction:** {'CKD' if prob >= best_th else 'Not CKD'}")
            
        with col_res2:
            st.markdown(f"""
                <div class='metric-card' style='border-left: 6px solid {assessment["Color"]};'>
                    <h4>Clinical Recommendation</h4>
                    <p style='font-size: 1.1rem; margin-top: 10px;'>{assessment["Action"]}</p>
                    <p style='font-size: 0.9rem; color: #94a3b8; margin-top: 15px;'>
                        *Decision based on optimal clinical threshold of <strong>{best_th}</strong>
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("### Feature Contribution Analysis")
        st.info("The chart below shows which features pushed the risk up (Red) or down (Blue) for THIS specific patient.")
        
        # 6. Individualized SHAP
        st.write("Input shape:", input_row.shape)
        try:
            with st.spinner("Calculating local SHAP for this patient..."):
                explainer_loc, shap_values_loc, X_df_loc = trainer.get_shap_explainer(trained_nl[best_name], input_row)
                st.write(f"✅ Local SHAP type: `{type(shap_values_loc).__name__}` | shape: `{np.array(shap_values_loc).shape}`")
                st.pyplot(viz.plot_local_shap(explainer_loc, shap_values_loc, X_df_loc, patient_idx=0))
        except Exception as e:
            st.error(f"Local SHAP failed: {e}")
            import traceback
            st.code(traceback.format_exc())

st.markdown("---")
st.caption("CKD Intelligence v3.2 — Precision Research Dashboard.")
