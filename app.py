import streamlit as st
import pandas as pd
import numpy as np
from data_processor import CKDDataProcessor
from model_trainer import CKDModelTrainer
from visualizer import CKDVisualizer

# Page Config
st.set_page_config(page_title="CKD Clinical Intelligence", page_icon="🧬", layout="wide")

# Custom Styles
st.markdown("""
    <style>
    .stTabs [aria-selected="true"] { background-color: #0d6efd !important; color: white !important; }
    .metric-card { 
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
        color: white !important; 
        padding: 20px; 
        border-radius: 12px; 
        border-left: 6px solid #3b82f6; 
        margin-bottom: 20px;
    }
    .metric-card h4, .metric-card p { color: white !important; margin: 0; }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #0d6efd;
    }
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
    st.info("Now with Live Patient Diagnosis Tool.")

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
    pipes_nl = trainer.get_v3_pipelines(n_neg, n_pos)
    res_nl, roc_nl, pr_nl, trained_nl = trainer.run_v3_experiment(X_tr_nl, X_te_nl, y_tr_f, y_te_f, pipelines=pipes_nl)
    
    return df_full, df_sample, res_nl, roc_nl, pr_nl, trained_nl, X_te_nl, y_te_f

df_full, df_sample, res_nl, roc_nl, pr_nl, trained_nl, X_te_nl, y_te_f = run_full_pipeline(sample_size)

# Main UI
tabs = st.tabs(["📊 Performance Dashboard", "🧠 Model Interpretation", "🏥 Patient Diagnosis Tool"])

# --- TAB 1: DASHBOARD ---
with tabs[0]:
    st.title("CKD Clinical Dashboard")
    ckd_pct = (df_full['Diagnosis'].mean() * 100)
    c1, c2, c3 = st.columns(3)
    with c1: st.plotly_chart(viz.plot_class_distribution(df_full, ckd_pct), use_container_width=True, key="dist")
    with c2: st.plotly_chart(viz.plot_precision_recall_f1(res_nl), use_container_width=True, key="met")
    with c3: st.plotly_chart(viz.plot_age_distribution(df_sample), use_container_width=True, key="age")
    st.subheader("Model Rankings")
    st.dataframe(res_nl[['Model', 'Balanced Accuracy', 'Macro F1', 'ROC-AUC']], use_container_width=True)

# --- TAB 2: SHAP ---
with tabs[1]:
    st.header("🧠 Research Interpretation (SHAP)")
    best_name = res_nl.iloc[0]["Model"]
    with st.spinner("Calculating Global SHAP..."):
        explainer, shap_values, X_df = trainer.get_shap_explainer(trained_nl[best_name], X_te_nl)
    cs1, cs2 = st.columns(2)
    with cs1: st.pyplot(viz.plot_shap_bar(explainer, shap_values, X_df, best_name))
    with cs2: st.pyplot(viz.plot_shap_summary(explainer, shap_values, X_df, best_name))

# --- TAB 3: DIAGNOSIS TOOL ---
with tabs[2]:
    st.header("🏥 Individual Patient Diagnosis")
    st.markdown("Input patient demographics to predict CKD risk and see individual contributing factors.")
    
    with st.form("diagnosis_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", 20, 90, 50)
            gender = st.selectbox("Gender", ["Male", "Female"])
            bmi = st.number_input("BMI", 15.0, 45.0, 25.0)
            ethnicity = st.selectbox("Ethnicity", ["Caucasian", "African American", "Asian", "Hispanic"])
            
        with col2:
            smoking = st.selectbox("Smoking Status", ["Non-smoker", "Smoker"])
            alcohol = st.selectbox("Alcohol Consumption", ["None", "Moderate", "Heavy"])
            activity = st.number_input("Physical Activity (mins/week)", 0, 300, 150)
            diet = st.slider("Diet Quality Score", 0, 10, 5)
            
        with col3:
            systolic = st.number_input("Systolic BP", 90, 200, 120)
            diastolic = st.number_input("Diastolic BP", 60, 120, 80)
            fbs = st.number_input("Fasting Blood Sugar", 70, 200, 100)
            adherence = st.selectbox("Medication Adherence", ["Low", "Moderate", "High"])

        submit = st.form_submit_button("🩺 Run Diagnosis")

    if submit:
        # Prepare Input Data
        input_data = pd.DataFrame([{
            "Age": age, "Gender": 1 if gender == "Male" else 0, "BMI": bmi,
            "Ethnicity": 0, # Simplified for demo
            "SocioeconomicStatus": 1, "EducationLevel": 1,
            "Smoking": 1 if smoking == "Smoker" else 0,
            "AlcoholConsumption": 1 if alcohol != "None" else 0,
            "PhysicalActivity": activity, "DietQuality": diet, "SleepQuality": 7,
            "FamilyHistoryKidneyDisease": 0, "FamilyHistoryHypertension": 0, "FamilyHistoryDiabetes": 0,
            "PreviousAcuteKidneyInjury": 0, "UrinaryTractInfections": 0,
            "SystolicBP": systolic, "DiastolicBP": diastolic, "FastingBloodSugar": fbs, "HbA1c": 5.5,
            "SerumElectrolytesSodium": 140, "SerumElectrolytesPotassium": 4.0, 
            "SerumElectrolytesCalcium": 9.5, "SerumElectrolytesPhosphorus": 3.5,
            "HemoglobinLevels": 14.0, "CholesterolTotal": 200, "CholesterolLDL": 100, 
            "CholesterolHDL": 50, "CholesterolTriglycerides": 150,
            "ACEInhibitors": 0, "Diuretics": 0, "NSAIDsUse": 0, "Statins": 0,
            "AntidiabeticMedications": 0, "Edema": 0, "FatigueLevels": 0, "NauseaVomiting": 0,
            "MuscleCramps": 0, "Itching": 0, "QualityOfLifeScore": 80, "HeavyMetalsExposure": 0,
            "OccupationalExposureChemicals": 0, "WaterQuality": 1, "MedicalCheckupsFrequency": 2,
            "MedicationAdherence": 1 if adherence != "Low" else 0, "HealthLiteracy": 2,
            "Adherence": 1 if adherence == "High" else (0 if adherence == "Moderate" else -1)
        }])
        
        # Ensure all columns are present (even those dropped in nl)
        # We need the full set of columns that X_tr_nl had
        final_input = input_data[X_te_nl.columns]
        
        best_pipe = trained_nl[res_nl.iloc[0]["Model"]]
        prob = best_pipe.predict_proba(final_input)[0, 1]
        prediction = "CKD Likely" if prob > 0.5 else "Low Risk"
        
        st.markdown(f"""
        <div class='prediction-box'>
        <h3>Diagnosis: <span style='color: {"#E74C3C" if prob > 0.5 else "#27AE60"}'>{prediction}</span></h3>
        <h1>Risk Score: {prob:.1%}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Why this prediction?")
        # Calculate Local SHAP
        explainer_local, shap_vals_local, X_df_local = trainer.get_shap_explainer(best_pipe, final_input)
        st.pyplot(viz.plot_local_shap(explainer_local, shap_vals_local, X_df_local, patient_idx=0))

st.markdown("---")
st.caption("Clinical Intelligence System v3.1 — Professional Patient Diagnosis Interface.")
