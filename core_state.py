import streamlit as st
from data_processor import CKDDataProcessor
from model_trainer import CKDModelTrainer
from visualizer import CKDVisualizer, COLORS
from report_generator import CKDReportGenerator
from stage_predictor import CKDStagePredictor

# Initialize instances
processor = CKDDataProcessor()
trainer = CKDModelTrainer()
viz = CKDVisualizer()
reporter = CKDReportGenerator()

def inject_custom_css():
    theme = st.sidebar.selectbox("🎨 Theme", ["Premium Dark (Glass)", "Clinical Light"], index=0, key="global_theme")
    
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

        .kpi-row {{ display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }}
        
        .kpi-box {{
            flex: 1; 
            min-width: 200px;
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

def check_login():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['role'] = None

    if not st.session_state['logged_in']:
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
        st.stop()

def setup_sidebar():
    with st.sidebar:
        st.title("🧬 CKD Intelligence")
        st.image("https://cdn-icons-png.flaticon.com/512/3067/3067451.png", width=80)
        sample_size = st.slider("Stratified Sample Size", 1000, 10000, 5000, key="global_sample")
        use_cv = st.checkbox("Enable Cross-Validation (Slower)", value=False, key="global_cv")
        st.divider()
        st.markdown("[🔗 View Source on GitHub](https://github.com/prakash14789/Chronic-Kidney-Disease)")
        st.success("v3.3 Research Pipeline Active")
        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state['logged_in'] = False
            st.session_state['role'] = None
            st.rerun()
    return sample_size, use_cv

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
    
    pipes_f = trainer.get_v3_pipelines(n_neg, n_pos, use_smote=True)
    res_f, roc_f, pr_f, trained_f = trainer.run_v3_experiment(X_tr_f, X_te_f, y_tr_f, y_te_f, pipelines=pipes_f, use_cv=use_cv)
    
    pipes_nl = trainer.get_v3_pipelines(n_neg, n_pos, use_smote=True)
    res_nl, roc_nl, pr_nl, trained_nl = trainer.run_v3_experiment(X_tr_nl, X_te_nl, y_tr_f, y_te_f, pipelines=pipes_nl, use_cv=use_cv)
    
    pipes_no_smote = trainer.get_v3_pipelines(n_neg, n_pos, use_smote=False)
    res_no_smote, _, _, _ = trainer.run_v3_experiment(X_tr_nl, X_te_nl, y_tr_f, y_te_f, pipelines=pipes_no_smote, use_cv=use_cv)
    
    best_name = res_nl.iloc[0]["Model"]
    y_proba = trained_nl[best_name].predict_proba(X_te_nl)[:, 1]
    _, best_th = trainer.tune_threshold(y_te_f, y_proba)
    
    return (df_full, df_sample, res_f, roc_f, res_nl, roc_nl, pr_nl,
            trained_nl, X_tr_nl, X_te_nl, y_tr_f, y_te_f, best_th, res_no_smote)

@st.cache_resource
def get_cached_shap(_model, X_test):
    return trainer.get_shap_explainer(_model, X_test)
