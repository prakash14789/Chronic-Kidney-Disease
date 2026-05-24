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
        COLORS["accent"] = "#38BDF8"
        
        bg_gradient = "linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)"
        card_bg = "rgba(255, 255, 255, 0.7)"
        border_color = "rgba(255, 255, 255, 0.4)"
        shadow = "0 8px 32px 0 rgba(31, 38, 135, 0.07)"
        text_color = COLORS["text"]
        accent_color = COLORS["primary"]
    else:
        COLORS["bg"] = "#0B1120"
        COLORS["text"] = "#F8FAFC"
        COLORS["primary"] = "#8B5CF6" # Vibrant Purple
        COLORS["secondary"] = "#3B82F6" # Electric Blue
        COLORS["accent"] = "#10B981" # Emerald Green
        
        bg_gradient = "radial-gradient(circle at top left, #1e1b4b, #0B1120 70%)"
        card_bg = "rgba(17, 24, 39, 0.65)"
        border_color = "rgba(255, 255, 255, 0.1)"
        shadow = "0 8px 32px 0 rgba(0, 0, 0, 0.3)"
        text_color = COLORS["text"]
        accent_color = COLORS["primary"]

    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

        html, body, [class*="css"], .stApp {{
            font-family: 'Outfit', sans-serif !important;
        }}

        .stApp {{ 
            background: {bg_gradient}; 
            color: {text_color}; 
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        @keyframes pulseGlow {{
            0% {{ box-shadow: 0 0 0 0 rgba(139, 92, 246, 0.4); }}
            70% {{ box-shadow: 0 0 15px 10px rgba(139, 92, 246, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(139, 92, 246, 0); }}
        }}

        /* Enhanced Glass Cards */
        .metric-card, .glass-card, .kpi-box {{ 
            background: {card_bg}; 
            backdrop-filter: blur(20px) saturate(200%);
            -webkit-backdrop-filter: blur(20px) saturate(200%);
            color: {text_color} !important; 
            border-radius: 20px; 
            border: 1px solid {border_color};
            box-shadow: {shadow};
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            animation: fadeIn 0.8s ease-out forwards;
            position: relative;
            overflow: hidden;
        }}
        
        /* Premium glowing border on hover */
        .metric-card:hover, .glass-card:hover, .kpi-box:hover {{
            transform: translateY(-8px) scale(1.02);
            border-color: {accent_color};
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3), 0 0 20px rgba(139, 92, 246, 0.2);
        }}

        .metric-card {{ padding: 28px; margin-bottom: 24px; border-left: 6px solid {accent_color}; }}
        .glass-card {{ padding: 28px; margin-bottom: 20px; }}
        
        .metric-card h4, .metric-card p {{ color: {text_color} !important; margin: 0; }}

        .kpi-row {{ display: flex; gap: 20px; margin-bottom: 30px; flex-wrap: wrap; }}
        
        .kpi-box {{
            flex: 1; 
            min-width: 220px;
            padding: 30px 20px; 
            text-align: center;
        }}

        .kpi-box h2 {{ 
            background: linear-gradient(135deg, {COLORS["primary"]}, {COLORS["secondary"]});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 10px 0 0 0; 
            font-size: 2.8rem; 
            font-weight: 800; 
            letter-spacing: -1px;
        }}
        
        .kpi-box p {{ 
            color: {text_color}; 
            margin: 0; 
            font-size: 1rem; 
            font-weight: 500;
            opacity: 0.8; 
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        /* Ultra-premium glowing buttons */
        .stButton>button {{
            background: linear-gradient(135deg, {COLORS["primary"]}, {COLORS["secondary"]}) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            letter-spacing: 0.5px !important;
            padding: 10px 24px !important;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
            position: relative;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3) !important;
        }}
        
        .stButton>button:hover {{
            transform: translateY(-3px) scale(1.05) !important;
            box-shadow: 0 8px 25px rgba(139, 92, 246, 0.5) !important;
            animation: pulseGlow 1.5s infinite;
        }}
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {{
            background: rgba(11, 17, 32, 0.8) !important;
            backdrop-filter: blur(20px);
            border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        }}
        
        /* Gradient Headers */
        h1, h2, h3 {{
            letter-spacing: -0.05em;
        }}
        
        .stMarkdown h1 {{
            background: linear-gradient(to right, {COLORS["primary"]}, {COLORS["accent"]});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
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
