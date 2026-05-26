import streamlit as st
import functools

# Monkey-patch plotly_chart to force theme=None so our custom premium theme applies
_original_plotly_chart = st.plotly_chart
@functools.wraps(_original_plotly_chart)
def _premium_plotly_chart(figure_or_data, use_container_width=False, theme=None, **kwargs):
    return _original_plotly_chart(figure_or_data, use_container_width=use_container_width, theme=None, **kwargs)
st.plotly_chart = _premium_plotly_chart

import streamlit.components.v1 as components
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

def render_guided_tour(page_name="Home"):
    if page_name == "Home":
        steps_js = """[
            { element: '[data-testid="stSidebar"]', popover: { title: 'Navigation Sidebar', description: 'Configure dataset size, toggle cross-validation, and navigate to different pages.', side: "right", align: 'start' } },
            { element: 'div.kpi-row', popover: { title: 'Key Metrics', description: 'These cards show the best model performance and ROC-AUC at a glance.', side: "bottom", align: 'start' } },
            { element: '[data-testid="stHeader"]', popover: { title: 'Application Header', description: 'This section gives you top-level navigation and state.', side: "bottom", align: 'start' } },
            { element: '[data-testid="stMarkdownContainer"] h2', popover: { title: 'Data Audit', description: 'Here we analyze class imbalance and data distributions before modeling.', side: "top", align: 'start' } }
        ]"""
    elif page_name == "Model Analytics":
        steps_js = """[
            { element: '[data-testid="stSidebar"]', popover: { title: 'Navigation Sidebar', description: 'Access other tools from here.', side: "right", align: 'start' } },
            { element: 'div[data-testid="stTabs"]', popover: { title: 'Experiments & Analytics', description: 'Switch between full feature experiments, no-leakage models, SMOTE insights, and Optuna tuning.', side: "bottom", align: 'start' } }
        ]"""
    elif page_name == "Interpretability":
        steps_js = """[
            { element: '[data-testid="stSidebar"]', popover: { title: 'Navigation Sidebar', description: 'Global settings apply to all interpretability tools.', side: "right", align: 'start' } },
            { element: 'div[data-testid="stTabs"]', popover: { title: 'Interpretability Tools', description: 'Tune clinical decision thresholds, view global SHAP explanations, and analyze error patterns.', side: "bottom", align: 'start' } }
        ]"""
    elif page_name == "Clinical Tools":
        steps_js = """[
            { element: '[data-testid="stSidebar"]', popover: { title: 'Navigation Sidebar', description: 'Return to other dashboard sections.', side: "right", align: 'start' } },
            { element: 'div[data-testid="stTabs"]', popover: { title: 'Clinical Interfaces', description: 'Use tools like Precision Patient Diagnosis, Batch Prediction, and Risk Timelines.', side: "bottom", align: 'start' } }
        ]"""
    else:
        steps_js = "[]"

    components.html(f"""
    <script>
    if (!window.parent.document.getElementById('driver-js-script')) {{
        var link = window.parent.document.createElement('link');
        link.id = 'driver-js-css';
        link.rel = 'stylesheet';
        link.href = 'https://cdn.jsdelivr.net/npm/driver.js@1.0.1/dist/driver.css';
        window.parent.document.head.appendChild(link);
        
        var script = window.parent.document.createElement('script');
        script.id = 'driver-js-script';
        script.src = 'https://cdn.jsdelivr.net/npm/driver.js@1.0.1/dist/driver.js.iife.js';
        script.onload = function() {{
            startTour();
        }};
        window.parent.document.head.appendChild(script);
    }} else {{
        startTour();
    }}
    
    function startTour() {{
        const driver = window.parent.driver.js.driver;
        const driverObj = driver({{
          showProgress: true,
          steps: {steps_js}
        }});
        driverObj.drive();
    }}
    </script>
    """, height=0, width=0)

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
        
        # Premium animated gradient background
        bg_gradient = "linear-gradient(-45deg, #0B1120, #1e1b4b, #0f172a, #171033)"
        card_bg = "rgba(15, 20, 35, 0.55)"
        border_color = "rgba(255, 255, 255, 0.15)"
        shadow = "0 8px 32px 0 rgba(0, 0, 0, 0.4)"
        text_color = COLORS["text"]
        accent_color = COLORS["primary"]

    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

        html, body, [class*="css"], .stApp {{
            font-family: 'Outfit', sans-serif !important;
        }}

        .stApp {{ 
            background: {bg_gradient}; 
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            color: {text_color}; 
        }}
        
        @keyframes gradientBG {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        @keyframes pulseGlow {{
            0% {{ box-shadow: 0 0 0 0 rgba(139, 92, 246, 0.6); }}
            70% {{ box-shadow: 0 0 20px 15px rgba(139, 92, 246, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(139, 92, 246, 0); }}
        }}

        /* Enhanced Glass Cards with Tilt Support */
        .metric-card, .glass-card, .kpi-box, .tilt-card {{ 
            background: {card_bg}; 
            backdrop-filter: blur(25px) saturate(250%);
            -webkit-backdrop-filter: blur(25px) saturate(250%);
            color: {text_color} !important; 
            border-radius: 24px; 
            border: 1px solid {border_color};
            box-shadow: {shadow};
            transition: border-color 0.4s, box-shadow 0.4s;
            animation: fadeIn 0.8s ease-out forwards;
            position: relative;
            overflow: hidden;
            /* Inner glow for glassmorphism */
            box-shadow: inset 0 0 20px rgba(255, 255, 255, 0.05), {shadow};
        }}
        
        /* Premium glowing border on hover */
        .metric-card:hover, .glass-card:hover, .kpi-box:hover, .tilt-card:hover {{
            border-color: rgba(255, 255, 255, 0.4);
            box-shadow: inset 0 0 20px rgba(255, 255, 255, 0.1), 0 15px 35px rgba(0, 0, 0, 0.4), 0 0 25px rgba(139, 92, 246, 0.3);
        }}

        .metric-card {{ padding: 28px; margin-bottom: 24px; border-left: 6px solid {accent_color}; }}
        .glass-card {{ padding: 28px; margin-bottom: 20px; }}
        
        .metric-card h4, .metric-card p {{ color: {text_color} !important; margin: 0; }}

        .kpi-row {{ display: flex; gap: 24px; margin-bottom: 30px; flex-wrap: wrap; }}
        
        .kpi-box {{
            flex: 1; 
            min-width: 220px;
            padding: 35px 25px; 
            text-align: center;
            /* Ensure vanilla-tilt 3d elements pop out */
            transform-style: preserve-3d; 
        }}

        .kpi-box h2, .animate-number {{ 
            background: linear-gradient(135deg, {COLORS["primary"]}, {COLORS["accent"]});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 15px 0 0 0; 
            font-size: 3.2rem; 
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700; 
            letter-spacing: -2px;
            transform: translateZ(30px); /* 3D pop effect */
        }}
        
        .kpi-box p {{ 
            color: {text_color}; 
            margin: 0; 
            font-size: 1.1rem; 
            font-weight: 600;
            opacity: 0.8; 
            text-transform: uppercase;
            letter-spacing: 2px;
            transform: translateZ(20px); /* 3D pop effect */
        }}
        
        /* Ultra-premium glowing buttons */
        .stButton>button {{
            background: linear-gradient(135deg, {COLORS["primary"]}, {COLORS["secondary"]}) !important;
            color: white !important;
            border: none !important;
            border-radius: 14px !important;
            font-weight: 700 !important;
            letter-spacing: 1px !important;
            padding: 12px 28px !important;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
            position: relative;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3) !important;
            font-family: 'Space Grotesk', sans-serif !important;
        }}
        
        .stButton>button:hover {{
            transform: translateY(-4px) scale(1.05) !important;
            box-shadow: 0 10px 30px rgba(139, 92, 246, 0.6) !important;
            animation: pulseGlow 1.5s infinite;
        }}
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {{
            background: rgba(11, 17, 32, 0.4) !important;
            backdrop-filter: blur(40px);
            border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
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
            text-align: center;
        }}
        </style>
    """, unsafe_allow_html=True)
    
    inject_premium_js()

def inject_premium_js():
    components.html("""
    <script>
    const parentDoc = window.parent.document;
    
    // 1. Particle Background
    if (!parentDoc.getElementById('particles-js-script')) {
        const script = parentDoc.createElement('script');
        script.id = 'particles-js-script';
        script.src = 'https://cdn.jsdelivr.net/npm/particles.js@2.0.0/particles.min.js';
        script.onload = function() {
            if (!parentDoc.getElementById('particles-bg')) {
                const particleDiv = parentDoc.createElement('div');
                particleDiv.id = 'particles-bg';
                particleDiv.style.position = 'fixed';
                particleDiv.style.top = '0';
                particleDiv.style.left = '0';
                particleDiv.style.width = '100vw';
                particleDiv.style.height = '100vh';
                particleDiv.style.zIndex = '-1';
                particleDiv.style.pointerEvents = 'none'; // so it doesn't block clicks
                parentDoc.querySelector('.stApp').prepend(particleDiv);
                
                window.parent.particlesJS('particles-bg', {
                  "particles": {
                    "number": { "value": 60, "density": { "enable": true, "value_area": 800 } },
                    "color": { "value": "#8b5cf6" },
                    "shape": { "type": "circle" },
                    "opacity": { "value": 0.4, "random": true },
                    "size": { "value": 3, "random": true },
                    "line_linked": { "enable": true, "distance": 150, "color": "#3b82f6", "opacity": 0.2, "width": 1 },
                    "move": { "enable": true, "speed": 1.5, "direction": "none", "random": true, "straight": false, "out_mode": "out", "bounce": false }
                  },
                  "interactivity": {
                    "detect_on": "window",
                    "events": {
                      "onhover": { "enable": true, "mode": "grab" },
                      "onclick": { "enable": true, "mode": "push" },
                      "resize": true
                    },
                    "modes": { "grab": { "distance": 180, "line_linked": { "opacity": 0.6 } }, "push": { "particles_nb": 4 } }
                  },
                  "retina_detect": true
                });
            }
        };
        parentDoc.head.appendChild(script);
    }

    // 2. Vanilla Tilt & Animated Numbers
    if (!parentDoc.getElementById('vanilla-tilt-script')) {
        const tiltScript = parentDoc.createElement('script');
        tiltScript.id = 'vanilla-tilt-script';
        tiltScript.src = 'https://cdnjs.cloudflare.com/ajax/libs/vanilla-tilt/1.8.1/vanilla-tilt.min.js';
        parentDoc.head.appendChild(tiltScript);
    }
    
    const applyPremiumEffects = () => {
        // Vanilla Tilt
        if (window.parent.VanillaTilt) {
            const cards = parentDoc.querySelectorAll('.tilt-card:not(.tilt-applied), .kpi-box:not(.tilt-applied), .metric-card:not(.tilt-applied), .glass-card:not(.tilt-applied)');
            if (cards.length > 0) {
                window.parent.VanillaTilt.init(cards, {
                    max: 12, speed: 400, glare: true, "max-glare": 0.2, scale: 1.03
                });
                cards.forEach(c => c.classList.add('tilt-applied'));
            }
        }
        
        // Number Animation
        const numbers = parentDoc.querySelectorAll('.animate-number:not(.anim-applied)');
        numbers.forEach(el => {
            el.classList.add('anim-applied');
            const target = parseFloat(el.getAttribute('data-target'));
            if (isNaN(target)) return;
            const isPercent = el.getAttribute('data-percent') === 'true';
            const isDecimal = el.getAttribute('data-decimal') === 'true';
            const duration = 2000;
            
            let start = null;
            const updateNumber = (currentTime) => {
                if (!start) start = currentTime;
                const elapsed = currentTime - start;
                const progress = Math.min(elapsed / duration, 1);
                const easeProgress = 1 - Math.pow(1 - progress, 4);
                const current = target * easeProgress;
                
                let display = current;
                if (isPercent) display = current.toFixed(2) + '%';
                else if (isDecimal) display = current.toFixed(4);
                else display = Math.floor(current).toLocaleString();
                
                el.innerText = display;
                
                if (progress < 1) {
                    requestAnimationFrame(updateNumber);
                } else {
                    if (isPercent) el.innerText = target.toFixed(2) + '%';
                    else if (isDecimal) el.innerText = target.toFixed(4);
                    else el.innerText = target.toLocaleString();
                }
            };
            requestAnimationFrame(updateNumber);
        });
    };

    // Run immediately for already rendered elements
    applyPremiumEffects();
    
    // Setup global MutationObserver
    if (!window.parent.ckdObserver) {
        window.parent.ckdObserver = new MutationObserver(() => {
            applyPremiumEffects();
        });
        if (parentDoc.body) {
            window.parent.ckdObserver.observe(parentDoc.body, { childList: true, subtree: true });
        } else {
            parentDoc.addEventListener('DOMContentLoaded', () => {
                window.parent.ckdObserver.observe(parentDoc.body, { childList: true, subtree: true });
            });
        }
    }
    </script>
    """, height=0, width=0)

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

def setup_sidebar(page_name="Home"):
    with st.sidebar:
        st.title("🧬 CKD Intelligence")
        st.image("https://cdn-icons-png.flaticon.com/512/3067/3067451.png", width=80)
        sample_size = st.slider("Stratified Sample Size", 1000, 10000, 5000, key="global_sample")
        use_cv = st.checkbox("Enable Cross-Validation (Slower)", value=False, key="global_cv")
        st.divider()
        st.markdown("[🔗 View Source on GitHub](https://github.com/prakash14789/Chronic-Kidney-Disease)")
        st.success("v3.3 Research Pipeline Active")
        st.divider()
        if st.button("🧭 Take a Tour", use_container_width=True):
            render_guided_tour(page_name)
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
