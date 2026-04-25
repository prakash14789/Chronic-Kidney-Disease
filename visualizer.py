import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np

COLORS = {
    "bg": "#0E1117",
    "grid": "#2A2E39",
    "text": "#E6EDF3",

    # Metrics
    "precision": "#4CC9F0",
    "recall": "#F72585",
    "f1": "#B8F2E6",

    # Models
    "primary": "#4361EE",
    "secondary": "#3A0CA3",
    "accent": "#7209B7",

    # CKD
    "ckd": "#FF6B6B",
    "non_ckd": "#4D96FF"
}

class CKDVisualizer:
    @staticmethod
    def _apply_dark_theme(fig):
        fig.update_layout(
            plot_bgcolor=COLORS["bg"],
            paper_bgcolor=COLORS["bg"],
            font=dict(color=COLORS["text"]),
            xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
            yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        return fig

    @staticmethod
    def plot_class_distribution(df, ckd_pct):
        fig = px.pie(df, names='Diagnosis', title=f"Class Imbalance ({ckd_pct:.1f}% CKD)",
                     color='Diagnosis',
                     color_discrete_map={1: COLORS["ckd"], 0: COLORS["non_ckd"]})
        fig.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color=COLORS["bg"], width=2)))
        return CKDVisualizer._apply_dark_theme(fig)

    @staticmethod
    def plot_misleading_accuracy(ckd_pct):
        dummy_metrics = {
            "Raw Accuracy": ckd_pct / 100,
            "Balanced Accuracy": 0.50,
            "Macro F1": 0.48,
        }
        fig = px.bar(x=list(dummy_metrics.keys()), y=list(dummy_metrics.values()), 
                     color=list(dummy_metrics.keys()), title="Why Raw Accuracy is Misleading",
                     labels={'x': 'Metric', 'y': 'Score'},
                     color_discrete_map={
                         "Raw Accuracy": COLORS["accent"],
                         "Balanced Accuracy": COLORS["primary"],
                         "Macro F1": COLORS["f1"]
                     })
        fig.add_hline(y=0.5, line_dash="dash", annotation_text="Random Guess Level", line_color=COLORS["text"])
        return CKDVisualizer._apply_dark_theme(fig)

    @staticmethod
    def plot_roc_curves(roc_data):
        fig = go.Figure()
        colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]]
        for i, (name, data) in enumerate(roc_data.items()):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(x=data[0], y=data[1], name=f"{name} (AUC={data[2]:.3f})", 
                                   mode='lines', line=dict(color=color, width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash='dash', color=COLORS["text"]), showlegend=False))
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", title="ROC Curves")
        return CKDVisualizer._apply_dark_theme(fig)

    @staticmethod
    def plot_pr_curves(pr_data):
        fig = go.Figure()
        colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]]
        for i, (name, data) in enumerate(pr_data.items()):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(x=data[1], y=data[0], name=f"{name} (AvgPrec={data[2]:.3f})", 
                                   mode='lines', line=dict(color=color, width=3)))
        fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", title="Precision-Recall Curves")
        return CKDVisualizer._apply_dark_theme(fig)

    @staticmethod
    def plot_threshold_tuning(th_df, best_th):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=th_df["Threshold"], y=th_df["Macro F1"], name="Macro F1", 
                               mode='lines+markers', line=dict(color=COLORS["f1"])))
        fig.add_trace(go.Scatter(x=th_df["Threshold"], y=th_df["Bal-Acc"], name="Balanced Acc", 
                               mode='lines+markers', line=dict(color=COLORS["primary"])))
        fig.add_vline(x=best_th, line_dash="dot", annotation_text=f"Optimal TH: {best_th}", line_color=COLORS["recall"])
        fig.update_layout(xaxis_title="Decision Threshold", yaxis_title="Score", title="Threshold Tuning Optimization")
        return CKDVisualizer._apply_dark_theme(fig)

    @staticmethod
    def plot_age_distribution(df):
        fig = px.histogram(df, x="Age", color="Diagnosis", barmode="overlay",
                           color_discrete_map={0: COLORS["non_ckd"], 1: COLORS["ckd"]},
                           marginal="box", title="Age Distribution by CKD Status")
        return CKDVisualizer._apply_dark_theme(fig)

    @staticmethod
    def plot_precision_recall_f1(res_df):
        # Melt dataframe for easier plotting
        melt_df = res_df.melt(id_vars=["Model"], value_vars=["Macro Precision", "Macro Recall", "Macro F1"], 
                               var_name="Metric", value_name="Score")
        
        fig = px.bar(melt_df, x="Model", y="Score", color="Metric", barmode="group",
                     title="Precision / Recall / F1-Score by Model",
                     color_discrete_map={
                         "Macro Precision": COLORS["precision"],
                         "Macro Recall": COLORS["recall"],
                         "Macro F1": COLORS["f1"]
                     })
        fig.update_layout(yaxis_range=[0.5, 1.05], yaxis_title="Score")
        return CKDVisualizer._apply_dark_theme(fig)

    @staticmethod
    def plot_correlation_heatmap(df):
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor(COLORS["bg"])
        ax.set_facecolor(COLORS["bg"])
        
        numeric_df = df.select_dtypes(include=[np.number])
        top_cols = numeric_df.var().nlargest(20).index
        corr = numeric_df[top_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="vlag", annot_kws={"size": 7}, ax=ax)
        plt.title("Correlation Heatmap (Top 20 Features)", color=COLORS["text"])
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_clinical_boxplots(df):
        clinical = ["GFR", "SerumCreatinine", "BUNLevels", "HbA1c", "ProteinInUrine"]
        fig = px.box(df, x="Diagnosis", y=[c for c in clinical if c in df.columns],
                     facet_col="variable", facet_col_wrap=3,
                     color="Diagnosis", title="Clinical Feature Distributions",
                     color_discrete_map={0: COLORS["non_ckd"], 1: COLORS["ckd"]})
        fig.update_yaxes(matches=None)
        return CKDVisualizer._apply_dark_theme(fig)

    @staticmethod
    def plot_shap_summary(explainer, shap_values, X_df, model_name):
        plt.style.use('dark_background')
        plt.rcParams.update({
            'text.color': COLORS["text"],
            'axes.labelcolor': COLORS["text"],
            'xtick.color': COLORS["text"],
            'ytick.color': COLORS["text"]
        })
        fig = plt.figure(figsize=(10, 6))
        fig.patch.set_facecolor(COLORS["bg"])
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        shap.summary_plot(sv, X_df, show=False, plot_type="dot")
        plt.title(f"SHAP Summary — {model_name}", color=COLORS["text"], pad=20)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_shap_bar(explainer, shap_values, X_df, model_name):
        plt.style.use('dark_background')
        plt.rcParams.update({
            'text.color': COLORS["text"],
            'axes.labelcolor': COLORS["text"],
            'xtick.color': COLORS["text"],
            'ytick.color': COLORS["text"]
        })
        fig = plt.figure(figsize=(10, 6))
        fig.patch.set_facecolor(COLORS["bg"])
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        shap.summary_plot(sv, X_df, show=False, plot_type="bar")
        plt.title(f"SHAP Feature Importance — {model_name}", color=COLORS["text"], pad=20)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_local_shap(explainer, shap_values, X_df, patient_idx=0):
        plt.style.use('dark_background')
        plt.rcParams.update({
            'text.color': COLORS["text"],
            'axes.labelcolor': COLORS["text"],
            'xtick.color': COLORS["text"],
            'ytick.color': COLORS["text"]
        })
        fig = plt.figure(figsize=(10, 5))
        fig.patch.set_facecolor(COLORS["bg"])
        if hasattr(shap_values, "base_values"):
            shap.plots.waterfall(shap_values[patient_idx], show=False)
        else:
            if isinstance(shap_values, list):
                sv = shap_values[1][patient_idx]
                base = explainer.expected_value[1]
            else:
                sv = shap_values[patient_idx]
                base = explainer.expected_value
            shap.plots._waterfall.waterfall_legacy(base, sv, feature_names=X_df.columns, show=False)
        plt.title("Individual Risk Factors", color=COLORS["text"], pad=20)
        plt.tight_layout()
        return fig
