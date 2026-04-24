import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import shap

class CKDVisualizer:
    @staticmethod
    def plot_class_distribution(df, ckd_pct):
        fig = px.pie(df, names='Diagnosis', title=f"Class Imbalance ({ckd_pct:.1f}% CKD)",
                     color_discrete_sequence=['#E74C3C', '#3498DB'])
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig

    @staticmethod
    def plot_misleading_accuracy(ckd_pct):
        dummy_metrics = {
            "Raw Accuracy (Predict All CKD)": ckd_pct / 100,
            "Balanced Accuracy (Predict All CKD)": 0.50,
            "Macro F1 (Predict All CKD)": 0.48,
        }
        fig = px.bar(x=list(dummy_metrics.keys()), y=list(dummy_metrics.values()), 
                     color=list(dummy_metrics.keys()), title="Why Raw Accuracy is Misleading",
                     labels={'x': 'Metric', 'y': 'Score'})
        fig.add_hline(y=0.5, line_dash="dash", annotation_text="Random Guess Level")
        return fig

    @staticmethod
    def plot_roc_curves(roc_data):
        fig = go.Figure()
        for name, data in roc_data.items():
            fig.add_trace(go.Scatter(x=data[0], y=data[1], name=f"{name} (AUC={data[2]:.3f})", mode='lines'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash='dash'), showlegend=False))
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", title="ROC Curves (Leakage-Free)")
        return fig

    @staticmethod
    def plot_pr_curves(pr_data):
        fig = go.Figure()
        for name, data in pr_data.items():
            fig.add_trace(go.Scatter(x=data[1], y=data[0], name=f"{name} (AvgPrec={data[2]:.3f})", mode='lines'))
        fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", title="Precision-Recall Curves")
        return fig

    @staticmethod
    def plot_threshold_tuning(th_df, best_th):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=th_df["Threshold"], y=th_df["Macro F1"], name="Macro F1", mode='lines+markers'))
        fig.add_trace(go.Scatter(x=th_df["Threshold"], y=th_df["Bal-Acc"], name="Balanced Acc", mode='lines+markers'))
        fig.add_vline(x=best_th, line_dash="dot", annotation_text=f"Optimal TH: {best_th}")
        fig.update_layout(xaxis_title="Decision Threshold", yaxis_title="Score", title="Threshold Tuning Optimization")
        return fig

    @staticmethod
    def plot_age_distribution(df):
        fig = px.histogram(df, x="Age", color="Diagnosis", barmode="overlay",
                           color_discrete_map={0: '#3498DB', 1: '#E74C3C'},
                           marginal="box", title="Age Distribution by CKD Status")
        return fig

    @staticmethod
    def plot_precision_recall_f1(res_df):
        # We need to ensure these metrics exist in res_df
        # If not already there, we might need model_trainer to return them
        metrics = ["Macro Precision", "Macro Recall", "Macro F1"]
        fig = px.bar(res_df, x="Model", y=metrics, barmode="group",
                     title="Precision / Recall / F1-Score by Model",
                     color_discrete_sequence=['#4C72B0', '#DD8452', '#55A868'])
        fig.update_layout(yaxis_range=[0.5, 1.05], yaxis_title="Score")
        return fig

    @staticmethod
    def plot_shap_summary(explainer, shap_values, X_df, model_name):
        plt.figure(figsize=(10, 6))
        # Handle cases where shap_values might be a list (multiclass or multiclass-like)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values
        
        shap.summary_plot(sv, X_df, show=False, plot_type="dot")
        plt.title(f"SHAP Summary — {model_name}")
        plt.tight_layout()
        return plt.gcf()

    @staticmethod
    def plot_shap_bar(explainer, shap_values, X_df, model_name):
        plt.figure(figsize=(10, 6))
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values
            
        shap.summary_plot(sv, X_df, show=False, plot_type="bar")
        plt.title(f"SHAP Feature Importance — {model_name}")
        plt.tight_layout()
        return plt.gcf()
