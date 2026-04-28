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
        colors = ["#4CC9F0", "#F72585", "#4361EE", "#B8F2E6", "#7209B7", 
                  "#FF9F1C", "#FF6B6B", "#E9FF70", "#B5179E", "#4895EF"]
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
        colors = ["#4CC9F0", "#F72585", "#4361EE", "#B8F2E6", "#7209B7", 
                  "#FF9F1C", "#FF6B6B", "#E9FF70", "#B5179E", "#4895EF"]
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
                           histnorm="percent", opacity=0.6,
                           color_discrete_map={0: COLORS["non_ckd"], 1: COLORS["ckd"]},
                           marginal="box", title="Age Distribution by CKD Status (Normalized)")
        
        fig.update_layout(
            yaxis_title="Percent of Class",
            xaxis_title="Patient Age",
            legend_title="Diagnosis",
            bargap=0.05
        )
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
    def _fix_matplotlib_colors(fig):
        plt.rcParams.update({
            'text.color': COLORS["text"],
            'axes.labelcolor': COLORS["text"],
            'xtick.color': COLORS["text"],
            'ytick.color': COLORS["text"],
            'axes.edgecolor': COLORS["grid"]
        })
        for ax in fig.get_axes():
            ax.set_facecolor(COLORS["bg"])
            ax.tick_params(axis='both', which='both', colors=COLORS["text"], labelsize=10)
            ax.xaxis.label.set_color(COLORS["text"])
            ax.yaxis.label.set_color(COLORS["text"])
            ax.title.set_color(COLORS["text"])
            # Fix for SHAP specifically: find all text objects
            for text in ax.get_children():
                if isinstance(text, plt.Text):
                    text.set_color(COLORS["text"])
        return fig

    @staticmethod
    def plot_shap_summary(explainer, shap_values, X_df, model_name):
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(10, 6))
        fig.patch.set_facecolor(COLORS["bg"])
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        shap.summary_plot(sv, X_df, show=False, plot_type="dot")
        plt.title(f"SHAP Summary — {model_name}", color=COLORS["text"], pad=25, fontsize=14)
        return CKDVisualizer._fix_matplotlib_colors(fig)

    @staticmethod
    def plot_shap_bar(explainer, shap_values, X_df, model_name):
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(10, 6))
        fig.patch.set_facecolor(COLORS["bg"])
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        shap.summary_plot(sv, X_df, show=False, plot_type="bar")
        plt.title(f"SHAP Feature Importance — {model_name}", color=COLORS["text"], pad=25, fontsize=14)
        return CKDVisualizer._fix_matplotlib_colors(fig)

    @staticmethod
    def plot_local_shap(explainer, shap_values, X_df, patient_idx=0):
        plt.style.use('dark_background')
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
        plt.title("Individual Risk Factors", color=COLORS["text"], pad=25, fontsize=14)
        return CKDVisualizer._fix_matplotlib_colors(fig)

    @staticmethod
    def plot_feature_direction(df):
        """CKD vs Non-CKD mean feature differences."""
        ckd = df[df["Diagnosis"] == 1].select_dtypes(include=[np.number]).mean()
        non_ckd = df[df["Diagnosis"] == 0].select_dtypes(include=[np.number]).mean()
        diff = (ckd - non_ckd).drop("Diagnosis", errors="ignore").sort_values()
        top = pd.concat([diff.head(8), diff.tail(8)])
        colors = [COLORS["non_ckd"] if v < 0 else COLORS["ckd"] for v in top.values]
        fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation='h', marker_color=colors))
        fig.update_layout(title="Feature Direction: CKD vs Non-CKD (Mean Difference)",
                          xaxis_title="Mean Difference (CKD − Non-CKD)")
        return CKDVisualizer._apply_dark_theme(fig)

    @staticmethod
    def plot_grouped_shap(group_importance):
        """Grouped SHAP contribution as horizontal bar %."""
        names = list(group_importance.keys())
        values = list(group_importance.values())
        total = sum(values) or 1
        pcts = [v / total * 100 for v in values]
        colors = ["#4CC9F0", "#F72585", "#4361EE", "#B8F2E6", "#7209B7", "#FF9F1C"]
        fig = go.Figure(go.Bar(x=pcts, y=names, orientation='h',
                               marker_color=colors[:len(names)],
                               text=[f"{p:.1f}%" for p in pcts], textposition='auto'))
        fig.update_layout(title="Risk Factor Group Contribution (%)", xaxis_title="Contribution %")
        return CKDVisualizer._apply_dark_theme(fig)

    @staticmethod
    def plot_calibration(y_true, y_proba):
        """Calibration curve: predicted vs actual probability."""
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode='lines+markers', name='Model',
                                 line=dict(color=COLORS["primary"], width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfect',
                                 line=dict(dash='dash', color=COLORS["text"])))
        fig.update_layout(title="Calibration Curve — Is Your Probability Trustworthy?",
                          xaxis_title="Mean Predicted Probability", yaxis_title="Fraction of Positives")
        return CKDVisualizer._apply_dark_theme(fig)

    @staticmethod
    def plot_population_risk(y_proba, patient_prob=None):
        """Risk distribution histogram with optional patient marker."""
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=y_proba, nbinsx=30, marker_color=COLORS["primary"],
                                    opacity=0.7, name="Population"))
        if patient_prob is not None:
            fig.add_vline(x=patient_prob, line_dash="dash", line_color=COLORS["ckd"], line_width=3,
                          annotation_text=f"This Patient: {patient_prob:.1%}")
        fig.update_layout(title="Population Risk Distribution",
                          xaxis_title="Predicted CKD Probability", yaxis_title="Count")
        return CKDVisualizer._apply_dark_theme(fig)

    @staticmethod
    def plot_counterfactual(base_prob, results):
        """What-if analysis: risk change per intervention."""
        sorted_r = dict(sorted(results.items(), key=lambda x: x[1]))
        names = list(sorted_r.keys())
        vals = [v * 100 for v in sorted_r.values()]
        colors = [COLORS["non_ckd"] if v < 0 else COLORS["ckd"] for v in vals]
        fig = go.Figure(go.Bar(x=vals, y=names, orientation='h', marker_color=colors,
                               text=[f"{v:+.2f}%" for v in vals], textposition='auto'))
        fig.update_layout(title=f"What-If Analysis (Base Risk: {base_prob:.1%})",
                          xaxis_title="Risk Change (%)")
        return CKDVisualizer._apply_dark_theme(fig)

    @staticmethod
    def plot_risk_gauge(prob):
        """Gauge meter for patient risk score."""
        color = "#4D96FF" if prob < 0.3 else "#FFD93D" if prob < 0.7 else "#FF6B6B"
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100,
            title={"text": "CKD Risk Score", "font": {"color": COLORS["text"]}},
            number={"suffix": "%", "font": {"color": color}},
            gauge={"axis": {"range": [0, 100], "tickcolor": COLORS["text"]},
                   "bar": {"color": color}, "bgcolor": COLORS["grid"],
                   "steps": [{"range": [0, 30], "color": "rgba(77,150,255,0.2)"},
                             {"range": [30, 70], "color": "rgba(255,217,61,0.2)"},
                             {"range": [70, 100], "color": "rgba(255,107,107,0.2)"}],
                   "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.8,
                                 "value": prob * 100}}))
        fig.update_layout(paper_bgcolor=COLORS["bg"], font={"color": COLORS["text"]}, height=300)
        return fig

    @staticmethod
    def plot_confusion_matrix(counts):
        """Interactive confusion matrix heatmap."""
        labels = ["Non-CKD (0)", "CKD (1)"]
        z = [[counts["TN"], counts["FP"]], [counts["FN"], counts["TP"]]]
        text = [[f"TN: {z[0][0]}", f"FP: {z[0][1]}"], [f"FN: {z[1][0]}", f"TP: {z[1][1]}"]]
        fig = go.Figure(data=go.Heatmap(z=z, x=labels, y=labels, text=text,
                                         texttemplate="%{text}", showscale=False,
                                         colorscale=[[0, COLORS["grid"]], [1, COLORS["primary"]]]))
        fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
        return CKDVisualizer._apply_dark_theme(fig)

    @staticmethod
    def plot_model_stability(scores):
        """Box plot of accuracy across multiple splits."""
        fig = go.Figure()
        fig.add_trace(go.Box(y=scores, name="Balanced Accuracy", marker_color=COLORS["primary"],
                             boxpoints='all', jitter=0.3))
        fig.add_hline(y=np.mean(scores), line_dash="dash", line_color=COLORS["f1"],
                      annotation_text=f"Mean: {np.mean(scores):.4f}")
        fig.update_layout(title=f"Model Stability ({len(scores)} Splits)", yaxis_title="Balanced Accuracy")
        return CKDVisualizer._apply_dark_theme(fig)

    @staticmethod
    def plot_protective_factors(shap_values, feature_names, patient_idx=0):
        """Split view: protective factors vs risk factors for a patient."""
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        vals = sv.values[patient_idx] if hasattr(sv, 'values') else sv[patient_idx]
        feat_shap = pd.Series(vals, index=feature_names)
        protective = feat_shap[feat_shap < 0].sort_values().head(8)
        risk = feat_shap[feat_shap > 0].sort_values(ascending=False).head(8)
        fig = go.Figure()
        if len(protective) > 0:
            fig.add_trace(go.Bar(y=protective.index, x=protective.values, orientation='h',
                                 name="✅ Protective (↓ Risk)", marker_color=COLORS["non_ckd"]))
        if len(risk) > 0:
            fig.add_trace(go.Bar(y=risk.index, x=risk.values, orientation='h',
                                 name="🚨 Risk (↑ Risk)", marker_color=COLORS["ckd"]))
        fig.update_layout(title="Why CKD / Why NOT CKD", xaxis_title="SHAP Value", barmode='relative')
        return CKDVisualizer._apply_dark_theme(fig)

    @staticmethod
    def plot_error_patterns(fp_data, fn_data, all_data):
        """Compare mean feature values of FP/FN vs population."""
        fig = go.Figure()
        cols = all_data.columns[:10]
        pop_mean = all_data[cols].mean()
        if len(fp_data) > 0:
            fp_diff = (fp_data[cols].mean() - pop_mean) / (pop_mean.abs() + 1e-9) * 100
            fig.add_trace(go.Bar(x=cols, y=fp_diff, name="False Positives", marker_color=COLORS["recall"]))
        if len(fn_data) > 0:
            fn_diff = (fn_data[cols].mean() - pop_mean) / (pop_mean.abs() + 1e-9) * 100
            fig.add_trace(go.Bar(x=cols, y=fn_diff, name="False Negatives", marker_color="#FFD93D"))
        fig.update_layout(title="Error Pattern Analysis (% Deviation from Population Mean)",
                          xaxis_title="Feature", yaxis_title="% Deviation", barmode="group")
        return CKDVisualizer._apply_dark_theme(fig)
