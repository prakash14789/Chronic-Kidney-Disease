import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

class CKDVisualizer:
    @staticmethod
    def plot_class_distribution(df):
        fig = px.pie(df, names='Diagnosis', title='CKD Prevalence',
                     color_discrete_sequence=['#4C72B0', '#DD8452'],
                     labels={'Diagnosis': 'Status'})
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig

    @staticmethod
    def plot_age_distribution(df):
        fig = px.histogram(df, x="Age", color="Diagnosis", barmode="overlay",
                           color_discrete_map={0: 'steelblue', 1: 'tomato'},
                           marginal="box", title="Age Distribution by Status")
        return fig

    @staticmethod
    def plot_correlation_heatmap(df):
        top15_corr = df.corr()['Diagnosis'].abs().sort_values(ascending=False).head(16).index
        corr_matrix = df[top15_corr].corr()
        fig = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='RdBu_r', 
                         title="Top 15 Feature Correlations")
        return fig

    @staticmethod
    def plot_model_comparison(results_df):
        fig = px.bar(results_df, x="Model", y="Accuracy", color="Accuracy",
                     color_continuous_scale="Viridis", title="Model Accuracy Benchmarking")
        return fig

    @staticmethod
    def plot_roc_curves(roc_data):
        fig = go.Figure()
        for name, data in roc_data.items():
            fig.add_trace(go.Scatter(x=data["fpr"], y=data["tpr"], 
                                     name=f"{name} (AUC={data['auc']:.3f})", mode='lines'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash='dash'), showlegend=False))
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                          title="Receiver Operating Characteristic (ROC) Curves")
        return fig

    @staticmethod
    def plot_feature_importance(model, feature_names):
        if hasattr(model, "feature_importances_"):
            feat_imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
            fig = px.bar(feat_imp, orientation='h', color=feat_imp.values, color_continuous_scale="Blues",
                         title="Top 15 Predictive Features")
            fig.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
            return fig
        return None
