import time
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class CKDModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        
    def get_pipeline(self, model_name):
        """Creates a leakage-free pipeline with scaling and the model."""
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=self.random_state),
            "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=self.random_state),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=self.random_state),
        }
        
        if model_name not in models:
            raise ValueError(f"Model {model_name} not supported.")
            
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', models[model_name])
        ])

    def evaluate_all(self, X_train, y_train, X_test, y_test):
        """Trains and evaluates all models using the Pipeline approach."""
        model_names = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
        results = []
        trained_pipelines = {}

        for name in model_names:
            pipeline = self.get_pipeline(name)
            
            t0 = time.time()
            pipeline.fit(X_train, y_train)
            elapsed = time.time() - t0
            
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            
            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1-Score": f1_score(y_test, y_pred, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, y_proba),
                "Time (s)": round(elapsed, 3)
            })
            trained_pipelines[name] = pipeline
            
        return pd.DataFrame(results).sort_values("Accuracy", ascending=False), trained_pipelines

    @staticmethod
    def run_sanity_check(pipeline, X_test, y_test):
        """Shuffles target labels to verify the performance drops (Requirement #8)."""
        y_test_shuffled = np.random.permutation(y_test)
        y_pred = pipeline.predict(X_test)
        acc_shuffled = accuracy_score(y_test_shuffled, y_pred)
        return acc_shuffled
