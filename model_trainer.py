import time
import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, 
    roc_curve, precision_score, recall_score, cohen_kappa_score, 
    average_precision_score, precision_recall_curve
)

# Imblearn for leakage-free SMOTE
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    from sklearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = False

from sklearn.pipeline import Pipeline as SkPipeline

# Optional high-performance models
try: from xgboost import XGBClassifier
except: XGBClassifier = None

try: from lightgbm import LGBMClassifier
except: LGBMClassifier = None

class CKDModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def build_pipeline(self, clf, needs_scaling=False, use_smote=True):
        """EXACT V3 Pipeline Builder."""
        steps = []
        if use_smote and SMOTE_AVAILABLE:
            steps.append(("smote", SMOTE(sampling_strategy=0.5, random_state=self.random_state)))
            PipelineCls = ImbPipeline
        else:
            PipelineCls = SkPipeline

        if needs_scaling:
            steps.append(("scaler", StandardScaler()))

        steps.append(("clf", clf))
        return PipelineCls(steps)

    def get_v3_pipelines(self, n_neg, n_pos, use_smote=True):
        """Returns the EXACT list of v3 pipelines."""
        base = [
            ("Logistic Regression", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=self.random_state), True),
            ("Decision Tree", DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=self.random_state), False),
            ("Random Forest", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=self.random_state, n_jobs=-1), False),
            ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, subsample=0.8, random_state=self.random_state), False),
            ("SVM", SVC(probability=True, kernel="rbf", class_weight="balanced", random_state=self.random_state), True),
            ("KNN", KNeighborsClassifier(n_neighbors=7, n_jobs=-1), True),
            ("Naive Bayes", GaussianNB(), False),
            ("Extra Trees", ExtraTreesClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=self.random_state, n_jobs=-1), False),
        ]
        
        if XGBClassifier:
            base.append(("XGBoost", XGBClassifier(n_estimators=100, max_depth=6, eval_metric="logloss", 
                                                  scale_pos_weight=n_neg / max(n_pos, 1), random_state=self.random_state), False))
        if LGBMClassifier:
            base.append(("LightGBM", LGBMClassifier(n_estimators=100, max_depth=6, is_unbalance=True, 
                                                    random_state=self.random_state, verbose=-1), False))
        
        return [(name, self.build_pipeline(clf, needs_scaling=sc, use_smote=use_smote)) for name, clf, sc in base]

    def run_v3_experiment(self, X_tr, X_te, y_tr, y_te, pipelines):
        """EXACT V3 Experiment Runner."""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        results, roc_data, pr_data, trained = [], {}, {}, {}

        for name, pipe in pipelines:
            t0 = time.time()
            try:
                cv_scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
                pipe.fit(X_tr, y_tr)
                y_pred = pipe.predict(X_te)
                y_proba = pipe.predict_proba(X_te)[:, 1]
                
                auc = roc_auc_score(y_te, y_proba)
                fpr, tpr, _ = roc_curve(y_te, y_proba)
                roc_data[name] = (fpr, tpr, auc)
                precision, recall, _ = precision_recall_curve(y_te, y_proba)
                pr_data[name] = (precision, recall, average_precision_score(y_te, y_proba))
                trained[name] = pipe

                results.append({
                    "Model": name,
                    "Accuracy": round(accuracy_score(y_te, y_pred), 4),
                    "Balanced Accuracy": round(balanced_accuracy_score(y_te, y_pred), 4),
                    "Macro Precision": round(precision_score(y_te, y_pred, average="macro", zero_division=0), 4),
                    "Macro Recall": round(recall_score(y_te, y_pred, average="macro", zero_division=0), 4),
                    "Macro F1": round(f1_score(y_te, y_pred, average="macro", zero_division=0), 4),
                    "ROC-AUC": round(auc, 4),
                    "Cohen Kappa": round(cohen_kappa_score(y_te, y_pred), 4),
                    "CV BalAcc Mean": round(cv_scores.mean(), 4),
                    "Train Time (s)": round(time.time() - t0, 2),
                })
            except Exception as e:
                import streamlit as st
                st.warning(f"⚠️ Skipped {name}: {e}")

        if not results:
            return pd.DataFrame(), roc_data, pr_data, trained

        df_res = (pd.DataFrame(results)
                    .sort_values("Balanced Accuracy", ascending=False)
                    .reset_index(drop=True))
        return df_res, roc_data, pr_data, trained

    @staticmethod
    def run_sanity_check(pipeline, X_test, y_test):
        """Shuffles target labels to verify the performance drops (Requirement #8)."""
        y_test_shuffled = np.random.permutation(y_test)
        y_pred = pipeline.predict(X_test)
        acc_shuffled = balanced_accuracy_score(y_test_shuffled, y_pred)
        return acc_shuffled

    def tune_threshold(self, y_true, y_proba):
        """EXACT V3 Threshold Tuning Logic."""
        thresholds = np.arange(0.05, 0.95, 0.05)
        rows = []
        for th in thresholds:
            yp = (y_proba >= th).astype(int)
            rows.append({
                "Threshold": round(th, 2),
                "Bal-Acc": round(balanced_accuracy_score(y_true, yp), 4),
                "Macro F1": round(f1_score(y_true, yp, average="macro", zero_division=0), 4),
            })
        df = pd.DataFrame(rows)
        best_th = df.loc[df["Macro F1"].idxmax(), "Threshold"]
        return df, best_th

    def get_shap_explainer(self, model, X_test):
        """Generates SHAP values for the best model."""
        import shap
        # Extract clf and transform X if needed
        clf = model.named_steps['clf']
        if 'scaler' in model.named_steps:
            X_trans = model.named_steps['scaler'].transform(X_test)
        else:
            X_trans = X_test.values
            
        X_df = pd.DataFrame(X_trans, columns=X_test.columns)
        
        try:
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_df)
            return explainer, shap_values, X_df
        except:
            # Fallback to KernelExplainer if TreeExplainer fails
            explainer = shap.Explainer(clf, X_df)
            shap_values = explainer(X_df)
            return explainer, shap_values, X_df
