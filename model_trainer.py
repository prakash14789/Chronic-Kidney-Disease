import time
import pandas as pd
import numpy as np
import copy
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# Optional high-performance models
try: from xgboost import XGBClassifier
except: XGBClassifier = None

try: from lightgbm import LGBMClassifier
except: LGBMClassifier = None

class CKDModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        # EXACT model definitions from your source code
        self.base_models = {
            "Logistic Regression" : LogisticRegression(max_iter=500, random_state=self.random_state),
            "Decision Tree"       : DecisionTreeClassifier(max_depth=8, random_state=self.random_state),
            "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1),
            "Gradient Boosting"   : GradientBoostingClassifier(n_estimators=100, random_state=self.random_state),
            "SVM"                 : SVC(probability=True, kernel="rbf", random_state=self.random_state),
            "KNN"                 : KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
            "Naive Bayes"         : GaussianNB(),
            "Extra Trees"         : ExtraTreesClassifier(n_estimators=100, max_depth=6, random_state=self.random_state, n_jobs=-1),
        }
        
        if XGBClassifier:
            self.base_models["XGBoost"] = XGBClassifier(
                n_estimators=100, max_depth=6, eval_metric="logloss",
                random_state=self.random_state, n_jobs=-1
            )
        if LGBMClassifier:
            self.base_models["LightGBM"] = LGBMClassifier(
                n_estimators=100, max_depth=6,
                random_state=self.random_state, n_jobs=-1, verbose=-1
            )

    def run_experiment(self, X_tr, X_te, y_tr, y_te, label=""):
        """
        Trains and evaluates all models exactly like your 'run_experiment' function.
        """
        results = []
        roc_data = {}
        trained_pipelines = {}
        
        # Models that need scaling (as defined in your code)
        scaled_models_set = {"Logistic Regression", "SVM", "KNN"}

        for name, model_obj in self.base_models.items():
            model = copy.deepcopy(model_obj)
            
            # Use Pipeline to handle scaling strictly for the required models
            if name in scaled_models_set:
                pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', model)])
            else:
                pipeline = Pipeline([('classifier', model)])
                
            t0 = time.time()
            pipeline.fit(X_tr, y_tr)
            elapsed = time.time() - t0
            
            y_pred = pipeline.predict(X_te)
            y_proba = pipeline.predict_proba(X_te)[:, 1]
            
            acc = accuracy_score(y_te, y_pred)
            auc = roc_auc_score(y_te, y_proba)
            fpr, tpr, _ = roc_curve(y_te, y_proba)
            
            results.append({
                "Model"          : name,
                "Test Accuracy"  : round(acc,  4),
                "Precision"      : round(precision_score(y_te, y_pred, zero_division=0), 4),
                "Recall"         : round(recall_score(y_te, y_pred, zero_division=0), 4),
                "F1-Score"       : round(f1_score(y_te, y_pred, zero_division=0), 4),
                "ROC-AUC"        : round(auc,  4),
                "Train Time (s)" : round(elapsed, 2),
            })
            roc_data[name] = {"fpr": fpr, "tpr": tpr, "auc": auc}
            trained_pipelines[name] = pipeline
            
        results_df = pd.DataFrame(results).sort_values("Test Accuracy", ascending=False)
        return results_df, roc_data, trained_pipelines
