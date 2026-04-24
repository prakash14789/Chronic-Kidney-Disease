import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve

# Try optional classifiers
try: from xgboost import XGBClassifier
except: XGBClassifier = None

try: from lightgbm import LGBMClassifier
except: LGBMClassifier = None

class CKDModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = self._initialize_models()
        self.scaled_models = ["Logistic Regression", "SVM", "KNN"]

    def _initialize_models(self):
        models = {
            "Logistic Regression": LogisticRegression(max_iter=500, random_state=self.random_state),
            "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=self.random_state),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=self.random_state),
            "SVM": SVC(probability=True, kernel="rbf", random_state=self.random_state),
            "KNN": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
            "Naive Bayes": GaussianNB(),
            "Extra Trees": ExtraTreesClassifier(n_estimators=100, max_depth=6, random_state=self.random_state, n_jobs=-1),
        }
        if XGBClassifier:
            models["XGBoost"] = XGBClassifier(n_estimators=100, max_depth=6, eval_metric="logloss", random_state=self.random_state)
        if LGBMClassifier:
            models["LightGBM"] = LGBMClassifier(n_estimators=100, max_depth=6, random_state=self.random_state, verbose=-1)
        return models

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, scaler):
        """Trains all models and returns results and ROC data."""
        results = []
        roc_data = {}
        trained_instances = {}
        
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        
        for name, model in self.models.items():
            is_scaled = name in self.scaled_models
            Xtr = X_train_sc if is_scaled else X_train
            Xte = X_test_sc if is_scaled else X_test
            
            t0 = time.time()
            model.fit(Xtr, y_train)
            elapsed = time.time() - t0
            
            y_pred = model.predict(Xte)
            y_proba = model.predict_proba(Xte)[:, 1]
            
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            
            trained_instances[name] = model
            roc_data[name] = {"fpr": fpr, "tpr": tpr, "auc": auc}
            
            results.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1-Score": f1_score(y_test, y_pred, zero_division=0),
                "ROC-AUC": auc,
                "Time (s)": elapsed
            })
            
        return pd.DataFrame(results).sort_values("Accuracy", ascending=False), roc_data, trained_instances
