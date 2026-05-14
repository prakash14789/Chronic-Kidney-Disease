import time
import pandas as pd
import numpy as np
import copy
from typing import Any, List, Tuple, Dict
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
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def build_pipeline(self, clf: Any, needs_scaling: bool = False, use_smote: bool = True) -> Any:
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

    def get_v3_pipelines(self, n_neg: int, n_pos: int, use_smote: bool = True) -> List[Tuple[str, Any]]:
        """Returns the EXACT list of v3 pipelines."""
        base = [
            ("Logistic Regression", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=self.random_state), True),
            ("Decision Tree", DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=self.random_state), False),
            ("Random Forest", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=self.random_state, n_jobs=-1), False),
            ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, subsample=0.8, random_state=self.random_state), False),
            ("SVM", SVC(probability=True, kernel="rbf", class_weight="balanced", random_state=self.random_state, max_iter=2000), True),
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

    def run_v3_experiment(self, X_tr: pd.DataFrame, X_te: pd.DataFrame, y_tr: pd.Series, y_te: pd.Series, pipelines: List[Tuple[str, Any]], use_cv: bool = False) -> Tuple[pd.DataFrame, Dict, Dict, Dict]:
        """EXACT V3 Experiment Runner optimized for speed."""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        results, roc_data, pr_data, trained = [], {}, {}, {}

        for name, pipe in pipelines:
            t0 = time.time()
            try:
                cv_mean = 0.0
                if use_cv:
                    cv_scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
                    cv_mean = cv_scores.mean()
                
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
                    "CV BalAcc Mean": round(cv_mean, 4) if use_cv else "Skipped",
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
    def run_sanity_check(pipeline: Any, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        """Shuffles target labels to verify the performance drops (Requirement #8)."""
        y_test_shuffled = np.random.permutation(y_test)
        y_pred = pipeline.predict(X_test)
        acc_shuffled = balanced_accuracy_score(y_test_shuffled, y_pred)
        return acc_shuffled

    def tune_threshold(self, y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[pd.DataFrame, float]:
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

    def get_clinical_assessment(self, probability: float) -> Dict[str, str]:
        """Risk Stratification and Clinical Recommendations."""
        if probability < 0.3:
            return {
                "Level": "Low Risk",
                "Color": "#4D96FF",  # Blue
                "Action": "Routine monitoring recommended. Maintain healthy lifestyle habits.",
                "Icon": "✅"
            }
        elif probability < 0.7:
            return {
                "Level": "Moderate Risk",
                "Color": "#FFD93D",  # Yellow
                "Action": "Further diagnostic tests advised. Schedule a follow-up consultation.",
                "Icon": "⚠️"
            }
        else:
            return {
                "Level": "High Risk",
                "Color": "#FF6B6B",  # Red
                "Action": "Immediate medical attention required. Urgent nephrologist referral suggested.",
                "Icon": "🚨"
            }

    def get_shap_explainer(self, model: Any, X_test: pd.DataFrame) -> Tuple[Any, Any, pd.DataFrame]:
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

    def find_similar_patients(self, X_train: pd.DataFrame, y_train: pd.Series, input_row: pd.DataFrame, n: int = 5) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """Find n most similar patients using euclidean distance."""
        from sklearn.metrics.pairwise import euclidean_distances
        dist = euclidean_distances(X_train.values, input_row.values)
        closest_idx = np.argsort(dist.ravel())[:n]
        return X_train.iloc[closest_idx], y_train.iloc[closest_idx], dist.ravel()[closest_idx]

    def compute_counterfactual(self, model: Any, input_row: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        """What-if: change one feature at a time, measure risk change."""
        base_prob = model.predict_proba(input_row)[0, 1]
        mods = {"SystolicBP": -20, "DiastolicBP": -10, "FastingBloodSugar": -20,
                "HbA1c": -1.0, "BMI": -3, "PhysicalActivity": +50,
                "DietQuality": +3, "SleepQuality": +2, "CholesterolTotal": -30, "FatigueLevels": -3}
        results = {}
        for feat, delta in mods.items():
            if feat in input_row.columns:
                temp = input_row.copy()
                temp[feat] = temp[feat] + delta
                results[f"{feat} ({delta:+g})"] = model.predict_proba(temp)[0, 1] - base_prob
        return base_prob, results

    def get_grouped_shap(self, shap_values: Any, feature_names: List[str]) -> Dict[str, float]:
        """Compute average absolute SHAP per feature group."""
        groups = {
            "Lifestyle": ["BMI", "PhysicalActivity", "DietQuality", "SleepQuality", "Smoking"],
            "Clinical": ["SystolicBP", "DiastolicBP", "FastingBloodSugar", "HbA1c", "HemoglobinLevels"],
            "Biochemical": ["SerumElectrolytesSodium", "SerumElectrolytesPotassium", "CholesterolTotal"],
            "Demographics": ["Age", "Gender"],
            "Family History": ["FamilyHistoryKidneyDisease", "FamilyHistoryHypertension", "FamilyHistoryDiabetes"],
            "Other": ["Adherence", "Edema", "FatigueLevels", "QualityOfLifeScore", "HeavyMetalsExposure"]
        }
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        feat_list = list(feature_names)
        result = {}
        for gname, feats in groups.items():
            total = sum(np.abs(sv[:, feat_list.index(f)]).mean() for f in feats if f in feat_list)
            result[gname] = total
        return result

    def run_stability_check_multi(self, X: pd.DataFrame, y: pd.Series, n_runs: int = 5) -> List[float]:
        """Run model across multiple splits to check stability."""
        from sklearn.model_selection import StratifiedShuffleSplit
        scores = []
        sss = StratifiedShuffleSplit(n_splits=n_runs, test_size=0.2, random_state=self.random_state)
        for train_idx, test_idx in sss.split(X, y):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            n_neg, n_pos = (y_tr == 0).sum(), (y_tr == 1).sum()
            pipes = self.get_v3_pipelines(n_neg, n_pos)
            _, pipe = pipes[0]
            pipe.fit(X_tr, y_tr)
            scores.append(balanced_accuracy_score(y_te, pipe.predict(X_te)))
        return scores

    def get_error_analysis(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Analyze FP, FN, TP, TN patterns."""
        y_pred = model.predict(X_test)
        fp_mask = (y_test.values == 0) & (y_pred == 1)
        fn_mask = (y_test.values == 1) & (y_pred == 0)
        tp_mask = (y_test.values == 1) & (y_pred == 1)
        tn_mask = (y_test.values == 0) & (y_pred == 0)
        return {
            "counts": {"TP": int(tp_mask.sum()), "TN": int(tn_mask.sum()),
                       "FP": int(fp_mask.sum()), "FN": int(fn_mask.sum())},
            "fp_data": X_test[fp_mask], "fn_data": X_test[fn_mask],
            "y_pred": y_pred
        }

    def get_patient_shap_highlights(self, shap_values: Any, feature_names: List[str], patient_idx: int = 0, top_n: int = 10) -> pd.DataFrame:
        """Extracts top_n features with highest absolute SHAP impact for a patient."""
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        # Handle SHAP .values or array
        vals = sv.values[patient_idx] if hasattr(sv, 'values') else sv[patient_idx]
        
        df = pd.DataFrame({
            'Feature': feature_names,
            'Impact': vals
        })
        df['AbsImpact'] = df['Impact'].abs()
        return df.sort_values('AbsImpact', ascending=False).head(top_n).drop(columns=['AbsImpact'])

    # ── OPTUNA HYPERPARAMETER TUNING ──────────────────────────
    def tune_with_optuna(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, n_trials: int = 50) -> Dict[str, Any]:
        """Hyperparameter tuning with Optuna for top models."""
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        results = {}

        # 1. Random Forest
        def rf_objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'class_weight': 'balanced',
                'random_state': self.random_state, 'n_jobs': -1
            }
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            return balanced_accuracy_score(y_test, model.predict(X_test))

        study_rf = optuna.create_study(direction='maximize', study_name='RandomForest')
        study_rf.optimize(rf_objective, n_trials=n_trials, show_progress_bar=False)
        results['Random Forest'] = {
            'best_params': study_rf.best_params,
            'best_score': study_rf.best_value,
            'optimization_history': [(t.number, t.value) for t in study_rf.trials if t.value is not None]
        }

        # 2. Gradient Boosting
        def gb_objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'random_state': self.random_state
            }
            model = GradientBoostingClassifier(**params)
            model.fit(X_train, y_train)
            return balanced_accuracy_score(y_test, model.predict(X_test))

        study_gb = optuna.create_study(direction='maximize', study_name='GradientBoosting')
        study_gb.optimize(gb_objective, n_trials=n_trials, show_progress_bar=False)
        results['Gradient Boosting'] = {
            'best_params': study_gb.best_params,
            'best_score': study_gb.best_value,
            'optimization_history': [(t.number, t.value) for t in study_gb.trials if t.value is not None]
        }

        # 3. XGBoost
        if XGBClassifier:
            def xgb_objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                    'eval_metric': 'logloss', 'random_state': self.random_state
                }
                model = XGBClassifier(**params)
                model.fit(X_train, y_train)
                return balanced_accuracy_score(y_test, model.predict(X_test))

            study_xgb = optuna.create_study(direction='maximize', study_name='XGBoost')
            study_xgb.optimize(xgb_objective, n_trials=n_trials, show_progress_bar=False)
            results['XGBoost'] = {
                'best_params': study_xgb.best_params,
                'best_score': study_xgb.best_value,
                'optimization_history': [(t.number, t.value) for t in study_xgb.trials if t.value is not None]
            }

        # 4. LightGBM
        if LGBMClassifier:
            def lgbm_objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                    'is_unbalance': True, 'random_state': self.random_state, 'verbose': -1
                }
                model = LGBMClassifier(**params)
                model.fit(X_train, y_train)
                return balanced_accuracy_score(y_test, model.predict(X_test))

            study_lgbm = optuna.create_study(direction='maximize', study_name='LightGBM')
            study_lgbm.optimize(lgbm_objective, n_trials=n_trials, show_progress_bar=False)
            results['LightGBM'] = {
                'best_params': study_lgbm.best_params,
                'best_score': study_lgbm.best_value,
                'optimization_history': [(t.number, t.value) for t in study_lgbm.trials if t.value is not None]
            }

        return results

    # ── RISK TIMELINE SIMULATION ──────────────────────────────
    def simulate_risk_progression(self, model: Any, patient_data: pd.DataFrame, feature_cols: List[str], years: int = 5, scenario: str = "no_intervention") -> pd.DataFrame:
        """Simulate patient risk over time with annual feature changes.
        
        Scenarios:
            no_intervention: natural disease progression
            with_intervention: lifestyle improvements applied
        """
        if scenario == "with_intervention":
            annual_changes = {
                'Age': 1.0, 'BMI': -0.5, 'SystolicBP': -3.0, 'DiastolicBP': -2.0,
                'FastingBloodSugar': -3.0, 'HbA1c': -0.1, 'CholesterolTotal': -5.0,
                'HemoglobinLevels': 0.05, 'FatigueLevels': -0.4,
                'PhysicalActivity': 10, 'DietQuality': 0.5, 'SleepQuality': 0.3,
                'QualityOfLifeScore': 3,
            }
        else:
            annual_changes = {
                'Age': 1.0, 'BMI': 0.3, 'SystolicBP': 1.5, 'DiastolicBP': 0.8,
                'FastingBloodSugar': 2.0, 'HbA1c': 0.1, 'CholesterolTotal': 3.0,
                'HemoglobinLevels': -0.1, 'FatigueLevels': 0.3,
                'PhysicalActivity': -5, 'DietQuality': -0.2, 'SleepQuality': -0.1,
                'QualityOfLifeScore': -2,
            }

        timeline = []
        current = patient_data.copy()

        for year in range(years + 1):
            prob = float(model.predict_proba(current[feature_cols])[0, 1])
            level = 'Low' if prob < 0.3 else 'Moderate' if prob < 0.7 else 'High'
            timeline.append({'Year': year, 'Risk_Score': prob, 'Risk_Level': level})
            # Apply changes for next year
            for feat, delta in annual_changes.items():
                if feat in current.columns:
                    current[feat] = current[feat] + delta

        return pd.DataFrame(timeline)

    # ── MODEL SAVING FOR API ──────────────────────────────────
    @staticmethod
    def save_model_for_api(model: Any, feature_names: List[str], model_name: str, threshold: float, output_dir: str = "saved_models") -> str:
        """Save trained model and metadata for the FastAPI endpoint."""
        import joblib, os
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(model, os.path.join(output_dir, "best_model.pkl"))
        joblib.dump(list(feature_names), os.path.join(output_dir, "feature_names.pkl"))
        joblib.dump({
            "model_name": model_name,
            "threshold": float(threshold),
            "features_count": len(feature_names),
        }, os.path.join(output_dir, "model_meta.pkl"))
        return os.path.join(output_dir, "best_model.pkl")
