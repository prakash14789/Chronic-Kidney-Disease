import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score


class CKDStagePredictor:
    """Multi-class CKD Stage Prediction based on KDIGO clinical guidelines.
    
    Predicts CKD stages (1-5) using non-leakage features, where stages
    are derived from GFR ranges per clinical standards.
    """

    STAGE_DEFINITIONS = {
        1: {"name": "Stage 1", "gfr_range": "≥90 mL/min", "severity": "Normal/High",
            "description": "Kidney damage with normal or increased GFR",
            "color": "#4D96FF", "icon": "🟢",
            "action": "Monitor annually. Manage underlying conditions (diabetes, hypertension)."},
        2: {"name": "Stage 2", "gfr_range": "60–89 mL/min", "severity": "Mild",
            "description": "Kidney damage with mildly decreased GFR",
            "color": "#4CC9F0", "icon": "🔵",
            "action": "Estimate progression rate. Manage cardiovascular risk factors."},
        3: {"name": "Stage 3", "gfr_range": "30–59 mL/min", "severity": "Moderate",
            "description": "Moderately decreased GFR",
            "color": "#FFD93D", "icon": "🟡",
            "action": "Evaluate and treat complications. Consider nephrologist referral."},
        4: {"name": "Stage 4", "gfr_range": "15–29 mL/min", "severity": "Severe",
            "description": "Severely decreased GFR",
            "color": "#FF9F1C", "icon": "🟠",
            "action": "Prepare for renal replacement therapy. Nephrologist required."},
        5: {"name": "Stage 5", "gfr_range": "<15 mL/min", "severity": "Kidney Failure",
            "description": "Kidney failure — end-stage renal disease",
            "color": "#FF6B6B", "icon": "🔴",
            "action": "Dialysis or transplant required. Immediate specialist care."}
    }

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.feature_names = None

    @staticmethod
    def gfr_to_stage(gfr):
        """Map GFR value to CKD stage per KDIGO guidelines."""
        if gfr >= 90: return 1
        elif gfr >= 60: return 2
        elif gfr >= 30: return 3
        elif gfr >= 15: return 4
        else: return 5

    def prepare_stage_data(self, df):
        """Create stage labels from GFR and prepare leakage-free features."""
        if 'GFR' not in df.columns:
            raise ValueError("GFR column required for stage labeling")

        df_stage = df.copy()
        df_stage['CKD_Stage'] = df_stage['GFR'].apply(self.gfr_to_stage)

        # Remove leakage columns + identifiers so the model predicts stages fairly
        leakage_cols = ["GFR", "SerumCreatinine", "BUNLevels", "ProteinInUrine", "ACR", "Diagnosis"]
        drop_cols = ["PatientID", "RecommendedVisitsPerMonth"] + leakage_cols
        features = df_stage.drop(columns=[c for c in drop_cols if c in df_stage.columns])

        if 'Adherence' in features.columns:
            # Check if it contains strings (sometimes it's object or string dtype)
            if features['Adherence'].dtype == 'object' or features['Adherence'].dtype == 'string' or isinstance(features['Adherence'].iloc[0], str):
                le = LabelEncoder()
                features['Adherence'] = le.fit_transform(features['Adherence'].astype(str))

        return features, df_stage['CKD_Stage']

    def train(self, df, sample_n=5000):
        """Train the multi-class stage predictor and return evaluation results."""
        # Stratified sample
        stages = df['GFR'].apply(self.gfr_to_stage)
        if len(df) > sample_n:
            idx, _ = train_test_split(
                df.index, train_size=sample_n,
                stratify=stages, random_state=self.random_state
            )
            df_sample = df.loc[idx].reset_index(drop=True)
        else:
            df_sample = df.copy()

        X, y = self.prepare_stage_data(df_sample)
        self.feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        self.model = GradientBoostingClassifier(
            n_estimators=150, max_depth=6, subsample=0.8,
            random_state=self.random_state
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        stage_dist = y.value_counts().sort_index().to_dict()

        return {
            'report': report,
            'balanced_accuracy': bal_acc,
            'stage_distribution': stage_dist,
            'X_test': X_test, 'y_test': y_test, 'y_pred': y_pred,
            'feature_importances': pd.Series(
                self.model.feature_importances_, index=self.feature_names
            ).sort_values(ascending=False)
        }

    def predict_stage(self, patient_data):
        """Predict CKD stage for a single patient row."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        input_df = patient_data[self.feature_names]
        probs = self.model.predict_proba(input_df)
        pred_stage = self.model.predict(input_df)[0]

        stage_probs = {}
        for i, cls in enumerate(self.model.classes_):
            stage_probs[int(cls)] = float(probs[0][i])

        return {
            'predicted_stage': int(pred_stage),
            'stage_probabilities': stage_probs,
            'stage_info': self.STAGE_DEFINITIONS[int(pred_stage)]
        }

    def get_stage_info(self, stage):
        """Get clinical info for a given CKD stage."""
        return self.STAGE_DEFINITIONS.get(stage, {})
