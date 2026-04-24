import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class CKDDataProcessor:
    def __init__(self, data_path="Chronickidneydiseases.csv", random_state=42):
        self.data_path = data_path
        self.random_state = random_state
        self.le_adherence = LabelEncoder()
        
    def load_raw_data(self):
        return pd.read_csv(self.data_path)
    
    def get_v3_refined_data(self, df_full, sample_n=5000):
        """EXACT V3 Data Refining Logic."""
        # 1. Stratified Sample
        idx, _ = train_test_split(
            df_full.index, train_size=sample_n,
            stratify=df_full["Diagnosis"], random_state=self.random_state
        )
        df = df_full.loc[idx].reset_index(drop=True)
        
        # 2. Drop identifiers
        drop_cols = ["PatientID", "RecommendedVisitsPerMonth"]
        df.drop(columns=drop_cols, inplace=True, errors="ignore")
        
        return df

    def split_and_encode_v3(self, df, target="Diagnosis"):
        """EXACT V3 Encoding and Splitting Logic."""
        X = df.drop(columns=[target])
        y = df[target]
        
        # Split first
        X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Fit encoder on TRAIN only
        le = LabelEncoder()
        X_tr = X_tr_raw.copy()
        X_te = X_te_raw.copy()
        X_tr["Adherence"] = le.fit_transform(X_tr["Adherence"])
        
        # Map to test (Handle unseen)
        adherence_map = {cls: idx for idx, cls in enumerate(le.classes_)}
        X_te["Adherence"] = X_te["Adherence"].map(lambda x: adherence_map.get(x, -1))
        
        return X_tr, X_te, y_tr, y_te
