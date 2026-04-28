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
        """V3 Encoding and Splitting — encode before split for consistency."""
        X = df.drop(columns=[target])
        y = df[target]

        # Encode BEFORE split — same mapping everywhere, no -1 fallback
        le = LabelEncoder()
        X["Adherence"] = le.fit_transform(X["Adherence"])

        # Now split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        # Reset indices to avoid SHAP misalignment
        return (
            X_tr.reset_index(drop=True),
            X_te.reset_index(drop=True),
            y_tr.reset_index(drop=True),
            y_te.reset_index(drop=True)
        )
