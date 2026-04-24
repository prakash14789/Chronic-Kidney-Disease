import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class CKDDataProcessor:
    def __init__(self, data_path="Chronickidneydiseases.csv", random_state=42):
        self.data_path = data_path
        self.random_state = random_state
        self.categorical_encoders = {}
        
    def load_and_clean_data(self, sample_n=5000):
        """Loads data, removes duplicates, and takes a stratified sample."""
        df = pd.read_csv(self.data_path)
        
        # 1. Remove Duplicates (Requirement #4)
        initial_count = len(df)
        df.drop_duplicates(inplace=True)
        final_count = len(df)
        if initial_count > final_count:
            print(f"Removed {initial_count - final_count} duplicate rows.")
            
        # 2. Drop identifiers and obvious target proxies (Requirement #1)
        # GFR, Creatinine, etc., ARE the diagnosis definition.
        leakage_cols = [
            "PatientID", "RecommendedVisitsPerMonth", 
            "GFR", "SerumCreatinine", "BUNLevels", 
            "ProteinInUrine", "ACR"
        ]
        df.drop(columns=leakage_cols, inplace=True, errors="ignore")
        
        # 3. Stratified Sampling (for performance)
        idx, _ = train_test_split(
            df.index, train_size=min(sample_n, len(df)),
            stratify=df["Diagnosis"], random_state=self.random_state
        )
        return df.loc[idx].reset_index(drop=True)
    
    def prepare_train_test(self, df, target="Diagnosis", test_size=0.2):
        """Splits data and handles encoding correctly (Requirement #2 & #3)."""
        X = df.drop(columns=[target])
        y = df[target]
        
        # 1. Split BEFORE any encoding/scaling
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # 2. Identify and encode categorical columns (Fit on Train, Transform on Test)
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        
        X_train_enc = X_train.copy()
        X_test_enc = X_test.copy()
        
        for col in categorical_cols:
            le = LabelEncoder()
            # Fit only on training data
            X_train_enc[col] = le.fit_transform(X_train[col].astype(str))
            
            # Map test data safely (Requirement #3 - Handle unseen categories)
            test_mapping = {val: i for i, val in enumerate(le.classes_)}
            X_test_enc[col] = X_test[col].astype(str).map(test_mapping).fillna(-1).astype(int)
            
            self.categorical_encoders[col] = le
            
        return X_train_enc, X_test_enc, y_train, y_test
