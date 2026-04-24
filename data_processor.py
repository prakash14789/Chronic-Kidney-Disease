import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

class CKDDataProcessor:
    def __init__(self, data_path="Chronickidneydiseases.csv", random_state=42):
        self.data_path = data_path
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.le_adherence = LabelEncoder()
        
    def load_raw_data(self):
        """Loads the full dataset from CSV."""
        return pd.read_csv(self.data_path)
    
    def get_stratified_sample(self, df, sample_n=5000):
        """Returns a stratified sample of the data."""
        idx, _ = train_test_split(
            df.index, train_size=sample_n,
            stratify=df["Diagnosis"], random_state=self.random_state
        )
        return df.loc[idx].reset_index(drop=True)
    
    def refine_data(self, df):
        """Cleans and prepares data for modeling."""
        df_clean = df.copy()
        
        # Drop irrelevant columns
        drop_cols = ["PatientID", "RecommendedVisitsPerMonth"]
        df_clean.drop(columns=drop_cols, inplace=True, errors="ignore")
        
        # Encode categorical data
        if 'Adherence' in df_clean.columns:
            df_clean["Adherence"] = self.le_adherence.fit_transform(df_clean["Adherence"])
            
        return df_clean
    
    def split_data(self, df, target="Diagnosis", test_size=0.2):
        """Splits data into train and test sets."""
        X = df.drop(columns=[target])
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test

    def get_leakage_free_data(self, X):
        """Removes features known to cause data leakage."""
        leakage_cols = ["GFR", "SerumCreatinine", "BUNLevels", "ProteinInUrine", "ACR"]
        return X.drop(columns=leakage_cols, errors="ignore")
