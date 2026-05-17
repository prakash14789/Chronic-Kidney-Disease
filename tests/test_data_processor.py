import pandas as pd
import numpy as np
from data_processor import CKDDataProcessor

def test_get_v3_refined_data():
    processor = CKDDataProcessor()
    # Create a dummy dataframe with minimum required columns
    df = pd.DataFrame({
        "PatientID": range(10),
        "Diagnosis": [0, 1] * 5,
        "Adherence": ["Yes", "No"] * 5,
        "RecommendedVisitsPerMonth": [1] * 10
    })
    
    # Use a small sample size for testing
    refined_df = processor.get_v3_refined_data(df, sample_n=4)
    
    assert len(refined_df) == 4
    assert "PatientID" not in refined_df.columns
    assert "RecommendedVisitsPerMonth" not in refined_df.columns
    assert "Diagnosis" in refined_df.columns

def test_split_and_encode_v3():
    processor = CKDDataProcessor()
    df = pd.DataFrame({
        "Diagnosis": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "Adherence": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
        "Age": range(10)
    })
    
    X_tr, X_te, y_tr, y_te = processor.split_and_encode_v3(df)
    
    assert len(X_tr) == 8
    assert len(X_te) == 2
    assert "Adherence" in X_tr.columns
    # Check if Adherence is encoded (should be numeric)
    assert np.issubdtype(X_tr["Adherence"].dtype, np.number)
