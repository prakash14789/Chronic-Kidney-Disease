"""
CKD Clinical Intelligence — REST API
FastAPI endpoint for real-time CKD risk prediction.
Run: uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    loaded = load_models()
    if loaded:
        logger.info(f"✅ Model loaded with {len(FEATURE_NAMES)} features")
    else:
        logger.warning("⚠️  No saved model found. Run the Streamlit app first and click 'Save Model for API'.")
    yield

app = FastAPI(
    title="CKD Clinical Intelligence API",
    description="REST API for Chronic Kidney Disease risk prediction powered by ML",
    version="3.2",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Globals ────────────────────────────────────────────────
MODEL = None
FEATURE_NAMES = None
MODEL_DIR = "saved_models"


# ── Pydantic Models ────────────────────────────────────────
class PatientInput(BaseModel):
    """Patient clinical input for CKD risk prediction."""
    Age: Optional[float] = Field(50, description="Patient age in years", ge=0, le=120)
    Gender: Optional[int] = Field(1, description="1=Male, 0=Female", ge=0, le=1)
    BMI: Optional[float] = Field(25.0, description="Body Mass Index", ge=10, le=100)
    Smoking: Optional[int] = Field(0, description="1=Yes, 0=No", ge=0, le=1)
    PhysicalActivity: Optional[float] = Field(150, description="Minutes per week", ge=0, le=10080)
    DietQuality: Optional[float] = Field(5, description="0-10 scale", ge=0, le=10)
    SleepQuality: Optional[float] = Field(7, description="0-10 scale", ge=0, le=10)
    SystolicBP: Optional[float] = Field(120, description="Systolic blood pressure mmHg", ge=50, le=300)
    DiastolicBP: Optional[float] = Field(80, description="Diastolic blood pressure mmHg", ge=30, le=200)
    FastingBloodSugar: Optional[float] = Field(100, description="mg/dL", ge=50, le=500)
    HbA1c: Optional[float] = Field(5.5, description="Glycated hemoglobin %", ge=3, le=20)
    HemoglobinLevels: Optional[float] = Field(14.0, description="g/dL", ge=5, le=25)
    CholesterolTotal: Optional[float] = Field(180, description="mg/dL", ge=50, le=1000)
    SerumElectrolytesSodium: Optional[float] = Field(140, description="mEq/L", ge=100, le=200)
    SerumElectrolytesPotassium: Optional[float] = Field(4.5, description="mEq/L", ge=1, le=10)
    FamilyHistoryKidneyDisease: Optional[int] = Field(0, description="1=Yes, 0=No", ge=0, le=1)
    FamilyHistoryHypertension: Optional[int] = Field(0, description="1=Yes, 0=No", ge=0, le=1)
    FamilyHistoryDiabetes: Optional[int] = Field(0, description="1=Yes, 0=No", ge=0, le=1)
    Edema: Optional[int] = Field(0, description="1=Yes, 0=No", ge=0, le=1)
    FatigueLevels: Optional[float] = Field(3, description="0-10 scale", ge=0, le=10)
    QualityOfLifeScore: Optional[float] = Field(70, description="0-100", ge=0, le=100)
    HeavyMetalsExposure: Optional[int] = Field(0, description="1=Yes, 0=No", ge=0, le=1)
    Adherence: Optional[int] = Field(0, description="0=Adherent, 1=Non-Adherent", ge=0, le=1)


class PredictionResponse(BaseModel):
    risk_score: float
    risk_percentage: str
    risk_level: str
    risk_color: str
    clinical_action: str
    model_used: str
    threshold: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    features_count: int


# ── Helper ─────────────────────────────────────────────────
def get_assessment(probability: float) -> Dict[str, str]:
    """Risk stratification matching the Streamlit app logic."""
    if probability < 0.3:
        return {"level": "Low Risk", "color": "#4D96FF",
                "action": "Routine monitoring recommended. Maintain healthy lifestyle."}
    elif probability < 0.7:
        return {"level": "Moderate Risk", "color": "#FFD93D",
                "action": "Further diagnostic tests advised. Schedule follow-up."}
    else:
        return {"level": "High Risk", "color": "#FF6B6B",
                "action": "Immediate medical attention required. Urgent nephrologist referral."}


def load_models() -> bool:
    """Load saved model and feature names."""
    global MODEL, FEATURE_NAMES
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    features_path = os.path.join(MODEL_DIR, "feature_names.pkl")

    if os.path.exists(model_path) and os.path.exists(features_path):
        MODEL = joblib.load(model_path)
        FEATURE_NAMES = joblib.load(features_path)
        return True
    return False


# ── Events ─────────────────────────────────────────────────
# Migrated to lifespan event handler above



# ── Endpoints ──────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """Check API health and model status."""
    meta = {}
    meta_path = os.path.join(MODEL_DIR, "model_meta.pkl")
    if os.path.exists(meta_path):
        meta = joblib.load(meta_path)

    return HealthResponse(
        status="healthy" if MODEL is not None else "model_not_loaded",
        model_loaded=MODEL is not None,
        model_name=meta.get("model_name", "unknown"),
        features_count=len(FEATURE_NAMES) if FEATURE_NAMES else 0
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(patient: PatientInput) -> PredictionResponse:
    """Predict CKD risk for a single patient."""
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run the Streamlit app and save the model first."
        )

    # Build input dataframe with correct feature order
    patient_dict = patient.model_dump()
    input_df = pd.DataFrame([patient_dict])

    # Fill any missing features with 0
    for feat in FEATURE_NAMES:
        if feat not in input_df.columns:
            input_df[feat] = 0

    input_df = input_df[FEATURE_NAMES]

    # Predict
    prob = float(MODEL.predict_proba(input_df)[0, 1])
    assessment = get_assessment(prob)

    # Load threshold
    meta_path = os.path.join(MODEL_DIR, "model_meta.pkl")
    meta = joblib.load(meta_path) if os.path.exists(meta_path) else {}

    return PredictionResponse(
        risk_score=round(prob, 4),
        risk_percentage=f"{prob:.1%}",
        risk_level=assessment["level"],
        risk_color=assessment["color"],
        clinical_action=assessment["action"],
        model_used=meta.get("model_name", "unknown"),
        threshold=meta.get("threshold", 0.5)
    )


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(patients: List[PatientInput]) -> Dict[str, Any]:
    """Predict CKD risk for multiple patients at once."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if not patients:
        return {"predictions": [], "count": 0}

    # Convert all patients to dicts
    patient_dicts = [p.model_dump() for p in patients]
    input_df = pd.DataFrame(patient_dicts)

    # Ensure all features are present and in correct order
    for feat in FEATURE_NAMES:
        if feat not in input_df.columns:
            input_df[feat] = 0
    input_df = input_df[FEATURE_NAMES]

    # Predict all at once
    probs = MODEL.predict_proba(input_df)[:, 1]

    results = []
    for prob in probs:
        prob_float = float(prob)
        assessment = get_assessment(prob_float)
        results.append({
            "risk_score": round(prob_float, 4),
            "risk_level": assessment["level"],
            "clinical_action": assessment["action"]
        })

    return {"predictions": results, "count": len(results)}


@app.post("/predict/fhir", tags=["Prediction", "EMR"])
def predict_fhir(patient: PatientInput) -> Dict[str, Any]:
    """Predict CKD risk and return a FHIR-compliant JSON bundle for EMR integration."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # Predict
    patient_dict = patient.model_dump()
    input_df = pd.DataFrame([patient_dict])
    for feat in FEATURE_NAMES:
        if feat not in input_df.columns:
            input_df[feat] = 0
    input_df = input_df[FEATURE_NAMES]
    
    prob = float(MODEL.predict_proba(input_df)[0, 1])
    assessment = get_assessment(prob)
    
    # Generate FHIR Bundle
    fhir_bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "gender": "male" if patient.Gender == 1 else "female",
                    "active": True
                }
            },
            {
                "resource": {
                    "resourceType": "RiskAssessment",
                    "status": "final",
                    "code": {
                        "coding": [{
                            "system": "http://snomed.info/sct",
                            "code": "709044004",
                            "display": "Chronic kidney disease risk assessment"
                        }]
                    },
                    "prediction": [{
                        "probabilityDecimal": round(prob, 4),
                        "qualitativeRisk": {
                            "text": assessment["level"]
                        }
                    }],
                    "mitigation": assessment["action"]
                }
            }
        ]
    }
    
    return fhir_bundle


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
