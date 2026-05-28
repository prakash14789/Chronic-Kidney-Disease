"""
CKD Clinical Intelligence — REST API
FastAPI endpoint for real-time CKD risk prediction.
Run: uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import os
import logging
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from database import log_audit_action, SessionLocal, User, verify_password

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

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Age": 65,
                    "Gender": 1,
                    "BMI": 28.5,
                    "Smoking": 1,
                    "PhysicalActivity": 90,
                    "DietQuality": 4,
                    "SleepQuality": 5,
                    "SystolicBP": 145,
                    "DiastolicBP": 90,
                    "FastingBloodSugar": 120,
                    "HbA1c": 6.8,
                    "HemoglobinLevels": 12.5,
                    "CholesterolTotal": 220,
                    "SerumElectrolytesSodium": 138,
                    "SerumElectrolytesPotassium": 4.8,
                    "FamilyHistoryKidneyDisease": 1,
                    "FamilyHistoryHypertension": 1,
                    "FamilyHistoryDiabetes": 1,
                    "Edema": 1,
                    "FatigueLevels": 7,
                    "QualityOfLifeScore": 55,
                    "HeavyMetalsExposure": 0,
                    "Adherence": 1
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    risk_score: float = Field(..., description="Raw probability score (0.0 to 1.0) of CKD presence.")
    risk_percentage: str = Field(..., description="Human-readable percentage string (e.g., '72.5%').")
    risk_level: str = Field(..., description="Stratified risk level: 'Low Risk', 'Moderate Risk', or 'High Risk'.")
    risk_color: str = Field(..., description="Hex color code associated with the risk level for UI rendering.")
    clinical_action: str = Field(..., description="Recommended next steps for the clinician.")
    model_used: str = Field(..., description="Name of the underlying ML model making the prediction.")
    threshold: float = Field(..., description="The probability threshold used for binary classification.")


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


# ── Auth & Security ────────────────────────────────────────

SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "super-secret-jwt-key-demo")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        user_id: int = payload.get("id")
        if username is None or user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    db = SessionLocal()
    user = db.query(User).filter(User.username == username).first()
    db.close()
    if user is None:
        raise credentials_exception
    return user


# ── Endpoints ──────────────────────────────────────────────

@app.post("/token", tags=["Auth"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    db = SessionLocal()
    user = db.query(User).filter(User.username == form_data.username).first()
    db.close()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "id": user.id}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}
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


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"],
          summary="Predict CKD risk for a single patient",
          description="Analyzes clinical and lifestyle factors to generate a personalized Chronic Kidney Disease risk assessment. Returns a comprehensive evaluation including probability score, risk stratification, and recommended clinical actions.",
          responses={
              200: {
                  "description": "Successfully calculated risk score.",
                  "content": {
                      "application/json": {
                          "example": {
                              "risk_score": 0.825,
                              "risk_percentage": "82.5%",
                              "risk_level": "High Risk",
                              "risk_color": "#FF6B6B",
                              "clinical_action": "Immediate medical attention required. Urgent nephrologist referral.",
                              "model_used": "XGBoost",
                              "threshold": 0.45
                          }
                      }
                  }
              },
              503: {"description": "Model not loaded. Ensure the backend ML system is initialized."}
          })
def predict(patient: PatientInput, current_user: User = Depends(get_current_user)) -> PredictionResponse:
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

    # Audit log
    log_audit_action(current_user.id, "PREDICT_RISK", "Unknown", f"Risk Score: {prob:.4f}")

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
def predict_batch(patients: List[PatientInput], current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
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

    # Audit log
    log_audit_action(current_user.id, "PREDICT_BATCH", "BATCH", f"Processed {len(results)} patients")

    return {"predictions": results, "count": len(results)}


@app.post("/predict/fhir", tags=["Prediction", "EMR"],
          summary="Generate FHIR-compliant RiskAssessment",
          description="Accepts standard patient input and returns an HL7 FHIR (Fast Healthcare Interoperability Resources) R4 Bundle containing the Patient resource and RiskAssessment resource. Ideal for direct integration with hospital Electronic Medical Records (EMR) systems like Epic or Cerner.")
def predict_fhir(patient: PatientInput, current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
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
    
    # Audit log
    log_audit_action(current_user.id, "PREDICT_FHIR", "Unknown", f"Generated FHIR Bundle. Risk: {prob:.4f}")

    return fhir_bundle

@app.post("/report/async", tags=["Reports"])
def generate_report_async(patient: PatientInput, patient_id: str = "Unknown", current_user: User = Depends(get_current_user)):
    """Trigger a background task to generate a complex PDF report."""
    from tasks import generate_comprehensive_report_task
    
    # In a real scenario, model prediction would be generated or fetched here
    patient_data = patient.model_dump()
    patient_data["patient_id"] = patient_id
    
    # Send task to Celery
    task = generate_comprehensive_report_task.delay(patient_data, {"mock": "prediction"})
    
    # Audit log
    log_audit_action(current_user.id, "EXPORT_PDF_ASYNC", patient_id, f"Task ID: {task.id}")
    
    return {
        "status": "Task queued",
        "task_id": task.id,
        "message": "Report generation is running in the background."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
