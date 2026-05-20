import os
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy_utils import StringEncryptedType
from sqlalchemy_utils.types.encrypted.encrypted_type import FernetEngine
from passlib.context import CryptContext

Base = declarative_base()

# Encryption key for DB at rest (in production, use environment variable)
# Generated via: base64.urlsafe_b64encode(os.urandom(32))
ENCRYPTION_KEY = os.environ.get("DB_ENCRYPTION_KEY", "z_Hh4_i-N2sU-18TINPylQO7H7J1pU8b1iE14uHnEyw=")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    hashed_password = Column(String(100))
    role = Column(String(20), default="doctor") # doctor, admin

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    action = Column(String(100)) # e.g., "VIEW_PATIENT", "PREDICT_RISK", "EXPORT_PDF"
    target_patient_id = Column(String(50), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    details = Column(String(255), nullable=True)

class PatientRecord(Base):
    __tablename__ = 'patient_records'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(50), nullable=False, index=True)
    age = Column(Float)
    gender = Column(String(10))
    risk_score = Column(Float)
    risk_level = Column(String(50))
    # Encrypting PII/sensitive info at rest
    clinical_action = Column(StringEncryptedType(String(200), ENCRYPTION_KEY, FernetEngine))
    features = Column(StringEncryptedType(String(5000), ENCRYPTION_KEY, FernetEngine)) # Store raw features as encrypted JSON string
    timestamp = Column(DateTime, default=datetime.utcnow)

# Using SQLite for simplicity in this demo
DB_DIR = "data"
os.makedirs(DB_DIR, exist_ok=True)
engine = create_engine(f'sqlite:///{DB_DIR}/ckd_patients_secure.db', connect_args={'check_same_thread': False})
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_initial_admin():
    session = SessionLocal()
    try:
        admin = session.query(User).filter(User.username == "admin").first()
        if not admin:
            hashed_pw = get_password_hash("admin123")
            admin = User(username="admin", hashed_password=hashed_pw, role="admin")
            session.add(admin)
            session.commit()
    finally:
        session.close()

# Initialize admin user
create_initial_admin()

def save_patient_record(patient_id: str, age: float, gender: str, risk_score: float, risk_level: str, clinical_action: str, features: dict):
    session = SessionLocal()
    try:
        record = PatientRecord(
            patient_id=patient_id,
            age=age,
            gender=gender,
            risk_score=risk_score,
            risk_level=risk_level,
            clinical_action=clinical_action,
            features=json.dumps(features) # Store as encrypted string
        )
        session.add(record)
        session.commit()
    finally:
        session.close()

def get_patient_history(patient_id: str):
    session = SessionLocal()
    try:
        records = session.query(PatientRecord).filter(PatientRecord.patient_id == patient_id).order_by(PatientRecord.timestamp).all()
        # Parse features back to dict
        for r in records:
            if r.features and isinstance(r.features, str):
                try:
                    r.features = json.loads(r.features)
                except:
                    r.features = {}
        return records
    finally:
        session.close()

def log_audit_action(user_id: int, action: str, target_patient_id: str = None, details: str = None):
    session = SessionLocal()
    try:
        log = AuditLog(
            user_id=user_id,
            action=action,
            target_patient_id=target_patient_id,
            details=details
        )
        session.add(log)
        session.commit()
    finally:
        session.close()
