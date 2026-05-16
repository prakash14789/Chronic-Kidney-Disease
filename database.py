import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class PatientRecord(Base):
    __tablename__ = 'patient_records'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(50), nullable=False)
    age = Column(Float)
    gender = Column(String(10))
    risk_score = Column(Float)
    risk_level = Column(String(50))
    clinical_action = Column(String(200))
    features = Column(JSON) # Store raw features for future reference
    timestamp = Column(DateTime, default=datetime.utcnow)

# Using SQLite for simplicity in this demo, could easily be postgresql://user:pass@localhost/db
engine = create_engine('sqlite:///ckd_patients.db', connect_args={'check_same_thread': False})
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

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
            features=features
        )
        session.add(record)
        session.commit()
    finally:
        session.close()

def get_patient_history(patient_id: str):
    session = SessionLocal()
    try:
        records = session.query(PatientRecord).filter(PatientRecord.patient_id == patient_id).order_by(PatientRecord.timestamp).all()
        return records
    finally:
        session.close()
