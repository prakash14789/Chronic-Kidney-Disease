import os
import time
from celery import Celery
import logging

# Ensure redis URL points to the docker container if running in docker,
# or localhost if running locally
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "ckd_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

logger = logging.getLogger(__name__)

@celery_app.task(bind=True)
def generate_comprehensive_report_task(self, patient_data: dict, model_prediction: dict):
    """
    Simulates a long-running background task, like generating a complex PDF report
    or running a background risk progression simulation.
    """
    logger.info(f"Starting report generation task for patient: {patient_data.get('patient_id', 'Unknown')}")
    
    # Simulate time-consuming work
    time.sleep(5) 
    
    logger.info("Report generation completed.")
    
    # In a real app, this might save the PDF to disk and return the file path
    return {
        "status": "success",
        "message": "Report generated successfully",
        "file_path": f"/reports/report_{patient_data.get('patient_id', 'Unknown')}.pdf"
    }
