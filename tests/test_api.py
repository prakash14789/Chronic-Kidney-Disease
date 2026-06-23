from fastapi.testclient import TestClient
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import api
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import app, get_current_user
import api
from ckd_app.database.secure_db import User

# Override dependency to simulate logged in user
def override_get_current_user():
    return User(id=1, username="test_admin", role="admin")

app.dependency_overrides[get_current_user] = override_get_current_user

client = TestClient(app)

def test_health_check_no_model():
    api.MODEL = None
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "model_not_loaded"
    assert response.json()["model_loaded"] is False

def test_health_check_with_model():
    class MockModel:
        pass
    
    api.MODEL = MockModel()
    api.FEATURE_NAMES = ["Age", "Gender"]
    
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True

def test_predict_model_not_loaded():
    api.MODEL = None
    response = client.post("/predict", json={"Age": 50})
    assert response.status_code == 503
    assert "Model not loaded" in response.json()["detail"]

def test_predict_success():
    class MockModel:
        def predict_proba(self, X):
            return np.array([[0.3, 0.7]]) # 0.7 risk
            
    api.MODEL = MockModel()
    api.FEATURE_NAMES = ["Age", "Gender"]
    
    response = client.post("/predict", json={"Age": 50, "Gender": 1})
    assert response.status_code == 200
    assert response.json()["risk_score"] == 0.7
    assert response.json()["risk_level"] == "High Risk"
    assert response.json()["risk_color"] == "#FF6B6B"

def test_predict_batch_success():
    class MockModel:
        def predict_proba(self, X):
            # Return 0.2 for first patient, 0.8 for second
            return np.array([[0.8, 0.2], [0.2, 0.8]])
            
    api.MODEL = MockModel()
    api.FEATURE_NAMES = ["Age", "Gender"]
    
    patients = [
        {"Age": 30, "Gender": 0},
        {"Age": 60, "Gender": 1}
    ]
    
    response = client.post("/predict/batch", json=patients)
    assert response.status_code == 200
    assert response.json()["count"] == 2
    assert response.json()["predictions"][0]["risk_score"] == 0.2
    assert response.json()["predictions"][0]["risk_level"] == "Low Risk"
    assert response.json()["predictions"][1]["risk_score"] == 0.8
    assert response.json()["predictions"][1]["risk_level"] == "High Risk"


def test_rag_query_success(monkeypatch):
    def mock_query(user_query, token=None):
        return {
            "answer": f"Mocked clinical response for: {user_query}",
            "sources": [{"title": "KDIGO Excerpt", "chunk": 0, "score": 0.95}]
        }
    monkeypatch.setattr(api.rag_engine, "query", mock_query)
    
    response = client.post("/rag/query", json={"query": "What BP is recommended?"})
    assert response.status_code == 200
    assert "Mocked clinical response" in response.json()["answer"]
    assert len(response.json()["sources"]) == 1
    assert response.json()["sources"][0]["title"] == "KDIGO Excerpt"


def test_rag_upload_success(monkeypatch):
    mock_called = False
    def mock_add_document(title, text, category="guidelines"):
        nonlocal mock_called
        mock_called = True
        
    monkeypatch.setattr(api.rag_engine, "add_document", mock_add_document)
    
    response = client.post("/rag/upload", json={
        "title": "New Guideline Reference",
        "content": "This is a test kidney guideline.",
        "category": "guidelines"
    })
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "successfully indexed" in response.json()["message"]
    assert mock_called is True

