from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Sentiment Analysis API is running!"}

def test_predict_positive():
    response = client.post("/predict", json={"text": "I am so happy"})
    assert response.status_code == 200
    assert "label" in response.json()
    assert "confidence" in response.json()