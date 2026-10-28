from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health_endpoint():
    """
    verify that the health endpoint returns a 200 status code and the expected response
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_churn_prediction():
    """
    verify that the predict endpoint returns a 200 status code and the expected response
    """
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "class_name" in response.json()
    assert "churn_probability" in response.json()
    assert response.json()["class_name"] in ["stayed", "churned"]
    assert response.json()["prediction"] in [0, 1]
    assert 0.0 <= response.json()["churn_probability"] <= 1.0
