from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_predict_endpoint():
    payload = {
        "tenure_months": 24,
        "monthly_charges": 65.5,
        "total_charges": 1572,
        "contract_type": "Month-to-month",
        "has_internet": "Yes",
        "payment_method": "Electronic check"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "churn_prediction" in response.json()
    assert response.json()["churn_prediction"] in [0, 1]