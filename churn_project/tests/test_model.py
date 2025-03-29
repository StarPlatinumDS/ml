import joblib
import pandas as pd
import os

def test_model_predicts_binary():
    model_path = os.path.join("models", "churn_model.joblib")
    model = joblib.load(model_path)

    #minimal input
    sample = pd.DataFrame([{
        "tenure_months": 12,
        "monthly_charges": 55.0,
        "total_charges": 660.0,
        "contract_type": "Month-to-month",
        "has_internet": "Yes",
        "payment_method": "Electronic check"
    }])

    prediction = model.predict(sample)[0]
    assert prediction in [0, 1], "Prediction should be binary"