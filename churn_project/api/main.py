from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

#Load model at startup
model = joblib.load('models/churn_model.joblib')

app = FastAPI(title='Churn Prediction API')

#Expected input
class CustomerInput(BaseModel):
    tenure_months: int
    monthly_charges: float
    total_charges: float
    contract_type: str
    has_internet: str
    payment_method: str

@app.post('/predict')
async def predict_churn(customer: CustomerInput):
    #Convert input to df
    data = pd.DataFrame([customer.model_dump()])

    #predict using pipeline
    prediction = model.predict(data)[0]

    return {'churn_prediction': int(prediction)}