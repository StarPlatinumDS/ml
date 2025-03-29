# Churn Prediction API – Usage Guide

This guide walks you through setting up, running, and testing the Churn Prediction API locally or in Docker.

---

## 🔧 1. Project Setup

### 🔹 Install dependencies

Using pip:

```bash
pip install -r requirements.txt
```
Or with conda (optional):
```
conda create -n churn_env python=3.9
conda activate churn_env
pip install -r requirements.txt
```

---
## 🚀 2. Train the Model
### Run the notebook:
```aiignore
notebooks/sim_churn.ipynb
```
This generates:
```
models/churn_model.joblib
```

---
## 🌐 3. Run API Locally
### Start the FastAPI server:
```
uvicorn api.main:app --reload
```
Go to: http://localhost:8000/docs

Test the /predict endpoint using example input.

---
## 🐳 4. Dockerize the Project
🔹 Build the Docker image:
```
docker build -t churn-api .
```

🔹 Run the container:
```
docker run -p 8000:8000 churn-api
```

Test again at: http://localhost:8000/docs

---
## 🧪 5. Run Tests
### Unit + integration tests live in the tests/ folder.

Run with:
```e
pytest tests/
```

---
## 💡 Sample JSON for /predict

```
{
  "tenure_months": 24,
  "monthly_charges": 65.5,
  "total_charges": 1572,
  "contract_type": "Month-to-month",
  "has_internet": "Yes",
  "payment_method": "Electronic check"
}
```