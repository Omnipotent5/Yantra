from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import pickle
import traceback

# Load your XGBoost model
try:
    with open("models\\xgboost_model.pkl", "rb") as file:
        model = pickle.load(file)
except Exception as e:
    print("Error loading the model:", e)

# Initialize FastAPI app
app = FastAPI()

# Define input schema
class URLData(BaseModel):
    url: str

# Dummy preprocessing function (adjust based on your dataset)
def preprocess_url(url: str):
    try:
        features = {
            "url_length": len(url),
            "has_https": int("https" in url),
        }
        return pd.DataFrame([features])
    except Exception as e:
        print("Error during preprocessing:", e)
        traceback.print_exc()

# Prediction endpoint
@app.post("/predict")
def predict(data: URLData):
    try:
        features = preprocess_url(data.url)
        prediction = model.predict(features)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        print("Error during prediction:", e)
        traceback.print_exc()
        return {"error": str(e)}
