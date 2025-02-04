
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import nest_asyncio
import uvicorn
import joblib
import pandas as pd
from pydantic import BaseModel
from urllib.parse import urlparse
import re

# Apply nest_asyncio to run in Jupyter Notebooks
nest_asyncio.apply()

app = FastAPI()

# Serve static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Load your XGBoost model
model = joblib.load('XGBoostClassifier.pickle.dat')

# Define the input data model
class URLRequest(BaseModel):
    url: str

# Define the API for URL prediction
@app.post("/predict")
async def predict(request: URLRequest):
    url = request.url

    features = extract_features(url)
    prediction = model.predict(features)[0]
    result = "Phishing" if prediction == 1 else "Safe"

    return {"url": url, "prediction": result}

# Feature extraction function
def extract_features(url):
    features = []
    url = preprocess_url(url)
    features.append(1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0)  # IP in URL
    features.append(1 if '@' in url else 0)  # '@' symbol in URL
    features.append(1 if len(url) >= 54 else 0)  # Length of URL
    return pd.DataFrame([features])

def preprocess_url(url):
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    return url

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
