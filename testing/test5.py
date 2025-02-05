from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle
import re
from urllib.parse import urlparse
import uvicorn

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Ensure POST is allowed
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


# Load the trained XGBoost model
with open('xgboost-web-app\\backend\\models\\xgboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Feature extraction function
def extract_features(url):
    features = {}
    features['Have_IP'] = 1 if re.search(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', url) else 0
    features['Have_At'] = 1 if '@' in url else 0
    features['URL_Length'] = len(url)
    features['URL_Depth'] = urlparse(url).path.count('/')
    features['Redirection'] = 1 if '//' in urlparse(url).path else 0
    features['https_Domain'] = 1 if 'http' in urlparse(url).netloc or 'https' in urlparse(url).netloc else 0
    features['TinyURL'] = 1 if len(urlparse(url).netloc) < 10 else 0
    features['Prefix/Suffix'] = 1 if '-' in urlparse(url).netloc else 0
    features['DNS_Record'] = 1  # Assume DNS record exists
    features['Web_Traffic'] = 1  # Assume average web traffic
    features['Domain_Age'] = 1   # Assume domain is aged
    features['Domain_End'] = 1   # Assume domain is not expiring soon
    features['iFrame'] = 0
    features['Mouse_Over'] = 0
    features['Right_Click'] = 0
    features['Web_Forwards'] = 0
    return pd.DataFrame([features])

# API Request Model
class URLInput(BaseModel):
    url: str

# Prediction Endpoint
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Ensure POST is allowed
    allow_headers=["*"],
)

@app.post("/predict")
async def predict_url(data: URLInput):
    print(f"Received URL: {data.url}")  # Log incoming URL
    try:
        features_df = extract_features(data.url)
        print("Extracted Features:", features_df)

        prediction = model.predict(features_df)
        result = 'Safe' if prediction[0] == 0 else 'Malicious'
        print(f"Prediction result: {result}")

        return {"url": data.url, "prediction": result}
    
    except Exception as e:
        print(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)

# Code to write the app to app.py
app_code = '''
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle
import re
from urllib.parse import urlparse
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files at /static instead of root
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


with open('./models/xgboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

def extract_features(url):
    features = {}
    features['Have_IP'] = 1 if re.search(r'\\b(?:[0-9]{1,3}\\.){3}[0-9]{1,3}\\b', url) else 0
    features['Have_At'] = 1 if '@' in url else 0
    features['URL_Length'] = len(url)
    features['URL_Depth'] = urlparse(url).path.count('/')
    features['Redirection'] = 1 if '//' in urlparse(url).path else 0
    features['https_Domain'] = 1 if 'http' in urlparse(url).netloc or 'https' in urlparse(url).netloc else 0
    features['TinyURL'] = 1 if len(urlparse(url).netloc) < 10 else 0
    features['Prefix/Suffix'] = 1 if '-' in urlparse(url).netloc else 0
    features['DNS_Record'] = 1
    features['Web_Traffic'] = 1
    features['Domain_Age'] = 1
    features['Domain_End'] = 1
    features['iFrame'] = 0
    features['Mouse_Over'] = 0
    features['Right_Click'] = 0
    features['Web_Forwards'] = 0
    return pd.DataFrame([features])

class URLInput(BaseModel):
    url: str

@app.post("/predict")
async def predict_url(data: URLInput):
    try:
        features_df = extract_features(data.url)
        prediction = model.predict(features_df)
        result = 'Safe' if prediction[0] == 0 else 'Malicious'
        return {"url": data.url, "prediction": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)
'''

with open("app.py", "w") as file:
    file.write(app_code)

print("app.py created successfully!")
