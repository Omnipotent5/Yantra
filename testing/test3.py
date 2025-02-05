import pandas as pd
import pickle
import re
from urllib.parse import urlparse
from sklearn.preprocessing import StandardScaler

# Load the trained model (XGBoost or MLP)
model_path = 'xgboost-web-app\\backend\\models\\mlp_model.pkl'  # Change to mlp_model.pkl to use MLP
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Detect if XGBoost or MLP is being used
is_xgboost = 'xgboost' in str(type(model)).lower()

# Feature Extraction Function
def extract_features(url, model_type='mlp'):
    features = {}

    if model_type == 'xgboost':
        features['Have_IP'] = 1 if re.search(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', url) else 0
        features['Have_At'] = 1 if '@' in url else 0
        features['https_Domain'] = 1 if 'http' in urlparse(url).netloc or 'https' in urlparse(url).netloc else 0
        features['Prefix/Suffix'] = 1 if '-' in urlparse(url).netloc else 0
        features['iFrame'] = 0
    else:
        features['Having_IP'] = 1 if re.search(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', url) else 0
        features['Have_At_Sign'] = 1 if '@' in url else 0
        features['HTTP_in_Domain'] = 1 if 'http' in urlparse(url).netloc or 'https' in urlparse(url).netloc else 0
        features['Prefix_Suffix'] = 1 if '-' in urlparse(url).netloc else 0
        features['Iframe'] = 0

    # Common features
    features['URL_Length'] = len(url)
    features['URL_Depth'] = urlparse(url).path.count('/')
    features['Redirection'] = 1 if '//' in urlparse(url).path else 0
    features['TinyURL'] = 1 if len(urlparse(url).netloc) < 10 else 0
    features['DNS_Record'] = 1
    features['Web_Traffic'] = 1
    features['Domain_Age'] = 1
    features['Domain_End'] = 1
    features['Mouse_Over'] = 0
    features['Right_Click'] = 0
    features['Web_Forwards'] = 0

    return pd.DataFrame([features])

# Example URL for testing
url = 'http://www.example.com'

# Extract features from the URL
features_df = extract_features(url, model_type='xgboost' if is_xgboost else 'mlp')

# Load the dataset to inspect feature names
final_dataset = pd.read_csv('xgboost-web-app\\backend\\final_dataset.csv')
X = final_dataset.drop(columns=['Domain', 'Label'])

# Display features extracted from the URL
print("Extracted features for prediction:", list(features_df.columns))

# Display feature names used in training
print("Feature names used in training:", list(X.columns))

# Rename XGBoost features to match training data
if is_xgboost:
    features_df.rename(columns={
        'Have_IP': 'Having_IP',
        'Have_At': 'Have_At_Sign',
        'https_Domain': 'HTTP_in_Domain',
        'Prefix/Suffix': 'Prefix_Suffix',
        'iFrame': 'Iframe'
    }, inplace=True)

# Reorder features to match model training
features_df = features_df[X.columns]

# Apply scaling only for MLP model
if not is_xgboost:
    scaler = StandardScaler()
    scaler.fit(X)
    features_scaled = scaler.transform(features_df)
    features_df = pd.DataFrame(features_scaled, columns=features_df.columns)

# Make prediction
prediction = model.predict(features_df)

# Interpret the result
result = 'Safe' if prediction[0] == 0 else 'Malicious'

print(f"The website '{url}' is predicted to be: {result}")
