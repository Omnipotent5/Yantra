import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Load the trained MLP model
with open('xgboost-web-app\\backend\models\\xgboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the dataset
final_dataset = pd.read_csv('xgboost-web-app\\backend\\final_dataset.csv')

# Separate features and labels
X = final_dataset.drop(columns=['Domain', 'Label'])
y = final_dataset['Label']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Make predictions on the entire dataset
predictions = model.predict(X_scaled)

# Evaluate performance
print(classification_report(y, predictions))
