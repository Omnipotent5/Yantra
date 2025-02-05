import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained MLP model
with open('xgboost-web-app\\backend\\models\\mlp_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the dataset
final_dataset = pd.read_csv('xgboost-web-app\\backend\\final_dataset.csv')

# Separate features and labels
X = final_dataset.drop(columns=['Domain', 'Label'])
y = final_dataset['Label']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Predict the first 10 samples instead of just one
sample_inputs = X_scaled[:10]
predictions = model.predict(sample_inputs)

# Display predictions
for i, pred in enumerate(predictions):
    result = 'Safe' if pred == 0 else 'Malicious'
    print(f"Website {i+1} is predicted to be: {result}")
