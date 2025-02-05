import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained MLP model
with open('./models/mlp_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the dataset
final_dataset = pd.read_csv('./final_dataset.csv')

# Drop non-numeric columns like 'Domain' and the target 'Label'
features = final_dataset.drop(columns=['Domain', 'Label'])

# Prepare example input (first row)
example_input = features.iloc[0].values.reshape(1, -1)

# Scale the data
scaler = StandardScaler()
scaler.fit(features)
example_input_scaled = scaler.transform(example_input)

# Make prediction
prediction = model.predict(example_input_scaled)

# Interpret the result
result = 'Safe' if prediction[0] == 0 else 'Malicious'

print(f"The website is predicted to be: {result}")
