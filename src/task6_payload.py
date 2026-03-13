import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

print("--- Starting Task 6: Payload Validation ---")

# Train a quick model just for this prediction test
iris = load_iris()
rf_model = RandomForestClassifier(n_estimators=10, random_state=42).fit(iris.data, iris.target)

# Validation Function [cite: 77-79]
def validate_payload(data):
    print("\nValidating incoming data payload...")
    
    # Check 1: Ensure input shape matches training data (4 features for Iris) [cite: 73]
    if data.shape[1] != 4:
        raise ValueError("Invalid input features: Expected 4 features.")
        
    # Check 2: Ensure no missing values [cite: 74]
    if data.isnull().values.any():
        raise ValueError("Invalid input: Contains missing values.")
        
    print(" Payload is valid! All checks passed.")
    return True

# Create dummy input data (1 flower, 4 measurements)
new_data = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]])

# Run validation and predict [cite: 80]
if validate_payload(new_data):
    print("Proceeding to inference...")
    prediction = rf_model.predict(new_data)
    print(f"Prediction output class: {prediction} (Target name: {iris.target_names[prediction[0]]})")