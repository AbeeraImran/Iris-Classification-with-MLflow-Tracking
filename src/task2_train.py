import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Setup from Task 1
mlflow.set_experiment("Iris_Classification_Experiment")

# Task 2: Load and Split Dataset
print("Loading Iris dataset and splitting into train/test sets...")
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)
print(f"Data split complete. Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# Initialize Models
lr_model = LogisticRegression(max_iter=200)
rf_model = RandomForestClassifier(n_estimators=100)

print("\n--- Training Models ---")

# Train Logistic Regression
with mlflow.start_run(run_name="Logistic_Regression"):
    print("Training Logistic Regression model...")
    lr_model.fit(X_train, y_train)
    print("✅ Logistic Regression training complete!")

# Train Random Forest
with mlflow.start_run(run_name="Random_Forest"):
    print("Training Random Forest model...")
    rf_model.fit(X_train, y_train)
    print("✅ Random Forest training complete!")

print("\nTask 2 Execution Finished Successfully!")