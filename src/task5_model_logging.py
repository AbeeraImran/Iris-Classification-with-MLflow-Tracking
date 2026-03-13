import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Setup
mlflow.set_experiment("Iris_Classification_Experiment")

# Load Data & Train a fresh model
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print("--- Starting Task 5: Model Logging ---")

with mlflow.start_run(run_name="Model_Logging_Task5"):
    # Log the model (This is the core requirement for Task 5)
    mlflow.sklearn.log_model(rf_model, "random_forest_model")
    print("Model logged successfully!")

print("\nTask 5 complete! Run `mlflow ui` and check the Artifacts tab.")