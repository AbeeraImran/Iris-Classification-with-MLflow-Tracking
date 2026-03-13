import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Setup
mlflow.set_experiment("Iris_Classification_Experiment")

# Train a quick model
iris = load_iris()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(iris.data, iris.target)

print("--- Starting Task 7: Model Registration ---")

with mlflow.start_run(run_name="Model_Registration_Task7") as run:
    # 1. Log the model first
    mlflow.sklearn.log_model(rf_model, "random_forest_model")
    
    # 2. Get the specific tracking URI for this exact model run
    model_uri = f"runs:/{run.info.run_id}/random_forest_model"
    
    # 3. Register the model to the Model Registry [cite: 85]
    model_name = "MLflow Iris Classifier" # [cite: 87]
    print(f"Registering model as: '{model_name}'...")
    mlflow.register_model(model_uri=model_uri, name=model_name)
    
    print(" Model registered successfully!")

print("\nTask 7 complete! Run `mlflow ui` and check the 'Models' tab at the top of the screen.")