import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix

print("End-to-End MLflow Pipeline...")

# ==========================================
# TASK 1: Experiment Tracking
# ==========================================
experiment_name = "Iris_Classification_Experiment"
mlflow.set_experiment(experiment_name)
print(f"\n[Task 1] Experiment set to: '{experiment_name}'")

# Load and Split Data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)
os.makedirs("plots", exist_ok=True)

# ==========================================
# TASK 6: Payload Validation Setup
# ==========================================
def validate_payload(data):
    if data.shape[1] != 4:
        raise ValueError("Invalid input features: Expected 4 features.")
    if data.isnull().values.any():
        raise ValueError("Invalid input: Contains missing values.")
    return True

# ==========================================
# TASK 2 & 3: Train Models & Log Metrics
# ==========================================
# Model 1: Logistic Regression
with mlflow.start_run(run_name="Pipeline_Logistic_Regression") as run_lr:
    lr_params = {"C": 1.0, "max_iter": 200, "solver": "lbfgs"}
    mlflow.log_params(lr_params) 

    lr_model = LogisticRegression(**lr_params)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)

    mlflow.log_metric("Accuracy", accuracy_score(y_test, lr_pred))
    mlflow.log_metric("F1_Score", f1_score(y_test, lr_pred, average='macro'))
    mlflow.log_metric("Precision", precision_score(y_test, lr_pred, average='macro'))
    
    print(f"[Task 2 & 3] Logistic Regression trained & logged. Run ID: {run_lr.info.run_id}")

# Model 2: Random Forest (We use this for Artifacts, Logging, and Registration)
with mlflow.start_run(run_name="Pipeline_Random_Forest") as run_rf:
    rf_params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
    mlflow.log_params(rf_params)

    rf_model = RandomForestClassifier(**rf_params)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    mlflow.log_metric("Accuracy", accuracy_score(y_test, rf_pred))
    mlflow.log_metric("F1_Score", f1_score(y_test, rf_pred, average='macro'))
    mlflow.log_metric("Precision", precision_score(y_test, rf_pred, average='macro'))
    print(f"[Task 2 & 3] Random Forest trained & logged. Run ID: {run_rf.info.run_id}")

    # ==========================================
    # TASK 4: Artifact Logging
    # ==========================================
    # 1. Confusion Matrix
    plt.figure(figsize=(6,4))
    cm = confusion_matrix(y_test, rf_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title("Random Forest Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    cm_path = "plots/confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    # 2. Performance Comparison
    plt.figure(figsize=(6,4))
    models = ['Logistic Regression', 'Random Forest']
    accuracies = [accuracy_score(y_test, lr_pred), accuracy_score(y_test, rf_pred)]
    sns.barplot(x=models, y=accuracies, hue=models, palette='viridis', legend=False)
    plt.title("Model Accuracy Comparison")
    plt.ylim(0.0, 1.1) 
    comp_path = "plots/performance_comparison.png"
    plt.savefig(comp_path)
    plt.close()

    mlflow.log_artifact(cm_path)
    mlflow.log_artifact(comp_path)
    print("[Task 4] Visual artifacts (plots) logged successfully.")

    # ==========================================
    # TASK 5: Model Logging
    # ==========================================
    mlflow.sklearn.log_model(rf_model, "random_forest_model")
    print("[Task 5] Model logged successfully to artifacts.")

    # ==========================================
    # TASK 7: Model Registration
    # ==========================================
    model_uri = f"runs:/{run_rf.info.run_id}/random_forest_model"
    mlflow.register_model(model_uri=model_uri, name="MLflow Iris Classifier")
    print("[Task 7] Model registered to Model Registry as 'MLflow Iris Classifier'.")

# ==========================================
# TASK 6: Payload Validation execution
# ==========================================
print("\n[Task 6] Testing Payload Validation for Inference...")
new_data = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]])

if validate_payload(new_data):
    print("         Payload is valid! Proceeding to inference...")
    prediction = rf_model.predict(new_data)
    print(f"         Prediction output class: {prediction} (Target name: {iris.target_names[prediction[0]]})")

print("\n Pipeline Execution Complete!")