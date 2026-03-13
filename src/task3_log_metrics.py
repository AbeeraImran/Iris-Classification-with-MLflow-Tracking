import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score

# Setup
mlflow.set_experiment("Iris_Classification_Experiment")

# Load Data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

print("--- Starting Task 3: Logging Parameters & Metrics ---")

# 1. Logistic Regression
with mlflow.start_run(run_name="Logistic_Regression_Task3"):
    # Define 3 parameters
    lr_params = {"C": 1.0, "max_iter": 200, "solver": "lbfgs"}
    
    # Log parameters
    for param_name, param_value in lr_params.items():
        mlflow.log_param(param_name, param_value)
    
    # Train
    lr_model = LogisticRegression(**lr_params)
    lr_model.fit(X_train, y_train)
    
    # Predict & Calculate 3 Metrics (Iris is multi-class, so we use macro average)
    y_pred = lr_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    prec = precision_score(y_test, y_pred, average='macro')
    
    # Log metrics
    mlflow.log_metric("Accuracy", acc)
    mlflow.log_metric("F1_Score", f1)
    mlflow.log_metric("Precision", prec)
    
    print(" Logistic Regression parameters and metrics logged!")

# 2. Random Forest
with mlflow.start_run(run_name="Random_Forest_Task3"):
    # Define 3 parameters
    rf_params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
    
    # Log parameters
    for param_name, param_value in rf_params.items():
        mlflow.log_param(param_name, param_value)
    
    # Train
    rf_model = RandomForestClassifier(**rf_params)
    rf_model.fit(X_train, y_train)
    
    # Predict & Calculate Metrics
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    prec = precision_score(y_test, y_pred, average='macro')
    
    # Log metrics
    mlflow.log_metric("Accuracy", acc)
    mlflow.log_metric("F1_Score", f1)
    mlflow.log_metric("Precision", prec)
    
    print(" Random Forest parameters and metrics logged!")

print("\nTask 3 complete! Launch MLflow UI to see the results.")