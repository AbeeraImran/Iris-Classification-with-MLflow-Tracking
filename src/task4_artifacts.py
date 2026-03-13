import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Setup
mlflow.set_experiment("Iris_Classification_Experiment")
# Create a folder to temporarily hold our plot images
os.makedirs("plots", exist_ok=True)

# Load Data & Train models quickly
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

lr = LogisticRegression(max_iter=200).fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

print("Generating plots...")

# 1. Create Confusion Matrix Plot (using Random Forest predictions)
plt.figure(figsize=(6,4))
cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title("Random Forest Confusion Matrix")
plt.ylabel('Actual')
plt.xlabel('Predicted')
cm_path = "plots/confusion_matrix.png"
plt.savefig(cm_path)
plt.close()

# 2. Create Performance Comparison Plot
plt.figure(figsize=(6,4))
models = ['Logistic Regression', 'Random Forest']
accuracies = [accuracy_score(y_test, lr_pred), accuracy_score(y_test, rf_pred)]
sns.barplot(x=models, y=accuracies, hue=models, palette='viridis', legend=False)
plt.title("Model Accuracy Comparison")
plt.ylim(0.0, 1.1) 
comp_path = "plots/performance_comparison.png"
plt.savefig(comp_path)
plt.close()

# Log Artifacts to MLflow (This is the core requirement for Task 4)
with mlflow.start_run(run_name="Artifact_Logging_Task4"):
    mlflow.log_artifact(cm_path)
    mlflow.log_artifact(comp_path)
    print("✅ Plots generated and logged as artifacts!")

print("\nTask 4 complete! Run `mlflow ui` and check the Artifacts tab.")