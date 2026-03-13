## Iris Classification with MLflow Tracking
This repository contains a complete MLOps pipeline for classifying the Iris dataset using Scikit-Learn and MLflow. The project demonstrates experiment tracking, model versioning, and automated payload validation.

## Project Features
# Experiment Tracking: Logs hyper-parameters and metrics (Accuracy, F1, Precision) for multiple models.

# Artifact Logging: Automatically generates and saves Confusion Matrices and Performance Comparison plots.

# Model Registry: Registers the best-performing model to the MLflow Model Registry for version control.

# Payload Validation: Includes a robust pre-inference check to ensure data integrity (shape and null-value checks).

## Tech Stack

Language: Python 3.x
ML Library: Scikit-Learn
Ops Tool: MLflow
Visualization: Matplotlib, Seaborn

 
## How to Run

# **Clone the repository:**
   bash
   git clone [https://github.com/AbeeraImran/Iris-Classification-with-MLflow-Tracking.git](https://github.com/AbeeraImran/Iris-Classification-with-MLflow-Tracking.git)
   cd mlflow_task

# Activate your environment:
Bash
.\venv\Scripts\activate

# Execute the pipeline:
Bash
python src/mlflow_pipeline.py

# View the Results:
Launch the MLflow UI to see the logged runs and artifacts:
Bash
mlflow ui
Open http://127.0.0.1:5000 in your browser.

## Summary
Both the Logistic Regression and Random Forest models achieved an accuracy of 1.0 on the test set. The models are logged and versioned within the MLflow dashboard for easy comparison.
