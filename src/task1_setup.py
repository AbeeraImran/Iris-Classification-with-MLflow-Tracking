import mlflow

#  Set experiment name
experiment_name = "Iris_Classification_Experiment"
mlflow.set_experiment(experiment_name)

# Start an MLflow run
with mlflow.start_run():
    # print the Run ID
    run_id = mlflow.active_run().info.run_id
    print("--- Task 1 Complete ---")
    print(f"Experiment Name: {experiment_name}")
    print(f"Run ID: {run_id}")