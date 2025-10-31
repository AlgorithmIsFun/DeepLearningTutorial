import mlflow
import pandas as pd
import numpy as np
import tensorflow as tf
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
from tensorflow import keras
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
signature = None
def create_and_train_model(learning_rate, momentum, epochs=10):
    """
    Create and train a neural network with specified hyperparameters.

    Returns:
        dict: Training results including model and metrics
    """
    # Prepare Data
    data = pd.read_csv("https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-white.csv",
                       sep=";", )
    train, test = train_test_split(data, test_size=0.25, random_state=42)
    train_x = train.drop(["quality"], axis=1).values
    train_y = train[["quality"]].values.ravel()
    test_x = test.drop(["quality"], axis=1).values
    test_y = test[["quality"]].values.ravel()
    train_x, valid_x, train_y, valid_y = train_test_split(
        train_x, train_y, test_size=0.2, random_state=42
    )
    signature = infer_signature(train_x, train_y)
    # Normalize input features for better training stability
    mean = np.mean(train_x, axis=0)
    var = np.var(train_x, axis=0)

    # Define model architecture
    model = keras.Sequential(
        [
            keras.Input([train_x.shape[1]]),
            keras.layers.Normalization(mean=mean, variance=var),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.2),  # Add regularization
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1),
        ]
    )

    # Compile with specified hyperparameters
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
        loss="mean_squared_error",
        metrics=[keras.metrics.RootMeanSquaredError()],
    )

    # Train with early stopping for efficiency
    early_stopping = keras.callbacks.EarlyStopping(
        patience=3, restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        train_x,
        train_y,
        validation_data=(valid_x, valid_y),
        epochs=epochs,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=0,  # Reduce output for cleaner logs
    )

    # Evaluate on validation set
    val_loss, val_rmse = model.evaluate(valid_x, valid_y, verbose=0)

    return {
        "model": model,
        "val_rmse": val_rmse,
        "val_loss": val_loss,
        "history": history,
        "epochs_trained": len(history.history["loss"]),
    }

def objective(params):
    """
    Objective function for hyperparameter optimization.
    This function will be called by Hyperopt for each trial.
    """
    with mlflow.start_run(nested=True):
        # Log hyperparameters being tested
        mlflow.log_params(
            {
                "learning_rate": params["learning_rate"],
                "momentum": params["momentum"],
                "optimizer": "SGD",
                "architecture": "64-32-1",
            }
        )

        # Train model with current hyperparameters
        result = create_and_train_model(
            learning_rate=params["learning_rate"],
            momentum=params["momentum"],
            epochs=15,
        )

        # Log training results
        mlflow.log_metrics(
            {
                "val_rmse": result["val_rmse"],
                "val_loss": result["val_loss"],
                "epochs_trained": result["epochs_trained"],
            }
        )

        # Log the trained model
        mlflow.tensorflow.log_model(result["model"], name="model", signature=signature)

        # Log training curves as artifacts
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(result["history"].history["loss"], label="Training Loss")
        plt.plot(result["history"].history["val_loss"], label="Validation Loss")
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(
            result["history"].history["root_mean_squared_error"], label="Training RMSE"
        )
        plt.plot(
            result["history"].history["val_root_mean_squared_error"],
            label="Validation RMSE",
        )
        plt.title("Model RMSE")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.legend()

        plt.tight_layout()
        plt.savefig("training_curves.png")
        mlflow.log_artifact("training_curves.png")
        plt.close()

        # Return loss for Hyperopt (it minimizes)
        return {"loss": result["val_rmse"], "status": STATUS_OK}


def mlfow_optimizer():
    # Create or set experiment
    experiment_name = "wine-quality-optimization"
    mlflow.set_experiment(experiment_name)

    print(f"Starting hyperparameter optimization experiment: {experiment_name}")
    print("This will run 15 trials to find optimal hyperparameters...")

    with mlflow.start_run(run_name="hyperparameter-sweep"):
        # Log experiment metadata
        mlflow.log_params(
            {
                "optimization_method": "Tree-structured Parzen Estimator (TPE)",
                "max_evaluations": 15,
                "objective_metric": "validation_rmse",
                "dataset": "wine-quality",
                "model_type": "neural_network",
            }
        )
        search_space = {
            "learning_rate": hp.loguniform("learning_rate", np.log(1e-5), np.log(1e-1)),
            "momentum": hp.uniform("momentum", 0.0, 0.9),
        }

        # Run optimization
        trials = Trials()
        best_params = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=15,
            trials=trials,
            verbose=True,
        )

        # Find and log best results
        best_trial = min(trials.results, key=lambda x: x["loss"])
        best_rmse = best_trial["loss"]

        # Log optimization results
        mlflow.log_params(
            {
                "best_learning_rate": best_params["learning_rate"],
                "best_momentum": best_params["momentum"],
            }
        )
        mlflow.log_metrics(
            {
                "best_val_rmse": best_rmse,
                "total_trials": len(trials.trials),
                "optimization_completed": 1,
            }
        )

def setup_mlflow():
    mlflow.set_tracking_uri("http://localhost:8080")
    apple_experiment = mlflow.set_experiment("Apple_Models")
    run_name = "apples_rf_test"
    artifact_path = "rf_apples"
    data = generate_apple_sales_data_with_promo_adjustment(base_demand=1_000, n_rows=1_000)
    X = data.drop(columns=["date", "demand"])
    y = data["demand"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    params = {
        "n_estimators": 100,
        "max_depth": 6,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
        "bootstrap": True,
        "oob_score": False,
        "random_state": 888,
    }
    rf = RandomForestRegressor(**params)
    rf.fit(X_train, y_train) #Train Model
    y_pred = rf.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred)
    metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}
    with mlflow.start_run(run_name=run_name) as run: #Log params, metrics and model
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(sk_model=rf, input_example=X_val, name=artifact_path)

def setup_mlflowclient():
    client = MlflowClient(tracking_uri="http://127.0.0.1:8080")
    all_experiments = client.search_experiments()
    #print(all_experiments) #List of experiment objects
    default_experiment = [
        {"name": experiment.name, "lifecycle_stage": experiment.lifecycle_stage}
        for experiment in all_experiments
        if experiment.name == "Default"
    ][0]
    #Extract Experiment name and stage if name=Default
    #pprint(default_experiment)
    # Get the experiment by name
    exp = client.get_experiment_by_name("Apple_Models")

    if exp is not None:
        if exp.lifecycle_stage == "deleted":
            # Restore it
            client.restore_experiment(exp.experiment_id)
            print(f"Restored experiment: {exp.name}")
        else:
            print(f"Experiment already active: {exp.name}")
    else:
        # Create a new experiment if it does not exist
        experiment_description = (
            "This is the grocery forecasting project. "
            "This experiment contains the produce models for apples."
        )
        experiment_tags = { #These tags are for client.search_experiments()
            "project_name": "grocery-forecasting",
            "store_dept": "produce",
            "team": "stores-ml",
            "project_quarter": "Q3-2023",
            "mlflow.note.content": experiment_description,
        }
        # Create the Experiment, providing a unique name
        produce_apples_experiment = client.create_experiment(
            name="Apple_Models", tags=experiment_tags
        )
    apples_experiment = client.search_experiments(
        filter_string="tags.`project_name` = 'grocery-forecasting'"
    )

    #print(vars(apples_experiment[0]))

def generate_apple_sales_data_with_promo_adjustment(base_demand: int = 1000, n_rows: int = 5000):
    """Generates a synthetic dataset for predicting apple sales demand with seasonality
    and inflation."""
    # Set seed for reproducibility
    np.random.seed(9999)
    # Create date range
    dates = [datetime.now() - timedelta(days=i) for i in range(n_rows)]
    dates.reverse()
    # Generate features
    df = pd.DataFrame(
        {
            "date": dates,
            "average_temperature": np.random.uniform(10, 35, n_rows),
            "rainfall": np.random.exponential(5, n_rows),
            "weekend": [(date.weekday() >= 5) * 1 for date in dates],
            "holiday": np.random.choice([0, 1], n_rows, p=[0.97, 0.03]),
            "price_per_kg": np.random.uniform(0.5, 3, n_rows),
            "month": [date.month for date in dates],
        }
    )
    # Introduce inflation over time (years)
    df["inflation_multiplier"] = (
        1 + (df["date"].dt.year - df["date"].dt.year.min()) * 0.03
    )
    # Incorporate seasonality due to apple harvests
    df["harvest_effect"] = np.sin(2 * np.pi * (df["month"] - 3) / 12) + np.sin(
        2 * np.pi * (df["month"] - 9) / 12
    )
    # Modify the price_per_kg based on harvest effect
    df["price_per_kg"] = df["price_per_kg"] - df["harvest_effect"] * 0.5
    # Adjust promo periods to coincide with periods lagging peak harvest by 1 month
    peak_months = [4, 10]  # months following the peak availability
    df["promo"] = np.where(
        df["month"].isin(peak_months),
        1,
        np.random.choice([0, 1], n_rows, p=[0.85, 0.15]),
    )
    # Generate target variable based on features
    base_price_effect = -df["price_per_kg"] * 50
    seasonality_effect = df["harvest_effect"] * 50
    promo_effect = df["promo"] * 200
    df["demand"] = (
        base_demand
        + base_price_effect
        + seasonality_effect
        + promo_effect
        + df["weekend"] * 300
        + np.random.normal(0, 50, n_rows)
    ) * df[
        "inflation_multiplier"
    ]  # adding random noise
    # Add previous day's demand
    df["previous_days_demand"] = df["demand"].shift(1)
    df["previous_days_demand"].fillna(
        method="bfill", inplace=True
    )  # fill the first row

    # Drop temporary columns
    df.drop(columns=["inflation_multiplier", "harvest_effect", "month"], inplace=True)

    return df
#setup_mlflowclient()
#setup_mlflow()
mlfow_optimizer()
"""
#Deploy Model
mlflow models serve -m "models:/wine-quality-predictor/1" --port 5002

# Build Docker image
mlflow models build-docker --model-uri "models:/wine-quality-predictor/1" --name "wine-quality-api"
  
# Run the container
docker run -p 5003:8080 wine-quality-api

# Test in another terminal
curl -X POST http://localhost:5003/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_split": {
      "columns": ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"],
      "data": [[7.0, 0.27, 0.36, 20.7, 0.045, 45, 170, 1.001, 3.0, 0.45, 8.8]]
    }
  }'

#Databricks: Deploy with Mosaic AI Model Serving

# First, register your model in Unity Catalog
import mlflow

mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run():
    # Log your model to Unity Catalog
    mlflow.tensorflow.log_model(
        model,
        name="wine-quality-model",
        registered_model_name="main.default.wine_quality_predictor",
    )

# Then create a serving endpoint using the Databricks UI:
# 1. Navigate to "Serving" in the Databricks workspace
# 2. Click "Create serving endpoint"
# 3. Select your registered model from Unity Catalog
# 4. Configure compute and traffic settings
# 5. Deploy and test your endpoint
"""