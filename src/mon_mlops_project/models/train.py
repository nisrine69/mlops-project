import os
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from mlflow.models.signature import infer_signature


TARGET = "MedHouseVal"


def evaluate(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


def main():
    # ----------------------
    # MLflow config
    # ----------------------
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "california_housing_price_prediction")
    mlflow.set_experiment(experiment_name)

    # ----------------------
    # Load data
    # ----------------------
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()

    X = df.drop(columns=TARGET)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Input example for MLflow model signature
    input_example = X_train.iloc[:5].copy()

    # ----------------------
    # 1) Baseline: Linear Regression + StandardScaler
    # ----------------------
    with mlflow.start_run(run_name="LinearRegression"):
        model_lr = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )

        model_lr.fit(X_train, y_train)
        y_pred = model_lr.predict(X_test)

        rmse, mae, r2 = evaluate(y_test, y_pred)

        # Log params + metrics
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("scaler", "StandardScaler")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Signature
        signature = infer_signature(input_example, model_lr.predict(input_example))

        # Log model (no deprecated artifact_path)
        mlflow.sklearn.log_model(
            sk_model=model_lr,
            name="model",
            input_example=input_example,
            signature=signature,
            registered_model_name="CaliforniaHousingModel",
        )

        print(f"[LinearRegression] RMSE={rmse:.4f} | MAE={mae:.4f} | R2={r2:.4f}")

    # ----------------------
    # 2) Advanced: Random Forest
    # ----------------------
    with mlflow.start_run(run_name="RandomForest"):
        model_rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )

        model_rf.fit(X_train, y_train)
        y_pred = model_rf.predict(X_test)

        rmse, mae, r2 = evaluate(y_test, y_pred)

        # Log params + metrics
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("max_depth", 20)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Signature
        signature = infer_signature(input_example, model_rf.predict(input_example))

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model_rf,
            name="model",
            input_example=input_example,
            signature=signature,
            registered_model_name="CaliforniaHousingModel",
        )

        print(f"[RandomForest]     RMSE={rmse:.4f} | MAE={mae:.4f} | R2={r2:.4f}")


if __name__ == "__main__":
    main()
