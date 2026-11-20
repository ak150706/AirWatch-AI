
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

FEATURES = [
    "pm25",
    "pm10",
    "no2",
    "temp",
    "hum"
]


def retrain():

    print("\nLoading existing training data")

    old_data = pd.read_csv("data/train.csv")

    print("Loading incoming labelled batch")

    new_data = pd.read_csv("data/incoming_batch.csv")

    # Combine old and new labelled data
    combined = pd.concat(
        [old_data, new_data],
        ignore_index=True
    ).drop_duplicates()

    X = combined[FEATURES]
    y = combined["label"].astype(int)

    print("Combined dataset size:", combined.shape)

    pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            )
        )
    ])

    mlflow.set_experiment("AirWatch-Retraining")

    with mlflow.start_run():

        pipeline.fit(X, y)

        os.makedirs("models", exist_ok=True)

        joblib.dump(
            pipeline,
            "models/candidate.joblib"
        )

        mlflow.log_param(
            "algorithm",
            "RandomForestClassifier"
        )

        mlflow.log_param(
            "training_records",
            len(combined)
        )

        mlflow.log_param(
            "n_estimators",
            200
        )

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="candidate_model"
        )

        print("MLflow run:", mlflow.active_run().info.run_id)

    print("\nCandidate model saved successfully")

    return pipeline
