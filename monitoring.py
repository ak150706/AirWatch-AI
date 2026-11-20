
"""AirWatch AI: model monitoring, evaluation and retraining."""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from airwatch.anomaly import neighbour_median
from airwatch.forecast import PARAMS, _X
from airwatch.simulator import make_network


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

HORIZON = 6

FEATURE_NAMES = [
    "pm25_lag_0", "pm25_lag_1", "pm25_lag_2",
    "pm25_lag_3", "pm25_lag_6", "pm25_lag_12",
    "pm25_lag_24", "pm25_mean_6h", "neighbour_pm25",
    "log_pm25_scale", "temperature", "humidity",
    "hour_sin", "hour_cos"
]


# TASK A: Generate and validate labelled data
def make_examples(seed, start, stop):
    net = make_network(seed=seed)
    nb = neighbour_median(net, net.pm25)

    t = np.arange(start, stop)
    X, base = _X(net.pm25, nb, net.temp, net.hum, t)

    y = net.pm25[:, t + HORIZON].reshape(-1) / base

    valid = (
        np.isfinite(X).all(axis=1)
        & np.isfinite(y)
        & np.isfinite(base)
        & (base > 0)
    )

    invalid_count = int(len(y) - valid.sum())

    return X[valid], y[valid], base[valid], invalid_count


def validate_data(name, X, y, invalid_count):
    if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"{name}: Required feature columns missing")

    if len(X) == 0 or len(X) != len(y):
        raise ValueError(f"{name}: Invalid dataset size")

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"{name}: Missing values detected")

    duplicates = len(X) - len(np.unique(X, axis=0))

    print(f"\n{name}")
    print("Dataset size:", len(X))
    print("Feature columns:", X.shape[1])
    print("Invalid rows removed:", invalid_count)
    print("Missing values after cleaning:", 0)
    print("Duplicate feature records:", duplicates)


# TASK B: Evaluate model
def calculate_metrics(actual, predicted):
    return {
        "MAE": round(float(mean_absolute_error(actual, predicted)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(actual, predicted))), 4),
        "R2": round(float(r2_score(actual, predicted)), 4)
    }


# TASK C: Train candidate model
def train_model(X, y):
    model = HistGradientBoostingRegressor(**PARAMS)
    model.fit(X, y)
    return model


def main():

    print("=" * 55)
    print("AIRWATCH AI - MODEL MONITORING AND RETRAINING")
    print("=" * 55)

    # Existing training data
    old_X, old_y, _, old_invalid = make_examples(
        seed=7, start=24, stop=402
    )

    # New incoming labelled data
    new_X, new_y, _, new_invalid = make_examples(
        seed=42, start=24, stop=456
    )

    # Separate labelled evaluation data
    test_X, test_y, test_base, test_invalid = make_examples(
        seed=99, start=24, stop=456
    )

    # Validate all datasets
    validate_data("OLD TRAINING DATA", old_X, old_y, old_invalid)
    validate_data("NEW INCOMING DATA", new_X, new_y, new_invalid)
    validate_data("EVALUATION DATA", test_X, test_y, test_invalid)

    print("\nIncoming target distribution:")
    print("Mean:", round(float(new_y.mean()), 4))
    print("Standard deviation:", round(float(new_y.std()), 4))
    print("Minimum:", round(float(new_y.min()), 4))
    print("Maximum:", round(float(new_y.max()), 4))

    # Load or initialize existing champion
    champion_path = ARTIFACTS / "champion_model.joblib"

    if not champion_path.exists():
        initial_model = train_model(old_X, old_y)

        joblib.dump({
            "model": initial_model,
            "feature_names": FEATURE_NAMES,
            "horizon_hours": HORIZON
        }, champion_path)

        print("\nInitial champion model created.")

    champion_bundle = joblib.load(champion_path)
    champion = champion_bundle["model"]

    # Evaluate existing champion
    actual = test_y * test_base
    champion_predictions = champion.predict(test_X) * test_base

    champion_metrics = calculate_metrics(
        actual, champion_predictions
    )

    # Combine old and new labelled data
    combined_X = np.vstack((old_X, new_X))
    combined_y = np.concatenate((old_y, new_y))

    print("\nCombined training records:", len(combined_X))

    # Train separate candidate
    candidate = train_model(combined_X, combined_y)

    candidate_path = ARTIFACTS / "candidate_model.joblib"

    joblib.dump({
        "model": candidate,
        "feature_names": FEATURE_NAMES,
        "horizon_hours": HORIZON
    }, candidate_path)

    candidate_predictions = candidate.predict(test_X) * test_base

    candidate_metrics = calculate_metrics(
        actual, candidate_predictions
    )

    # TASK D: Quality gate
    eligible = (
        candidate_metrics["MAE"]
        <= champion_metrics["MAE"] * 1.02
    )

    decision = (
        "ELIGIBLE_FOR_PROMOTION"
        if eligible
        else "RETAIN_EXISTING_CHAMPION"
    )

    # Evaluation report
    report = {
        "project": "AirWatch AI",
        "task": "Six-hour PM2.5 regression monitoring",
        "evaluation_records": int(len(actual)),
        "champion_metrics": champion_metrics,
        "candidate_metrics": candidate_metrics
    }

    report_path = ARTIFACTS / "evaluation_report.json"

    report_path.write_text(
        json.dumps(report, indent=4)
    )

    # Record quality gate decision
    decision_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metric": "MAE",
        "rule": "Candidate MAE <= Champion MAE * 1.02",
        "champion_metrics": champion_metrics,
        "candidate_metrics": candidate_metrics,
        "decision": decision,
        "champion_overwritten": False
    }

    latest_path = ARTIFACTS / "quality_gate_latest.json"

    latest_path.write_text(
        json.dumps(decision_record, indent=4)
    )

    # Append decision history
    history_path = ARTIFACTS / "quality_gate_decisions.jsonl"

    with history_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(decision_record) + "\n")

    # Display results
    print("\n" + "=" * 55)
    print("MODEL EVALUATION RESULTS")
    print("=" * 55)

    print("\nCHAMPION MODEL:")
    print(json.dumps(champion_metrics, indent=4))

    print("\nCANDIDATE MODEL:")
    print(json.dumps(candidate_metrics, indent=4))

    print("\nQUALITY GATE DECISION:", decision)
    print("Champion overwritten: False")

    print("\nGenerated artifacts:")
    print("Candidate:", candidate_path)
    print("Evaluation report:", report_path)
    print("Decision record:", latest_path)

    print("\nMONITORING WORKFLOW COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
