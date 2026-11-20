
import os

from ml_lifecycle.prepare_data import prepare_data
from ml_lifecycle.evaluate_model import evaluate_model
from ml_lifecycle.retrain import retrain
from ml_lifecycle.quality_gate import quality_gate


def run_workflow():

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("\n================================")
    print("       AIRWATCH AI LIFECYCLE")
    print("================================")

    # TASK A
    print("\nTASK A: PREPARE DATA")

    train = prepare_data("data/train.csv")

    incoming = prepare_data(
        "data/incoming_batch.csv"
    )

    validation = prepare_data(
        "data/validation.csv"
    )

    # TASK B
    print("\nTASK B: EVALUATE CHAMPION")

    evaluate_model(
        "data/validation.csv",
        "models/champion.joblib",
        "reports/champion_evaluation.json"
    )

    # TASK C
    print("\nTASK C: RETRAIN CANDIDATE")

    retrain()

    # TASK D
    print("\nTASK D: QUALITY GATE")

    result = quality_gate()

    print("\n================================")
    print("       WORKFLOW COMPLETED")
    print("================================")

    print("Decision:", result["decision"])

    print("\nGenerated reports:")
    print("reports/champion_evaluation.json")
    print("reports/candidate_evaluation.json")
    print("reports/quality_gate.json")


if __name__ == "__main__":
    run_workflow()
