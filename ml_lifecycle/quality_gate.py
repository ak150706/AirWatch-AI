
import json
from datetime import datetime, timezone

from ml_lifecycle.evaluate_model import evaluate_model


def quality_gate():

    print("\nComparing champion and candidate")

    # Evaluate champion
    champion = evaluate_model(
        "data/validation.csv",
        "models/champion.joblib",
        "reports/champion_evaluation.json"
    )

    # Evaluate candidate on exactly the same data
    candidate = evaluate_model(
        "data/validation.csv",
        "models/candidate.joblib",
        "reports/candidate_evaluation.json"
    )

    # Promotion criteria
    eligible = (
        candidate["f1_score"] >= 0.90
        and candidate["f1_score"] >= champion["f1_score"]
        and candidate["recall"] >= champion["recall"]
    )

    if eligible:
        decision = "PROMOTION_ELIGIBLE"
        reason = "Candidate passed all quality criteria"
    else:
        decision = "RETAIN_CHAMPION"
        reason = "Candidate failed one or more quality criteria"

    result = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "champion_metrics": champion,

        "candidate_metrics": candidate,

        "criteria": {
            "minimum_f1": 0.90,
            "candidate_f1_must_not_decrease": True,
            "candidate_recall_must_not_decrease": True
        },

        "decision": decision,

        "reason": reason,

        "champion_overwritten": False
    }

    with open(
        "reports/quality_gate.json",
        "w"
    ) as file:
        json.dump(result, file, indent=4)

    print("\n========== QUALITY GATE ==========")

    print("Champion F1:", champion["f1_score"])
    print("Candidate F1:", candidate["f1_score"])

    print("Champion Recall:", champion["recall"])
    print("Candidate Recall:", candidate["recall"])

    print("\nDecision:", decision)

    print("Reason:", reason)

    print("Champion overwritten: NO")

    print("\nDecision saved to reports/quality_gate.json")

    return result
