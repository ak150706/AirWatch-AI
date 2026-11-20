
import os
import joblib
import pandas as pd

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

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/train.csv")

X = df[FEATURES]
y = df["label"].astype(int)

model = Pipeline([
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

model.fit(X, y)

joblib.dump(
    model,
    "models/champion.joblib"
)

print("Initial champion model created!")
print("Saved at models/champion.joblib")
