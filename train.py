import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_FILE = "fraud_data.csv"
MODEL_FILE = "model.pkl"
METADATA_FILE = "model_metadata.json"


FEATURES = [
    "amount",
    "transaction_hour",
    "merchant_category",
    "distance_from_home",
    "previous_transactions"
]

TARGET = "fraud"


df = pd.read_csv(DATA_FILE)

X = df[FEATURES]
y = df[TARGET]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


numeric_features = [
    "amount",
    "transaction_hour",
    "distance_from_home",
    "previous_transactions"
]

categorical_features = [
    "merchant_category"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            "passthrough",
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)


model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ]
)


print("Training model...")

model.fit(X_train, y_train)


predictions = model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)


print("\nModel evaluation")
print("----------------")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nClassification report:")
print(classification_report(y_test, predictions))


joblib.dump(model, MODEL_FILE)


metadata = {
    "model_name": "fraud-detection",
    "model_version": "1.0.0",
    "framework": "scikit-learn",
    "model_file": MODEL_FILE,
    "features": FEATURES,
    "target": TARGET,
    "accuracy": round(float(accuracy), 4),
    "f1_score": round(float(f1), 4)
}


with open(METADATA_FILE, "w") as f:
    json.dump(
        metadata,
        f,
        indent=2
    )


print("\nModel saved:")
print(MODEL_FILE)

print("\nMetadata saved:")
print(METADATA_FILE)
