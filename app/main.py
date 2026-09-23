from fastapi import FastAPI
from pydantic import BaseModel

from app.model_loader import model


app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0"
)


class Transaction(BaseModel):
    amount: float
    transaction_hour: int
    merchant_category: str
    distance_from_home: float
    previous_transactions: int


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(transaction: Transaction):

    data = [[
        transaction.amount,
        transaction.transaction_hour,
        transaction.merchant_category,
        transaction.distance_from_home,
        transaction.previous_transactions
    ]]

    import pandas as pd

    df = pd.DataFrame(
        data,
        columns=[
            "amount",
            "transaction_hour",
            "merchant_category",
            "distance_from_home",
            "previous_transactions"
        ]
    )

    prediction = int(
        model.predict(df)[0]
    )

    probability = float(
        model.predict_proba(df)[0][1]
    )

    return {
        "fraud": bool(prediction),
        "probability": round(probability, 4),
        "model_name": "fraud-detection",
        "model_version": "1.0.0"
    }
