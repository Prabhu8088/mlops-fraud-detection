import joblib

MODEL_PATH = "model/model.pkl"

model = joblib.load(MODEL_PATH)

print("Fraud model loaded successfully")
