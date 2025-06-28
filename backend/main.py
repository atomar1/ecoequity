from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware


# Load expected feature names (trained order)
with open("backend/feature_names.txt") as f:
    expected_features = [line.strip() for line in f.readlines()]

# Load model and scaler
model = joblib.load("backend/model.pkl")
scaler = joblib.load("backend/scaler.pkl")

# Define FastAPI app
app = FastAPI(title="CalEnviroScreen Disadvantaged Community Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify ["http://localhost:3000"] for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the expected input schema
class InputFeatures(BaseModel):
    Poverty: float
    Unemployment: float
    PM25: float
    Ozone: float
    Diesel_PM: float
    Drinking_Water: float
    Asthma: float
    Low_Birth_Weight: float
    Traffic: float
    Linguistic_Isolation: float

@app.get("/")
def root():
    return {"message": "CalEnviroScreen Predictor API is running."}

@app.post("/predict")
def predict(data: InputFeatures):
    f = data.dict()

    # Step 1: Base inputs
    base_features = {
        "Poverty": f["Poverty"],
        "Unemployment": f["Unemployment"],
        "PM2.5": f["PM25"],
        "Ozone": f["Ozone"],
        "Diesel PM": f["Diesel_PM"],
        "Drinking Water": f["Drinking_Water"],
        "Asthma": f["Asthma"],
        "Low Birth Weight": f["Low_Birth_Weight"],
        "Traffic": f["Traffic"],
        "Linguistic Isolation": f["Linguistic_Isolation"],
    }

    # Step 2: Interactions
    interactions = {}
    keys = list(base_features.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k = f"{keys[i]}_x_{keys[j]}"
            interactions[k] = base_features[keys[i]] * base_features[keys[j]]

    # Step 3: Polynomial terms
    poly_terms = {
        "Poverty^2": base_features["Poverty"] ** 2,
        "Unemployment^2": base_features["Unemployment"] ** 2,
        "Poverty Unemployment": base_features["Poverty"] * base_features["Unemployment"],
    }

    # Step 4: Combine all inputs
    all_inputs = {**base_features, **interactions, **poly_terms}

    # Step 5: Create DataFrame in the right column order
    row_df = pd.DataFrame([[all_inputs.get(col, 0) for col in expected_features]], columns=expected_features)

    # Step 6: Scale and predict
    print("Expected features:", expected_features)
    print("Input features:", list(row_df.columns))

    row_scaled = scaler.transform(row_df)
    pred = model.predict(row_scaled)[0]
    prob = model.predict_proba(row_scaled)[0][1]

    return {
        "prediction": int(pred),
        "confidence": round(float(prob), 4)
    }