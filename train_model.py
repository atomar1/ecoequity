# prompt: give me the complete code only for the best model that i can run in a file train_model.py in my ide

import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import classification_report

# Path to dataset
DATA_PATH = "calenviroscreen-3.0-results-june-2018-update.csv"

# Load dataset
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: Dataset not found at {DATA_PATH}")
    exit()

# Target column: convert Yes/No to 1/0
df = df.dropna(subset=["SB 535 Disadvantaged Community"])
df["SB 535 Disadvantaged Community"] = df["SB 535 Disadvantaged Community"].map({"Yes": 1, "No": 0}).astype(int)

# Select base features
initial_features = [
    "Poverty", "Unemployment", "PM2.5", "Ozone", "Diesel PM", "Drinking Water",
    "Asthma", "Low Birth Weight", "Traffic", "Linguistic Isolation"
]
df = df.dropna(subset=initial_features)

# Create interaction features
interaction_features = []
for i in range(len(initial_features)):
    for j in range(i + 1, len(initial_features)):
        name = f"{initial_features[i]}_x_{initial_features[j]}"
        df[name] = df[initial_features[i]] * df[initial_features[j]]
        interaction_features.append(name)

# Add polynomial features (degree=2) for a subset
poly_cols = ["Poverty", "Unemployment"]
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[poly_cols])
poly_names = poly.get_feature_names_out(poly_cols)

# Drop original columns from polynomial output to avoid duplication
poly_df = pd.DataFrame(poly_features, columns=poly_names, index=df.index).drop(columns=poly_cols)
df = pd.concat([df, poly_df], axis=1)

# Final features list
all_features = initial_features + interaction_features + list(poly_df.columns)

# Build X and y
X = df[all_features]
y = df["SB 535 Disadvantaged Community"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Scale features and keep column names
scaler = StandardScaler()
scaler.set_output(transform="pandas")  # Ensure output has DataFrame with column names
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the best model
best_model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=2, random_state=42)
best_model.fit(X_train_scaled, y_train)

# Evaluate
print("Best Model (RandomForestClassifier) Report:")
print(classification_report(y_test, best_model.predict(X_test_scaled)))

# Save model, scaler, and feature names
os.makedirs("backend", exist_ok=True)
joblib.dump(best_model, "backend/model.pkl")
joblib.dump(scaler, "backend/scaler.pkl")

# Save ordered feature names for backend prediction
feature_list_path = "backend/feature_names.txt"
with open(feature_list_path, "w") as f:
    for name in all_features:
        f.write(name + "\n")
print(f"\nSaved {len(all_features)} feature names to '{feature_list_path}'")

# Cross-validation on full dataset
print("\nPerforming cross-validation on the entire dataset:")
X_scaled_full = scaler.transform(X)
cv_scores = cross_val_score(best_model, X_scaled_full, y, cv=5, scoring='f1', n_jobs=-1)
print("Cross-validation F1 scores for each fold:", cv_scores)
print("Mean F1 score:", cv_scores.mean())
print("Standard deviation of F1 scores:", cv_scores.std())
