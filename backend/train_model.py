import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("data/calenviroscreen-3.0-results-june-2018-update.csv")

print("Available columns:")
print(df.columns.tolist())
exit()
