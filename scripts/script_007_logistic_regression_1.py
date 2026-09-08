"""Baseline logistic regression using operational timing and order volume."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "clean_data.csv"

df = pd.read_csv(INPUT_FILE, low_memory=False)
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
df["mat_av_date"] = pd.to_datetime(df["mat_av_date"], errors="coerce")

df["days_to_availability"] = (
    df["mat_av_date"] - df["order_date"]
).dt.days
df = df[df["days_to_availability"] >= 0].copy()

df["failure_flag"] = (df["missing_quantity"] > 0).astype(int)
df["days_squared"] = df["days_to_availability"] ** 2
df["log_order_quantity"] = np.log1p(df["order_quantity"])

features = [
    "days_to_availability",
    "days_squared",
    "log_order_quantity",
]
X = df[features].fillna(0)
y = df["failure_flag"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, zero_division=0))
print(confusion_matrix(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

coefficients = pd.DataFrame(
    {"Variable": features, "Coefficient": model.coef_[0]}
)
print(coefficients)
