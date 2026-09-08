"""Random Forest with temporal validation and explicit leakage control."""

from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "enriched_data.csv"
IMPORTANCE_FILE = ROOT / "outputs" / "random_forest_feature_importance.csv"

df = pd.read_csv(INPUT_FILE, low_memory=False)
df.columns = df.columns.str.strip().str.upper()

df["ORDER_DATE"] = pd.to_datetime(df["ORDER_DATE"], errors="coerce")
df = df.dropna(subset=["ORDER_DATE"]).copy()

df["ORDER_MONTH"] = df["ORDER_DATE"].dt.month
df["ORDER_WEEKDAY"] = df["ORDER_DATE"].dt.weekday
df["FAILURE_FLAG_LINE"] = (df["MISSING_QUANTITY"] > 0).astype(int)

promo_map = {
    "Y": 1, "N": 0,
    "YES": 1, "NO": 0,
    "TRUE": 1, "FALSE": 0,
    "1": 1, "0": 0,
}
df["PROMO_ONLY_BIN"] = (
    df["PROMO_ONLY"]
    .astype(str)
    .str.strip()
    .str.upper()
    .map(promo_map)
)

# Temporal split: 2023-2024 for training, 2025 for testing.
train = df[
    (df["ORDER_DATE"] >= "2023-01-01")
    & (df["ORDER_DATE"] <= "2024-12-31")
].copy()
test = df[
    (df["ORDER_DATE"] >= "2025-01-01")
    & (df["ORDER_DATE"] <= "2025-12-31")
].copy()

features = [
    "ORDER_QUANTITY",
    "TOTAL_ORDER_ITEMS",
    "ORDER_MONTH",
    "ORDER_WEEKDAY",
    "PROMO_ONLY_BIN",
    "GROUP_DUMMY",
    "CATEGORY_DUMMY",
    "VENDOR_DUMMY",
]
target = "FAILURE_FLAG_LINE"
categorical = ["GROUP_DUMMY", "CATEGORY_DUMMY", "VENDOR_DUMMY"]

train_model = pd.get_dummies(
    train[features + [target]],
    columns=categorical,
    drop_first=True,
)
test_model = pd.get_dummies(
    test[features + [target]],
    columns=categorical,
    drop_first=True,
)

y_train = train_model.pop(target).astype(int)
y_test = test_model.pop(target).astype(int)

X_train, X_test = train_model.align(
    test_model,
    join="left",
    axis=1,
    fill_value=0,
)
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# Post-event or outcome-derived variables are deliberately excluded:
# MISSING_QUANTITY, MISSING_ORDER_ITEMS, MISSING_RATE_LINE,
# MAT_AV_DATE and DAYS_TO_AVAILABILITY.
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("PR AUC:", average_precision_score(y_test, y_prob))
print("Positive-class baseline:", y_test.mean())
print(classification_report(y_test, y_pred, zero_division=0))
print(confusion_matrix(y_test, y_pred))

feature_importance = (
    pd.DataFrame(
        {
            "Feature": X_train.columns,
            "Importance": model.feature_importances_,
        }
    )
    .sort_values("Importance", ascending=False)
)

IMPORTANCE_FILE.parent.mkdir(parents=True, exist_ok=True)
feature_importance.to_csv(IMPORTANCE_FILE, index=False)
