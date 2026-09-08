"""Temporal logistic regression for identifying future critical materials."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "enriched_data.csv"
OUTPUT_DIR = ROOT / "outputs"

df = pd.read_csv(INPUT_FILE, low_memory=False)
df.columns = df.columns.str.strip().str.upper()

df["ORDER_DATE"] = pd.to_datetime(df["ORDER_DATE"], errors="coerce")
df["ORDER_QUANTITY"] = pd.to_numeric(df["ORDER_QUANTITY"], errors="coerce")
df["MISSING_QUANTITY"] = pd.to_numeric(
    df["MISSING_QUANTITY"], errors="coerce"
)
df["YEAR"] = df["ORDER_DATE"].dt.year

df = df[df["YEAR"].isin([2023, 2024, 2025])].dropna(
    subset=[
        "MATERIAL_DUMMY",
        "ORDER_QUANTITY",
        "MISSING_QUANTITY",
        "YEAR",
    ]
).copy()
df = df[df["ORDER_QUANTITY"] >= 0].copy()

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
    .fillna(0)
    .astype(int)
)

material_attributes = (
    df[
        [
            "MATERIAL_DUMMY",
            "CATEGORY_DUMMY",
            "GROUP_DUMMY",
            "PROMO_ONLY_BIN",
        ]
    ]
    .drop_duplicates(subset=["MATERIAL_DUMMY"])
)


def create_material_summary(data: pd.DataFrame, year: int) -> pd.DataFrame:
    """Aggregate ordered and missing quantities by material for one year."""
    return (
        data[data["YEAR"] == year]
        .groupby("MATERIAL_DUMMY")
        .agg(
            order_quantity=("ORDER_QUANTITY", "sum"),
            missing_quantity=("MISSING_QUANTITY", "sum"),
        )
        .reset_index()
    )


def create_critical_flag(material_df: pd.DataFrame) -> pd.DataFrame:
    """Flag the materials required to reach at least 80% of missing units."""
    result = material_df.sort_values(
        "missing_quantity", ascending=False
    ).reset_index(drop=True)

    total_missing = result["missing_quantity"].sum()
    result["cum_missing"] = result["missing_quantity"].cumsum()
    result["cum_pct_missing"] = np.where(
        total_missing > 0,
        result["cum_missing"] / total_missing,
        0.0,
    )
    result["critical_flag"] = 0

    reached_80 = result["cum_pct_missing"] >= 0.80
    if reached_80.any():
        cutoff = np.argmax(reached_80.to_numpy())
        result.loc[:cutoff, "critical_flag"] = 1

    return result


materials_2023 = create_material_summary(df, 2023)
materials_2024 = create_material_summary(df, 2024)
materials_2025 = create_material_summary(df, 2025)

target_2024 = create_critical_flag(materials_2024)[
    ["MATERIAL_DUMMY", "critical_flag"]
].rename(columns={"critical_flag": "critical_flag_2024"})

target_2025 = create_critical_flag(materials_2025)[
    ["MATERIAL_DUMMY", "critical_flag"]
].rename(columns={"critical_flag": "critical_flag_2025"})


def build_features(material_summary: pd.DataFrame) -> pd.DataFrame:
    features = material_summary[
        ["MATERIAL_DUMMY", "order_quantity"]
    ].copy()
    features["log_order_quantity"] = np.log1p(
        features["order_quantity"]
    )
    return features.merge(
        material_attributes,
        on="MATERIAL_DUMMY",
        how="left",
    )


features_2023 = build_features(materials_2023)
features_2024 = build_features(materials_2024)

# Temporal validation:
# 2023 features -> 2024 criticality
# 2024 features -> 2025 criticality
train = features_2023.merge(
    target_2024,
    on="MATERIAL_DUMMY",
    how="inner",
).rename(columns={"critical_flag_2024": "critical_flag"})

test = features_2024.merge(
    target_2025,
    on="MATERIAL_DUMMY",
    how="inner",
).rename(columns={"critical_flag_2025": "critical_flag"})

categorical = ["CATEGORY_DUMMY", "GROUP_DUMMY"]
train = pd.get_dummies(train, columns=categorical, drop_first=True)
test = pd.get_dummies(test, columns=categorical, drop_first=True)
train, test = train.align(test, join="left", axis=1, fill_value=0)

dummy_features = [
    col
    for col in train.columns
    if col.startswith("CATEGORY_DUMMY_")
    or col.startswith("GROUP_DUMMY_")
]
feature_cols = (
    ["log_order_quantity"]
    + dummy_features
    + ["PROMO_ONLY_BIN"]
)

X_train = train[feature_cols].fillna(0).astype(float)
y_train = train["critical_flag"].astype(int)
X_test = test[feature_cols].fillna(0).astype(float)
y_test = test["critical_flag"].astype(int)

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1-score:", f1_score(y_test, y_pred, zero_division=0))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("PR AUC:", average_precision_score(y_test, y_prob))
print("Positive-class baseline:", y_test.mean())
print(classification_report(y_test, y_pred, zero_division=0))
print(confusion_matrix(y_test, y_pred))

coefficients = (
    pd.DataFrame(
        {"Variable": feature_cols, "Coefficient": model.coef_[0]}
    )
    .sort_values("Coefficient", ascending=False)
)

audit_2024 = create_critical_flag(materials_2024.copy())
audit_2025 = create_critical_flag(materials_2025.copy())

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
coefficients.to_csv(
    OUTPUT_DIR / "logistic_regression_2_coefficients.csv",
    index=False,
)
audit_2024.to_csv(
    OUTPUT_DIR / "critical_flag_audit_2024.csv",
    index=False,
)
audit_2025.to_csv(
    OUTPUT_DIR / "critical_flag_audit_2025.csv",
    index=False,
)
