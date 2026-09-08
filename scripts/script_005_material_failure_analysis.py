"""Analyze failure concentration by anonymized material identifier."""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "clean_data.csv"
OUTPUT_FILE = ROOT / "outputs" / "material_analysis.csv"
PARETO_FILE = ROOT / "outputs" / "material_pareto_80.csv"

df = pd.read_csv(INPUT_FILE, low_memory=False)
df.columns = df.columns.str.strip().str.upper()

for col in ["ORDER_QUANTITY", "MISSING_QUANTITY"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(
    subset=["MATERIAL_DUMMY", "ORDER_QUANTITY", "MISSING_QUANTITY"]
).copy()
df = df[df["ORDER_QUANTITY"] > 0].copy()
df["FAILURE_FLAG_LINE"] = (df["MISSING_QUANTITY"] > 0).astype(int)

material_summary = (
    df.groupby("MATERIAL_DUMMY")
    .agg(
        order_quantity=("ORDER_QUANTITY", "sum"),
        missing_quantity=("MISSING_QUANTITY", "sum"),
        total_lines=("MATERIAL_DUMMY", "size"),
        failure_lines=("FAILURE_FLAG_LINE", "sum"),
        orders=("ORDERS_DUMMY", "nunique"),
    )
    .reset_index()
)

material_summary["missing_rate"] = (
    material_summary["missing_quantity"] / material_summary["order_quantity"]
)
material_summary["failure_line_rate"] = (
    material_summary["failure_lines"] / material_summary["total_lines"]
)

total_order = material_summary["order_quantity"].sum()
total_missing = material_summary["missing_quantity"].sum()

material_summary["volume_share"] = (
    material_summary["order_quantity"] / total_order
)
material_summary["failure_share"] = np.where(
    total_missing > 0,
    material_summary["missing_quantity"] / total_missing,
    0,
)
material_summary["failure_exposure_ratio"] = np.where(
    material_summary["volume_share"] > 0,
    material_summary["failure_share"] / material_summary["volume_share"],
    np.nan,
)
material_summary["share_difference"] = (
    material_summary["failure_share"] - material_summary["volume_share"]
)

material_pareto = material_summary.sort_values(
    "missing_quantity", ascending=False
).copy()
material_pareto["cum_failure_share"] = material_pareto[
    "failure_share"
].cumsum()

reached_80 = material_pareto["cum_failure_share"] >= 0.80
if reached_80.any():
    cutoff = np.argmax(reached_80.to_numpy())
    materials_80 = material_pareto.iloc[: cutoff + 1].copy()
else:
    materials_80 = material_pareto.copy()

print("Total materials:", material_summary["MATERIAL_DUMMY"].nunique())
print("Materials required for ~80% of failures:", len(materials_80))
print(
    "Share of materials:",
    len(materials_80) / material_summary["MATERIAL_DUMMY"].nunique(),
)
print("Accumulated failure share:", materials_80["failure_share"].sum())

# Exposure sensitivity used in the study.
for threshold in [0.0001, 0.0005, 0.0010]:
    subset = material_summary[
        material_summary["volume_share"] >= threshold
    ]
    print(
        f"Exposure >= {threshold:.4f}: "
        f"materials={len(subset)}, "
        f"volume_coverage={subset['volume_share'].sum():.4f}, "
        f"failure_coverage={subset['failure_share'].sum():.4f}"
    )

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
material_summary.to_csv(OUTPUT_FILE, index=False)
materials_80.to_csv(PARETO_FILE, index=False)
