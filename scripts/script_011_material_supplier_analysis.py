"""Analyze supplier-level failure concentration and operational exposure."""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "enriched_data.csv"
OUTPUT_DIR = ROOT / "outputs"

df = pd.read_csv(INPUT_FILE, low_memory=False)
df.columns = df.columns.str.strip().str.upper()

for col in ["ORDER_QUANTITY", "MISSING_QUANTITY"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(
    subset=["VENDOR_DUMMY", "ORDER_QUANTITY", "MISSING_QUANTITY"]
).copy()
df = df[df["ORDER_QUANTITY"] > 0].copy()
df["FAILURE_FLAG_LINE"] = (df["MISSING_QUANTITY"] > 0).astype(int)

vendor_summary = (
    df.groupby("VENDOR_DUMMY")
    .agg(
        order_quantity=("ORDER_QUANTITY", "sum"),
        missing_quantity=("MISSING_QUANTITY", "sum"),
        total_lines=("VENDOR_DUMMY", "size"),
        failure_lines=("FAILURE_FLAG_LINE", "sum"),
        materials=("MATERIAL_DUMMY", "nunique"),
        orders=("ORDERS_DUMMY", "nunique"),
    )
    .reset_index()
)

vendor_summary["missing_rate"] = (
    vendor_summary["missing_quantity"] / vendor_summary["order_quantity"]
)
vendor_summary["failure_line_rate"] = (
    vendor_summary["failure_lines"] / vendor_summary["total_lines"]
)

total_order = vendor_summary["order_quantity"].sum()
total_missing = vendor_summary["missing_quantity"].sum()

vendor_summary["volume_share"] = vendor_summary["order_quantity"] / total_order
vendor_summary["failure_share"] = np.where(
    total_missing > 0,
    vendor_summary["missing_quantity"] / total_missing,
    0,
)
vendor_summary["failure_exposure_ratio"] = np.where(
    vendor_summary["volume_share"] > 0,
    vendor_summary["failure_share"] / vendor_summary["volume_share"],
    np.nan,
)
vendor_summary["share_difference"] = (
    vendor_summary["failure_share"] - vendor_summary["volume_share"]
)

vendor_pareto = vendor_summary.sort_values(
    "missing_quantity", ascending=False
).copy()
vendor_pareto["cum_failure_share"] = vendor_pareto[
    "failure_share"
].cumsum()

reached_80 = vendor_pareto["cum_failure_share"] >= 0.80
if reached_80.any():
    cutoff = np.argmax(reached_80.to_numpy())
    vendors_80 = vendor_pareto.iloc[: cutoff + 1].copy()
else:
    vendors_80 = vendor_pareto.copy()

print("Total vendors:", vendor_summary["VENDOR_DUMMY"].nunique())
print("Vendors required for ~80% of failures:", len(vendors_80))
print(
    "Vendor share:",
    len(vendors_80) / vendor_summary["VENDOR_DUMMY"].nunique(),
)
print("Accumulated failure share:", vendors_80["failure_share"].sum())

thresholds = [0.005, 0.010, 0.020]
sensitivity = []
rankings = {}

for threshold in thresholds:
    subset = vendor_summary[
        vendor_summary["volume_share"] >= threshold
    ].copy()
    subset = subset.sort_values(
        "failure_exposure_ratio", ascending=False
    )
    rankings[threshold] = subset

    sensitivity.append(
        {
            "minimum_volume_share": threshold,
            "number_vendors": len(subset),
            "vendor_percentage": len(subset) / len(vendor_summary),
            "volume_coverage": subset["volume_share"].sum(),
            "failure_coverage": subset["failure_share"].sum(),
        }
    )

sensitivity_df = pd.DataFrame(sensitivity)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
vendor_summary.sort_values(
    "missing_quantity", ascending=False
).to_csv(OUTPUT_DIR / "vendor_analysis.csv", index=False)
vendors_80.to_csv(
    OUTPUT_DIR / "vendor_pareto_80.csv",
    index=False,
)
sensitivity_df.to_csv(
    OUTPUT_DIR / "vendor_exposure_sensitivity.csv",
    index=False,
)
rankings[0.005].to_csv(
    OUTPUT_DIR / "vendor_min_exposure_005.csv",
    index=False,
)
