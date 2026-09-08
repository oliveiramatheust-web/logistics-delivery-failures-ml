"""Analyze failures and material-availability timing by anonymized customer."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "clean_data.csv"
OUTPUT_FILE = ROOT / "outputs" / "customer_analysis.csv"

df = pd.read_csv(INPUT_FILE, low_memory=False)

df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
df["mat_av_date"] = pd.to_datetime(df["mat_av_date"], errors="coerce")

df["days_to_availability"] = (
    df["mat_av_date"] - df["order_date"]
).dt.days

df = df[df["days_to_availability"] >= 0].copy()

customer_summary = (
    df.groupby("customer_dummy")
    .agg(
        order_quantity=("order_quantity", "sum"),
        missing_quantity=("missing_quantity", "sum"),
        orders=("orders_dummy", "nunique"),
        distinct_materials=("material_dummy", "nunique"),
        avg_days_to_availability=("days_to_availability", "mean"),
    )
    .reset_index()
)

customer_summary["missing_rate"] = (
    customer_summary["missing_quantity"]
    / customer_summary["order_quantity"]
)
customer_summary["failure_flag_customer"] = (
    customer_summary["missing_quantity"] > 0
).astype(int)
customer_summary["avg_units_per_order"] = (
    customer_summary["order_quantity"] / customer_summary["orders"]
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
customer_summary.to_csv(OUTPUT_FILE, index=False)
print(f"Customer summary saved to: {OUTPUT_FILE}")
