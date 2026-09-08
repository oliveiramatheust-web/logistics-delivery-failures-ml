"""Create order-level and daily summaries and descriptive statistics."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "clean_data.csv"
ORDER_OUTPUT = ROOT / "outputs" / "order_summary.csv"
DAILY_OUTPUT = ROOT / "outputs" / "daily_summary.csv"

df = pd.read_csv(INPUT_FILE, low_memory=False)

for col in ["order_date", "mat_av_date"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")

numeric_cols = [
    "total_order_items",
    "missing_order_items",
    "order_quantity",
    "missing_quantity",
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["order_date", "mat_av_date"] + numeric_cols).copy()

df_order = (
    df.groupby("orders_dummy")
    .agg(
        order_date=("order_date", "first"),
        customer_dummy=("customer_dummy", "first"),
        total_order_items=("total_order_items", "max"),
        order_quantity=("order_quantity", "sum"),
        missing_quantity=("missing_quantity", "sum"),
    )
    .reset_index()
)
df_order["missing_rate"] = (
    df_order["missing_quantity"] / df_order["order_quantity"]
)

df_daily = (
    df.groupby("order_date")
    .agg(
        orders=("orders_dummy", "nunique"),
        order_quantity=("order_quantity", "sum"),
        missing_quantity=("missing_quantity", "sum"),
        lines=("total_order_items", "sum"),
    )
    .reset_index()
)
df_daily["missing_rate"] = (
    df_daily["missing_quantity"] / df_daily["order_quantity"]
)
df_daily["avg_units_per_order"] = (
    df_daily["order_quantity"] / df_daily["orders"]
)
df_daily["avg_lines_per_order"] = df_daily["lines"] / df_daily["orders"]

ORDER_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df_order.to_csv(ORDER_OUTPUT, index=False)
df_daily.to_csv(DAILY_OUTPUT, index=False)

print(df_daily["missing_rate"].describe())
print("Median:", df_daily["missing_rate"].median())
print("Days:", len(df_daily))
