"""Clean and standardize the consolidated logistics dataset."""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "consolidated_data.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "clean_data.csv"

df = pd.read_csv(INPUT_FILE, low_memory=False)

# Column order used in the original analytical dataset.
df.columns = [
    "mat_av_date",
    "order_date",
    "orders_dummy",
    "material_dummy",
    "customer_dummy",
    "total_order_items",
    "missing_order_items",
    "order_quantity",
    "missing_quantity",
]

for col in ["order_date", "mat_av_date"]:
    df[col] = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce")

numeric_cols = [
    "total_order_items",
    "missing_order_items",
    "order_quantity",
    "missing_quantity",
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["order_date", "mat_av_date"] + numeric_cols).copy()

df["failure_flag_line"] = (df["missing_quantity"] > 0).astype(int)
df["missing_rate_line"] = np.where(
    df["order_quantity"] > 0,
    df["missing_quantity"] / df["order_quantity"],
    0,
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)
print(f"Clean dataset saved to: {OUTPUT_FILE}")
