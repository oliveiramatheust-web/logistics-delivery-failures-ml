"""Consolidate quarterly CSV files into a single analytical dataset."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_FILE = ROOT / "data" / "processed" / "consolidated_data.csv"

csv_files = sorted(RAW_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in {RAW_DIR}. "
        "The original study data are not distributed with this repository."
    )

dataframes = [pd.read_csv(file, low_memory=False) for file in csv_files]
df = pd.concat(dataframes, ignore_index=True)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print(f"Files consolidated: {len(csv_files)}")
print(f"Final shape: {df.shape}")
print(f"Saved to: {OUTPUT_FILE}")
