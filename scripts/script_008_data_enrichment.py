"""Enrich the clean dataset with anonymized material attributes."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "clean_data.csv"
REFERENCE_FILE = ROOT / "data" / "reference" / "material_attributes.xlsx"
OUTPUT_FILE = ROOT / "data" / "processed" / "enriched_data.csv"

df = pd.read_csv(INPUT_FILE, low_memory=False)
df.columns = df.columns.str.strip().str.upper()

attributes = pd.read_excel(
    REFERENCE_FILE,
    sheet_name="MATERIAL_ATTRIBUTES",
)
attributes.columns = attributes.columns.str.strip().str.upper()

attributes = attributes[
    [
        "MATERIAL_DUMMY",
        "GROUP_DUMMY",
        "PROMO_ONLY",
        "CATEGORY_DUMMY",
        "VENDOR_DUMMY",
    ]
].drop_duplicates()

df_enriched = df.merge(
    attributes,
    on="MATERIAL_DUMMY",
    how="left",
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df_enriched.to_csv(OUTPUT_FILE, index=False)
print(f"Enriched dataset saved to: {OUTPUT_FILE}")
