"""Estimate the daily failure-rate OLS model and calculate VIF."""

from pathlib import Path
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "outputs" / "daily_summary.csv"

df = pd.read_csv(INPUT_FILE)

features = [
    "orders",
    "order_quantity",
    "avg_units_per_order",
    "avg_lines_per_order",
]

X = sm.add_constant(df[features])
y = df["missing_rate"]

model = sm.OLS(y, X).fit()
print(model.summary())

vif = pd.DataFrame(
    {
        "Variable": X.columns,
        "VIF": [
            variance_inflation_factor(X.values, i)
            for i in range(X.shape[1])
        ],
    }
)
print("\nVariance Inflation Factors")
print(vif)
