# Logistics Delivery Failures — Statistical and Machine Learning Analysis

This repository contains sanitized Python scripts supporting an academic
study on logistics delivery failures. The code documents the analytical
methodology while excluding proprietary datasets, internal paths, company
systems, credentials, and identifiable operational information.

## Repository scope

The scripts cover:

1. CSV consolidation
2. Data wrangling
3. Analytical summaries
4. Linear regression
5. Material-level failure and Pareto analysis
6. Customer and processing-time analysis
7. Logistic Regression I
8. Data enrichment with anonymized material attributes
9. Random Forest with temporal validation and leakage control
10. Logistic Regression II for future critical-material classification
11. Supplier-level failure and exposure analysis

The numbering intentionally corresponds to the source-code appendices in
the associated academic study.

## Methodology equivalence

See [`METHODOLOGY_EQUIVALENCE.md`](METHODOLOGY_EQUIVALENCE.md) for the appendix-to-script mapping and the sanitization rules applied to the publication versions.

## Data availability

The original dataset is **not included** because it is subject to
confidentiality restrictions.

No real customer, material, supplier, order, company, server, directory,
or system identifiers are included in this repository.

The analytical code expects anonymized identifiers such as:

- `orders_dummy`
- `material_dummy`
- `customer_dummy`
- `group_dummy`
- `category_dummy`
- `vendor_dummy`

## Expected directory structure

```text
logistics-delivery-failures-ml/
├── data/
│   ├── raw/
│   ├── processed/
│   └── reference/
├── outputs/
└── scripts/
```

`data/raw/` is used by Script 001 for source CSV files.

`data/processed/` contains locally generated intermediate datasets and is
excluded from version control.

`data/reference/material_attributes.xlsx` is an optional local reference
file used by Script 008. It must contain a worksheet named
`MATERIAL_ATTRIBUTES` with these anonymized columns:

- `MATERIAL_DUMMY`
- `GROUP_DUMMY`
- `PROMO_ONLY`
- `CATEGORY_DUMMY`
- `VENDOR_DUMMY`

The original reference file is not distributed.

## Methodological note

These files are intended to document the methodology used in the academic
study. Full empirical reproducibility requires access to the confidential
source data and therefore is not provided by this public repository.

## Technologies

- Python
- pandas
- NumPy
- statsmodels
- scikit-learn
- openpyxl
