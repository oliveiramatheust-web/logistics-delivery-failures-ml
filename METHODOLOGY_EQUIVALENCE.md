# Methodology Equivalence and Sanitization Notes

This repository contains condensed, sanitized publication versions of the
Python source code documented in the academic-study appendices.

The public scripts are **methodologically equivalent**, not line-for-line
copies. Repetitive console output, local file-system details, internal file
names, and explanatory comments that do not affect the analytical procedure
were shortened or removed.

## Mapping to the academic appendices

| Appendix source | Public script | Preserved analytical procedure |
|---|---|---|
| 001 | `script_001_data_consolidation.py` | Concatenation of source CSV files |
| 002 | `script_002_data_wrangling.py` | Type conversion, invalid-record removal, failure flag and line failure rate |
| 003_V2 | `script_003_analytical_summary.py` | Order-level aggregation, daily aggregation and descriptive statistics |
| 004 | `script_004_linear_regression.py` | OLS specification and VIF assessment |
| 005_V2 | `script_005_material_failure_analysis.py` | Material aggregation, failure/exposure metrics, 80% Pareto and exposure sensitivity |
| 006 | `script_006_customer_lead_time_analysis.py` | Customer aggregation and days-to-availability analysis |
| 007 | `script_007_logistic_regression_1.py` | Baseline logistic regression, original features and 70/30 split |
| 008 | `script_008_data_enrichment.py` | Merge of anonymized material attributes |
| 009_V2 | `script_009_random_forest.py` | Random Forest, 2023–2024 training / 2025 test, class balancing and leakage control |
| 010_V4 | `script_010_logistic_regression_2.py` | Annual material aggregation, 80% critical flag, 2023→2024 training and 2024→2025 temporal test |
| 011_V2 | `script_011_material_supplier_analysis.py` | Supplier aggregation, failure/exposure metrics, 80% Pareto and exposure sensitivity |

## Elements intentionally preserved

The following methodological elements were retained because changing them
could change the interpretation or reported results of the study:

- anonymized analytical identifiers (`*_DUMMY`);
- failure definitions based on missing quantity;
- aggregation levels used by each analysis;
- the 80% Pareto cutoff rule, including the observation that crosses 80%;
- exposure thresholds used in the material and supplier sensitivity analyses;
- OLS model specification and VIF calculation;
- Logistic Regression I feature construction and 70/30 split;
- Random Forest parameters:
  - `n_estimators=100`
  - `max_depth=10`
  - `random_state=42`
  - `class_weight="balanced"`
- Random Forest temporal validation:
  - training: 2023–2024
  - testing: 2025
- exclusion of outcome-derived and post-event variables from the Random Forest;
- Logistic Regression II:
  - critical-material target based on approximately 80% of missing units;
  - 2023 features → 2024 criticality for training;
  - 2024 features → 2025 criticality for testing;
  - logarithmic historical volume feature;
  - category/group dummy variables;
  - promotional indicator;
  - `class_weight="balanced"`.

## Sanitization changes

The following changes were made solely for publication safety and portability:

- removed local Windows paths and project-folder names;
- removed original operational file names;
- replaced local paths with repository-relative `pathlib` paths;
- replaced the original enrichment workbook name and worksheet name with
  generic publication equivalents;
- excluded all source data, processed data and generated analytical outputs;
- excluded credentials, tokens, URLs, e-mail addresses and company/system
  references;
- retained only anonymized analytical field names;
- reduced repetitive print statements and comments where they did not alter
  calculations.

## Reproducibility limitation

The public repository documents the analytical procedure but cannot provide
full empirical reproducibility because the original data and reference files
are confidential and are not distributed.
