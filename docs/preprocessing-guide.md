# BreadCast Data Preprocessing Guide

## 1. Overview

The preprocessing module prepares bakery sales data for model training, monitoring, retraining and holdout prediction. It ensures that data is cleaned and transformed consistently using configuration parameters defined in `params.yaml`.

## 2. Data Cleaning

### Duplicate Removal

Exact duplicate records are removed from the training dataset using the pandas `drop_duplicates()` function.

### Price Unit Correction

Incorrect prices for the College Gate outlet are divided by 100. For example, an incorrect price of 5000 is converted to 50.

For later and holdout datasets, this correction is applied only when the median price of the affected outlet exceeds 1000.

### Target Value Correction

Training target values exceeding 1000 are treated as suspected typing errors and divided by 10. For example, 4000 loaves sold is corrected to 400.

### Original Data Preservation

Cleaning functions create a copy of the input DataFrame before applying transformations, protecting the original data from unintended modifications.

## 3. Feature Selection

The project uses eight input features.

| Feature | Type |
|---|---|
| price_rs | Numerical |
| temp_c | Numerical |
| footfall_index | Numerical |
| competitor_shops | Numerical |
| shelf_hours | Numerical |
| orders_previous_day | Numerical |
| day_type | Categorical |
| outlet | Categorical |

The target variable is `loaves_sold`. Record IDs and dates are excluded from the model input features.

## 4. Preprocessing Pipeline

The pipeline uses Scikit-learn's ColumnTransformer to process numerical and categorical features separately.

- **StandardScaler:** Standardizes numerical features using their training mean and standard deviation.
- **OneHotEncoder:** Converts categorical values into numerical representations.
- **Unknown Category Handling:** The encoder uses `handle_unknown="ignore"` to support previously unseen categories during transformation.

The pipeline does not include an explicit missing-value imputation step.

## 5. Data Preparation Workflow

### Training Data

Raw training data → Duplicate removal → Price correction → Target correction → Feature-target separation → Feature transformation

### Later Data

Raw later data → Conditional price correction → Feature-target separation → Monitoring and retraining

### Holdout Data

Raw holdout data → Conditional price correction → Feature and record ID extraction → Final prediction

## 6. Additional Test Coverage

Three additional automated tests were added to strengthen preprocessing validation.

| Test | Purpose | Result |
|---|---|---|
| Original Data Preservation | Verifies that training data cleaning does not modify the original DataFrame. | PASSED |
| Correct Price Preservation | Ensures valid prices in later and holdout data are not incorrectly converted. | PASSED |
| Unseen Category Handling | Verifies that an unknown outlet can be transformed while maintaining output dimensions and finite values. | PASSED |

## 7. Extended Validation Results

Execution date: 27 November 2026

Command: `python -m pytest tests/test_preprocessing.py -v`

| Metric | Result |
|---|---|
| Original Tests | 8 |
| Additional Tests | 3 |
| Total Tests | 11 |
| Passed | 11 |
| Failed | 0 |
| Success Rate | 100% |
| Execution Time | 2.01 seconds |

## 8. Conclusion

All eleven preprocessing tests passed successfully. The additional tests improve coverage of input data preservation, valid price handling and unseen categorical values without changing the existing preprocessing implementation.

## 9. Full Project Test Suite Validation

The complete project test suite was executed using `python -m pytest tests/ -v` to verify that the additional preprocessing tests did not introduce regressions.

- Total tests: 35
- Passed: 34
- Failed: 0
- Skipped: 1
- Execution time: 4.09 seconds

One data quality test was skipped because the DVC-tracked raw training dataset was unavailable locally. A dependency deprecation warning was also reported. All eleven preprocessing tests passed successfully.