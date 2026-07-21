# Predicting 30-Day Hospital Readmission in Diabetic Patients
### A Distributed Machine Learning Approach - Apache Spark MLlib

**Student Names:** Alina Baig and Kulsoom Zaidi
**Course:** Big Data Analytics and Text Mining - Module 2
**University:** Università di Bologna - MSc Artificial Intelligence

---

## Overview

This project builds a distributed machine learning pipeline in **PySpark / Spark MLlib** to predict whether a diabetic patient will be **readmitted to hospital within 30 days** of discharge, using the UCI *Diabetes 130-US Hospitals (1999-2008)* dataset. The pipeline covers data cleaning, feature engineering, exploratory analysis, leakage-safe train/test splitting, class-imbalance handling, cross-validated model tuning, threshold tuning for clinical use, and fairness/robustness checks across age subgroups.

## Dataset

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
- **Records:** 101,766 patient visits | **Features:** 50 clinical variables
- **Reference:** Strack et al. (2014), *BioMed Research International*


## Environment / Library Versions

| Library | Version |
|---|---|
| PySpark | 3.5 |
| Python | 3.10 |
| pandas | 2.0 |
| matplotlib | 3.7 |
| seaborn | 0.12 |
| scikit-learn | 1.3 |
| xgboost | 2.0 |


## 6. Models

| # | Algorithm | Imbalance Strategy | Notes |
|---|---|---|---|
| 1 | Decision Tree | Class Weights | Most interpretable - clinical rules |
| 2 | Random Forest | Class Weights | Stable feature importance ranking |
| 3 | XGBoost | Class Weights | Best overall discriminator |


## Results

Final test-set metrics (accuracy, F1, recall, precision, AUC-ROC) are compared across all three models in the notebook's final results section. Expected AUC is in line with published benchmarks for this dataset (Strack et al., 2014: AUC ≈ 0.63-0.68). XGBoost is the strongest overall discriminator; threshold tuning further improves recall on the minority (readmitted) class, which matters most in this medical context.

## How to Run

1. Ensure a Spark cluster/master is reachable; adjust as needed for your environment.
2. Install the dependencies listed above.
3. Run the notebook top to bottom - it automatically downloads and extracts the UCI dataset (`diabetic_data.csv`) on first run if not already present.

