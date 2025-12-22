# House Rent Prediction: A Comparative Analysis of Regression Models

## Abstract

This project implements a complete machine learning pipeline for predicting residential rental prices in India. The analysis compares three regression approaches: basic Linear Regression, Linear Regression with feature interactions, and XGBoost. The study demonstrates the application of data preprocessing, feature engineering, model evaluation, and statistical diagnostics in a real-world regression problem.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Dataset Description](#2-dataset-description)
3. [Methodology](#3-methodology)
4. [Exploratory Data Analysis](#4-exploratory-data-analysis)
5. [Data Preprocessing](#5-data-preprocessing)
6. [Feature Engineering](#6-feature-engineering)
7. [Model Development](#7-model-development)
8. [Results and Comparison](#8-results-and-comparison)
9. [Conclusions](#9-conclusions)
10. [How to Reproduce](#10-how-to-reproduce)

---

## 1. Introduction

### 1.1 Problem Statement

Rental price prediction is a regression task where the objective is to estimate the monthly rent of a property based on its characteristics. Accurate prediction models can assist both landlords in setting competitive prices and tenants in evaluating rental offers.

### 1.2 Objectives

1. Perform comprehensive exploratory data analysis (EDA)
2. Develop and compare multiple regression models
3. Evaluate model performance using standard regression metrics
4. Analyze feature importance and model interpretability
5. Check for multicollinearity using Variance Inflation Factor (VIF)

### 1.3 Notebook Structure

The analysis is contained in `ds.ipynb` and is organized into nine sections:

| Section | Title | Description |
|---------|-------|-------------|
| 1 | Data Loading & Exploration | Import data, examine structure, perform EDA |
| 2 | Data Preprocessing | Handle outliers, parse complex columns |
| 3 | Feature Engineering | Create derived features, apply transformations |
| 4 | Model 1: Linear Regression | Train baseline model, evaluate performance |
| 5 | Feature Importance Analysis | Coefficient interpretation, VIF diagnostics |
| 6 | Model 2: LR with Combinations | Feature interaction approach |
| 7 | Model Comparison (1 vs 2) | Compare linear models |
| 8 | Model 3: XGBoost | Gradient boosting approach |
| 9 | Final Comparison | Comprehensive evaluation of all models |

---

## 2. Dataset Description

### 2.1 Data Source

The dataset contains 4,746 rental property listings from six major Indian cities.

**File location**: `Data/rent.csv`

### 2.2 Feature Dictionary

| Variable | Description | Type | Range/Categories |
|----------|-------------|------|------------------|
| `Posted On` | Listing publication date | Date | 2022 dates |
| `BHK` | Number of bedrooms | Numeric | 1-6 |
| `Rent` | Monthly rent in INR (target) | Numeric | ₹1,200 - ₹3,500,000 |
| `Size` | Area in square feet | Numeric | 10 - 8,000 |
| `Floor` | Floor information | Text | Format: "X out of Y" |
| `City` | City location | Categorical | 6 cities |
| `Furnishing Status` | Furnishing level | Categorical | Furnished, Semi-Furnished, Unfurnished |
| `Tenant Preferred` | Preferred tenant type | Categorical | Family, Bachelors, Bachelors/Family |
| `Bathroom` | Number of bathrooms | Numeric | 1-10 |

### 2.3 Target Variable

The target variable `Rent` exhibits strong right-skewness (mean: ₹34,993, median: ₹16,000). A logarithmic transformation is applied to normalize the distribution and improve model performance.

---

## 3. Methodology

### 3.1 Approach

This study employs a log-linear regression framework where:

```
log(Rent) = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε
```

This formulation allows coefficient interpretation as percentage changes in rent.

### 3.2 Models Evaluated

| Model | Algorithm | Feature Set | Complexity |
|-------|-----------|-------------|------------|
| Model 1 | Linear Regression | Basic features | 21 features |
| Model 2 | Linear Regression | With category combinations | 48 features |
| Model 3 | XGBoost Regressor | Basic features | 21 features |

### 3.3 Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| R² | 1 - SS_res/SS_tot | Variance explained (higher is better) |
| RMSE | √(Σ(yᵢ - ŷᵢ)²/n) | Root mean squared error in INR |
| MAE | Σ\|yᵢ - ŷᵢ\|/n | Mean absolute error in INR |
| MAPE | Σ\|yᵢ - ŷᵢ\|/yᵢ × 100/n | Mean absolute percentage error |

Note: R² is computed on log-scale; RMSE, MAE, and MAPE are computed on the original scale after exponential back-transformation.

---

## 4. Exploratory Data Analysis

### 4.1 Distribution Analysis

The target variable analysis reveals:
- Minimum rent: ₹1,200
- Maximum rent: ₹3,500,000
- Mean: ₹34,993
- Median: ₹16,000
- Skewness: Highly right-skewed

The logarithmic transformation produces an approximately normal distribution suitable for linear regression.

### 4.2 Categorical Variable Distribution

Analysis of categorical features shows:
- **City**: Balanced distribution across 6 cities (Bangalore, Chennai, Delhi, Hyderabad, Kolkata, Mumbai)
- **Furnishing Status**: Three levels with varying frequencies
- **Tenant Preferred**: Three categories reflecting landlord preferences

### 4.3 Key Observations

1. Mumbai exhibits significantly higher rental prices compared to other cities
2. Positive correlation exists between property size and rent
3. Furnished properties command premium prices
4. No missing values in the dataset

---

## 5. Data Preprocessing

### 5.1 Outlier Treatment

- Maximum rent value (₹3,500,000) identified as extreme outlier and removed
- Percentile-based filtering applied (20th-80th percentile for Rent)

### 5.2 Floor Column Parsing

The `Floor` column is parsed to extract numeric features:

| Original Value | Floor Level | Total Floors |
|----------------|-------------|--------------|
| "2 out of 5" | 2 | 5 |
| "Ground out of 3" | 0 | 3 |
| "Upper Basement out of 2" | -1 | 2 |

### 5.3 Temporal Feature Extraction

From the `Posted On` date column:
- `day of week posted`: Integer (0=Monday, 6=Sunday)
- `quarter posted`: Integer (1-4)

---

## 6. Feature Engineering

### 6.1 Data Splitting

| Set | Size | Percentage |
|-----|------|------------|
| Training | 3,796 | 80% |
| Testing | 949 | 20% |

Random state: 42 (for reproducibility)

### 6.2 Feature Transformations

| Feature Type | Transformation | Rationale |
|--------------|----------------|-----------|
| Size | Log transformation | Linearize relationship with rent |
| Numeric (BHK, Bathroom, Floor Level, Total Floors) | Median imputation | Handle potential missing values |
| Categorical | One-Hot Encoding | Convert to binary indicators |

### 6.3 Dummy Variable Trap Avoidance

For each categorical variable, the first category is dropped to prevent perfect multicollinearity:
- Reference category for City: Bangalore
- Reference category for Furnishing: Furnished
- Reference category for Tenant: Bachelors

### 6.4 Feature Combinations (Model 2)

Model 2 implements joint encoding for City × Quarter × Tenant:
- Combined categories: 6 × 2 × 3 = 36 unique combinations
- Final feature count after dummy drop: 48

---

## 7. Model Development

### 7.1 Model 1: Basic Linear Regression

**Configuration:**
```python
LinearRegression(fit_intercept=True)
```

**Features:** 21 (after preprocessing)

### 7.2 Model 2: Linear Regression with Feature Combinations

**Configuration:**
```python
LinearRegression(fit_intercept=True)
```

**Features:** 48 (including category combinations)

### 7.3 Model 3: XGBoost Regressor

**Configuration:**
```python
XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

**Features:** 21 (same as Model 1)

---

## 8. Results and Comparison

### 8.1 Performance Metrics

#### Training Set Performance

| Metric | Model 1 | Model 2 | Model 3 |
|--------|---------|---------|---------|
| R² | 0.7847 | 0.7881 | 0.9122 |
| RMSE (₹) | 36,081 | 36,077 | 18,454 |
| MAE (₹) | 12,565 | 12,482 | 6,812 |
| MAPE (%) | 34.77 | 34.47 | 21.62 |

#### Test Set Performance

| Metric | Model 1 | Model 2 | Model 3 |
|--------|---------|---------|---------|
| R² | 0.7826 | 0.7781 | **0.8112** |
| RMSE (₹) | 43,970 | 44,184 | **37,170** |
| MAE (₹) | 13,259 | 13,440 | **11,750** |
| MAPE (%) | 35.37 | 35.91 | **32.47** |

### 8.2 Generalization Analysis

| Model | R² (Train) | R² (Test) | Gap | Assessment |
|-------|------------|-----------|-----|------------|
| Model 1 | 0.7847 | 0.7826 | 0.0021 | Excellent generalization |
| Model 2 | 0.7881 | 0.7781 | 0.0100 | Good generalization |
| Model 3 | 0.9122 | 0.8112 | 0.1010 | Mild overfitting |

### 8.3 Feature Importance

#### Linear Regression Coefficients (Top 5)

| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| City: Mumbai | +1.027 | 180% higher rent vs. baseline |
| Furnishing: Unfurnished | -0.326 | 28% lower rent vs. furnished |
| City: Delhi | +0.312 | 37% higher rent vs. baseline |
| Bathroom | +0.269 | +27% rent per additional bathroom |
| log(Size) | +0.251 | +0.25% rent per 1% size increase |

#### XGBoost Feature Importance (Top 5)

| Feature | Importance | Percentage |
|---------|------------|------------|
| City: Mumbai | 0.562 | 56.2% |
| Bathroom | 0.200 | 20.0% |
| log(Size) | 0.035 | 3.5% |
| BHK | 0.032 | 3.2% |
| Total Floors | 0.031 | 3.1% |

### 8.4 Multicollinearity Diagnostics

VIF analysis for Model 1 indicates no multicollinearity issues:
- All features: VIF < 5
- Maximum VIF: 4.23 (Total Floors)
- Minimum VIF: 1.12 (Quarter: Q3)

---

## 9. Conclusions

### 9.1 Key Findings

1. **Best Performing Model**: XGBoost achieves superior performance across all metrics on the test set
2. **Feature Combinations**: Adding category interactions (Model 2) did not improve performance, suggesting the original features capture sufficient information
3. **Location Dominance**: City, particularly Mumbai, is the strongest predictor of rental prices
4. **Model Interpretability**: Linear Regression provides interpretable coefficients as percentage changes

### 9.2 Model Selection Recommendation

| Criterion | Recommended Model |
|-----------|-------------------|
| Predictive accuracy | Model 3 (XGBoost) |
| Interpretability | Model 1 (Linear Regression) |
| Balance | Model 1 (minimal overfitting, good performance) |

### 9.3 Limitations

- Dataset limited to six Indian cities
- Temporal features may not generalize across different time periods
- XGBoost shows signs of mild overfitting (10% R² gap)

### 9.4 Future Work

- Implement cross-validation for more robust evaluation
- Apply hyperparameter tuning for XGBoost
- Explore additional ensemble methods (LightGBM, CatBoost)
- Include geospatial features (latitude, longitude)

---

## 10. How to Reproduce

### 10.1 Requirements

```
Python >= 3.8
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
xgboost >= 1.5.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
statsmodels >= 0.13.0
```

### 10.2 Installation

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn statsmodels
```

### 10.3 Execution

```bash
jupyter notebook ds.ipynb
```

Navigate to `Kernel → Restart & Run All` to execute all cells.

---

## Project Structure

```
House-Rent-Prediction/
├── Data/
│   └── rent.csv                    # Raw dataset
├── ds.ipynb                        # Main analysis notebook
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # Data loading module
│   │   ├── data_transformation.py  # Preprocessing pipeline
│   │   └── combined_onehot_encoder.py  # Custom encoder
│   └── config.py                   # Configuration parameters
├── artifacts/                      # Generated outputs
├── model_trainer_final.py          # Training script
├── show_coefs.py                   # Coefficient analysis
└── README.md                       # This document
```

---

## References

1. Scikit-learn Documentation: https://scikit-learn.org/
2. XGBoost Documentation: https://xgboost.readthedocs.io/
3. Statsmodels Documentation: https://www.statsmodels.org/

---

*Document prepared for academic review.*
