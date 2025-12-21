# House Rent Prediction

## Project Overview

This project predicts house rental prices in India using machine learning. The model analyzes property characteristics (size, location, furnishing, etc.) to estimate monthly rent.

**Key approach**: We use a **log-linear regression model** where the dependent variable (Rent) is log-transformed. This allows interpreting coefficients as percentage changes in rent.

---

## Project Structure

```
House-Rent-Prediction/
├── Data/
│   └── rent.csv                 # Original dataset
├── src/
│   ├── components/
│   │   ├── data_ingestion.py    # Loads and splits data
│   │   ├── data_transformation.py  # Feature engineering & preprocessing
│   │   ├── model_trainer.py     # Trains the regression model
│   │   └── combined_onehot_encoder.py  # Custom encoder for category combinations
│   └── config.py                # All configurable parameters
├── artifacts/                   # Generated files (models, processed data)
├── model_trainer_final.py       # Main script to run the full pipeline
├── show_coefs.py                # Analyze model coefficients
└── correlation_matrix.py        # Visualize feature correlations
```

---

## How It Works

### Step 1: Data Ingestion
- Loads the dataset from `Data/rent.csv`
- Splits data into training (80%) and test (20%) sets
- Saves splits to `artifacts/train.csv` and `artifacts/test.csv`

### Step 2: Data Transformation

This is where feature engineering happens:

1. **Floor Level Extraction**  
   Parses "2 out of 5" → extracts floor number (2)

2. **Date Features**  
   From "Posted On" date extracts:
   - Day of week (Mon=0, Sun=6)
   - Quarter (Q1, Q2, Q3, Q4)

3. **Log-Scaling**  
   Applies natural logarithm to Size: `log(Size)`  
   This linearizes the relationship with rent.

4. **Categorical Encoding**  
   Converts categorical variables to dummy variables (one-hot encoding):
   - City
   - Furnishing Status
   - Tenant Preferred

5. **Category Combinations** (Optional)  
   Creates joint categories like "City × Quarter × Day" for capturing location-time effects.  
   Example: "Mumbai & Q1 & Mon" vs "Delhi & Q4 & Fri"

6. **Feature Interactions** (Optional)  
   Creates multiplicative interactions between features:
   - `num × num`: e.g., log(Size) × Floor Level
   - `num × cat`: e.g., log(Size) × each Furnishing dummy

### Step 3: Model Training
- Trains a Linear Regression model on processed features
- Saves the trained model to `artifacts/model.pkl`

---

## Configuration

All parameters are in `src/config.py`:

```python
# Which columns to log-scale
LOG_SCALING_COLS = ['Size']

# Categorical columns for one-hot encoding
CATEGORICAL_COLS = ['City', 'Furnishing Status', 'Tenant Preferred', ...]

# Category combinations (e.g., City × Day × Quarter)
COMBINATIONS_TO_APPLY = [
    {'columns': ['City', 'quarter posted', 'day of week posted'], 'keep_originals': False},
]

# Feature interactions (optional)
FEATURE_INTERACTIONS = [
    # {'x': 'log(Size)', 'y': 'Floor Level', 'type': 'num-num'},
]
```

---

## How to Run

### Train the Model
```bash
python model_trainer_final.py
```

This runs the full pipeline: data loading → transformation → model training.

### Analyze Coefficients
```bash
python show_coefs.py
```

Shows regression coefficients with significance levels and saves to `artifacts/coefficients.csv`.

### View Correlation Matrix
```bash
python correlation_matrix.py
```

Generates a heatmap of feature correlations.

---

## Output Files

After running the pipeline, `artifacts/` contains:

| File | Description |
|------|-------------|
| `model.pkl` | Trained regression model |
| `preprocessor.pkl` | Data preprocessing pipeline |
| `X_train_arr.csv` | Processed training features |
| `X_test_arr.csv` | Processed test features |
| `coefficients.csv` | Model coefficients with statistics |
| `correlation_matrix.csv` | Feature correlation matrix |

---

## Interpreting Results

Since we use **log(Rent)** as the target:

- A coefficient of **0.05** means: +1 unit in the feature → **+5% rent**
- A coefficient of **-0.10** means: +1 unit in the feature → **-10% rent**

For dummy variables (0/1):
- The coefficient shows the % difference compared to the baseline category

---

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- statsmodels (for coefficient analysis)
- matplotlib (for visualizations)

Install: `pip install pandas numpy scikit-learn statsmodels matplotlib`
