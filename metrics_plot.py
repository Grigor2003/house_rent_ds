import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load model and data
with open('artifacts/model.pkl', 'rb') as f:
    model = pickle.load(f)

X_train = pd.read_csv('artifacts/X_train_arr.csv')
X_test = pd.read_csv('artifacts/X_test_arr.csv')
y_train_log = np.log(pd.read_csv('artifacts/train.csv')['Rent'].values)
y_test_log = np.log(pd.read_csv('artifacts/test.csv')['Rent'].values)

# Actual rent (not log)
y_train_actual = pd.read_csv('artifacts/train.csv')['Rent'].values
y_test_actual = pd.read_csv('artifacts/test.csv')['Rent'].values

# Predictions (in log scale)
y_train_pred_log = model.predict(X_train)
y_test_pred_log = model.predict(X_test)

# Convert predictions back to original scale
y_train_pred = np.exp(y_train_pred_log)
y_test_pred = np.exp(y_test_pred_log)

# ============================================
# Calculate Metrics
# ============================================

# R² (on log scale - what model optimizes)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

r2_train = r2_score(y_train_log, y_train_pred_log)
r2_test = r2_score(y_test_log, y_test_pred_log)

# RMSE (on original scale)
rmse_train = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))

# MAE (on original scale)
mae_train = mean_absolute_error(y_train_actual, y_train_pred)
mae_test = mean_absolute_error(y_test_actual, y_test_pred)

# RMSPE (Root Mean Square Percentage Error)
def rmspe(y_true, y_pred):
    """Calculate RMSPE - penalizes percentage errors"""
    return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2)) * 100

rmspe_train = rmspe(y_train_actual, y_train_pred)
rmspe_test = rmspe(y_test_actual, y_test_pred)

# MAPE (Mean Absolute Percentage Error)
def mape(y_true, y_pred):
    """Calculate MAPE"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape_train = mape(y_train_actual, y_train_pred)
mape_test = mape(y_test_actual, y_test_pred)

# ============================================
# Print Results
# ============================================
print("=" * 60)
print("MODEL PERFORMANCE METRICS")
print("=" * 60)
print()
print(f"{'Metric':<20} {'Train':<15} {'Test':<15}")
print("-" * 50)
print(f"{'R²':<20} {r2_train:<15.4f} {r2_test:<15.4f}")
print(f"{'RMSE':<20} {rmse_train:<15.2f} {rmse_test:<15.2f}")
print(f"{'MAE':<20} {mae_train:<15.2f} {mae_test:<15.2f}")
print(f"{'RMSPE (%)':<20} {rmspe_train:<15.2f} {rmspe_test:<15.2f}")
print(f"{'MAPE (%)':<20} {mape_train:<15.2f} {mape_test:<15.2f}")
print()

# ============================================
# Create Table for Presentation
# ============================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

table_data = [
    ['R²', f'{r2_train:.4f}', f'{r2_test:.4f}'],
    ['RMSE', f'{rmse_train:,.0f}', f'{rmse_test:,.0f}'],
    ['MAE', f'{mae_train:,.0f}', f'{mae_test:,.0f}'],
    ['RMSPE (%)', f'{rmspe_train:.2f}%', f'{rmspe_test:.2f}%'],
    ['MAPE (%)', f'{mape_train:.2f}%', f'{mape_test:.2f}%'],
]

columns = ['Metric', 'Train', 'Test']

table = ax.table(
    cellText=table_data,
    colLabels=columns,
    cellLoc='center',
    loc='center',
    colColours=['#4472C4'] * len(columns)
)

table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1.5, 2.5)

# Style header
for i in range(len(columns)):
    table[(0, i)].set_text_props(weight='bold', color='white')
    table[(0, i)].set_fontsize(14)

# Color cells
for i in range(len(table_data)):
    row_idx = i + 1
    for j in range(len(columns)):
        table[(row_idx, j)].set_facecolor('#E6F3FF')

ax.set_title('Model Performance Metrics\nLog-Linear Regression', 
             fontsize=18, fontweight='bold', pad=20)


