import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ============================================
# Load data (column names are included in CSV)
# ============================================
X = pd.read_csv('artifacts/X_train_arr.csv')
y_raw = pd.read_csv('artifacts/train.csv')['Rent'].values
y = np.log(y_raw)

print(f'Number of features: {X.shape[1]}')
print(f'Number of samples: {X.shape[0]}')
print(f'Features: {list(X.columns)}')
print()

# ============================================
# Fit OLS model with statsmodels
# ============================================
X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const).fit()

print('=' * 90)
print('LINEAR REGRESSION - STATSMODELS OLS SUMMARY')
print('=' * 90)
print()
print(f'R-squared: {model.rsquared:.4f}')
print(f'Adj. R-squared: {model.rsquared_adj:.4f}')
print(f'F-statistic: {model.fvalue:.2f} (p-value: {model.f_pvalue:.2e})')
print()

# Get confidence intervals
conf_int = model.conf_int(alpha=0.05)
conf_int.columns = ['CI_Low', 'CI_High']

# Create results DataFrame
results = pd.DataFrame({
    'Coefficient': model.params,
    'Std Error': model.bse,
    'CI_Low': conf_int['CI_Low'],
    'CI_High': conf_int['CI_High'],
    't-stat': model.tvalues,
    'p-value': model.pvalues
})

results['Significant'] = results['p-value'].apply(
    lambda x: '***' if x < 0.001 else ('**' if x < 0.01 else ('*' if x < 0.05 else ''))
)
results['Abs'] = np.abs(results['Coefficient'])

# Print intercept
intercept = results.loc['const']
print(f"Intercept: {intercept['Coefficient']:.4f}")
print(f"           95% CI: [{intercept['CI_Low']:.4f}, {intercept['CI_High']:.4f}]")
print(f"           Std Error: {intercept['Std Error']:.4f}, p-value: {intercept['p-value']:.4e}")
print()

# Remove intercept and sort by absolute value
results_coef = results.drop('const').sort_values('Abs', ascending=False)

print('Coefficients (sorted by importance):')
print('-' * 90)
print(f'{"Feature":<35} {"Coef":>10} {"Std Err":>10} {"95% CI":>24} {"p-value":>12}')
print('-' * 90)

for name, row in results_coef.iterrows():
    ci_str = f"[{row['CI_Low']:>9.4f}, {row['CI_High']:>9.4f}]"
    p_str = "<0.0001" if row['p-value'] < 0.0001 else f"{row['p-value']:.4f}"
    print(f"{name:<35} {row['Coefficient']:>10.4f} {row['Std Error']:>10.4f} {ci_str:>24} {p_str:>10} {row['Significant']}")

# Save coefficients to CSV
results.to_csv('artifacts/coefficients.csv')
print('\nSaved coefficients to artifacts/coefficients.csv')

print()
print('=' * 90)
print('INTERPRETATION')
print('=' * 90)
print()
print('Significance: *** p<0.001, ** p<0.01, * p<0.05')
print('Target variable: log(Rent)')
print()

# Key insights
sig_positive = results_coef[(results_coef['Coefficient'] > 0) & (results_coef['p-value'] < 0.05)].head(5)
sig_negative = results_coef[(results_coef['Coefficient'] < 0) & (results_coef['p-value'] < 0.05)].head(5)

print('Top factors INCREASING rent (significant):')
for name, row in sig_positive.iterrows():
    pct = (np.exp(row['Coefficient']) - 1) * 100
    ci_pct_low = (np.exp(row['CI_Low']) - 1) * 100
    ci_pct_high = (np.exp(row['CI_High']) - 1) * 100
    print(f"  * {name}: +{pct:.1f}% (95% CI: [{ci_pct_low:.1f}%, {ci_pct_high:.1f}%])")

print()
print('Top factors DECREASING rent (significant):')
for name, row in sig_negative.iterrows():
    pct = (np.exp(row['Coefficient']) - 1) * 100
    ci_pct_low = (np.exp(row['CI_Low']) - 1) * 100
    ci_pct_high = (np.exp(row['CI_High']) - 1) * 100
    print(f"  * {name}: {pct:.1f}% (95% CI: [{ci_pct_low:.1f}%, {ci_pct_high:.1f}%])")

# ============================================
# Variance Inflation Factor (VIF)
# ============================================
print()
print('=' * 90)
print('VARIANCE INFLATION FACTOR (VIF) - Multicollinearity Check')
print('=' * 90)
print()
print('VIF > 5: Moderate multicollinearity')
print('VIF > 10: High multicollinearity (consider removing)')
print()

X_for_vif = X.copy()

# Remove zero variance columns
zero_var_cols = X_for_vif.columns[X_for_vif.std() == 0].tolist()
if zero_var_cols:
    print(f'Columns with zero variance (excluded): {zero_var_cols}')
    X_for_vif = X_for_vif.drop(columns=zero_var_cols)
    print()

# Add constant for correct VIF calculation
X_vif_const = sm.add_constant(X_for_vif)

vif_data = []
for i, col in enumerate(X_for_vif.columns):
    try:
        vif = variance_inflation_factor(X_vif_const.values, i + 1)
        vif_data.append({'Feature': col, 'VIF': vif})
    except:
        vif_data.append({'Feature': col, 'VIF': np.nan})

vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)

print(f'{"Feature":<35} {"VIF":>10} {"Status":>15}')
print('-' * 60)
for _, row in vif_df.iterrows():
    if pd.isna(row['VIF']) or np.isinf(row['VIF']):
        status = 'N/A'
    elif row['VIF'] > 20:
        status = 'HIGH!'
    elif row['VIF'] > 10:
        status = 'Moderate'
    else:
        status = 'OK'
    vif_val = f"{row['VIF']:.2f}" if not (pd.isna(row['VIF']) or np.isinf(row['VIF'])) else 'inf'
    print(f"{row['Feature']:<35} {vif_val:>10} {status:>15}")

# Save VIF to CSV
vif_df.to_csv('artifacts/vif.csv', index=False)
print('\nSaved VIF to artifacts/vif.csv')

# Summary
high_vif = vif_df[vif_df['VIF'] > 10]
if len(high_vif) > 0:
    print(f'\nWARNING: {len(high_vif)} features with VIF > 10 (high multicollinearity)')
else:
    print('\nOK: No features with VIF > 10')
