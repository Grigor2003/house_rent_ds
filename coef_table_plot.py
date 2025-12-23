import pandas as pd
import matplotlib.pyplot as plt

# Load coefficients
coefs = pd.read_csv('artifacts/coefficients.csv', index_col=0)

# Remove intercept
coefs = coefs[coefs.index != 'const']

# Sort by absolute coefficient value
coefs = coefs.sort_values('Coefficient', key=abs, ascending=False)

# Prepare data for table
table_data = []
for idx, row in coefs.iterrows():
    coef = row['Coefficient']
    std_err = row['Std Error']
    ci_low = row['CI_Low']
    ci_high = row['CI_High']
    p_val = row['p-value']
    
    # Format p-value
    if p_val < 0.0001:
        p_str = '<0.0001 ***'
    elif p_val < 0.001:
        p_str = f'{p_val:.4f} ***'
    elif p_val < 0.01:
        p_str = f'{p_val:.4f} **'
    elif p_val < 0.05:
        p_str = f'{p_val:.4f} *'
    else:
        p_str = f'{p_val:.4f}'
    
    table_data.append([
        idx,
        f'{coef:.4f}',
        f'{std_err:.4f}',
        f'[{ci_low:.3f}, {ci_high:.3f}]',
        p_str
    ])

# Create figure and axis
fig, ax = plt.subplots(figsize=(16, 12))
ax.axis('off')

# Column headers
columns = ['Feature', 'Coefficient', 'Std Error', '95% CI', 'p-value']

# Create table
table = ax.table(
    cellText=table_data,
    colLabels=columns,
    cellLoc='center',
    loc='center',
    colColours=['#4472C4'] * len(columns)
)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.0)

# Style header
for i in range(len(columns)):
    table[(0, i)].set_text_props(weight='bold', color='white')
    table[(0, i)].set_fontsize(12)

# Color rows based on significance and direction
for i, row in enumerate(table_data):
    row_idx = i + 1  # +1 because of header
    coef_val = float(row[1])
    p_val_str = row[4]
    
    # Determine significance
    is_significant = '***' in p_val_str or '**' in p_val_str or ('*' in p_val_str and p_val_str.count('*') == 1)
    
    if is_significant:
        if coef_val > 0:
            bg_color = '#FFCCCC'  # Light red for positive significant
        else:
            bg_color = '#CCE5FF'  # Light blue for negative significant
    else:
        bg_color = '#F0F0F0'  # Gray for non-significant
    
    for j in range(len(columns)):
        table[(row_idx, j)].set_facecolor(bg_color)

# Add title
ax.set_title('Regression Coefficients Summary\n(Log-Linear Model: log(Rent) ~ Features)', 
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('artifacts/coef_table.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: artifacts/coef_table.png")

plt.show()

# Also save as CSV for easy copy-paste
coefs_export = coefs[['Coefficient', 'Std Error', 'CI_Low', 'CI_High', 'p-value', 'Significant']].copy()
coefs_export.to_csv('artifacts/coefficients_summary.csv')
print("Saved: artifacts/coefficients_summary.csv")