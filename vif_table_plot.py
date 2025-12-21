import pandas as pd
import matplotlib.pyplot as plt

# Load VIF data
vif = pd.read_csv('artifacts/vif.csv')

# Rename columns if needed
if 'Unnamed: 0' in vif.columns:
    vif = vif.rename(columns={'Unnamed: 0': 'Feature'})

# Sort by VIF descending
vif = vif.sort_values('VIF', ascending=False)

print("VIF values:")
print(vif.to_string(index=False))
print()

# Prepare data for table
table_data = []
for idx, row in vif.iterrows():
    feature = row['Feature']
    vif_val = row['VIF']
    
    # Determine status
    if vif_val > 10:
        status = 'HIGH ⚠️'
    elif vif_val > 5:
        status = 'Moderate'
    else:
        status = 'OK ✓'
    
    table_data.append([
        feature,
        f'{vif_val:.2f}',
        status
    ])

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 10))
ax.axis('off')

# Column headers
columns = ['Feature', 'VIF', 'Status']

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

# Color rows based on VIF value
for i, row in enumerate(table_data):
    row_idx = i + 1  # +1 because of header
    vif_val = float(row[1])
    
    if vif_val > 10:
        bg_color = '#FFCCCC'  # Light red - high multicollinearity
    elif vif_val > 5:
        bg_color = '#FFE5CC'  # Light orange - moderate
    else:
        bg_color = '#CCFFCC'  # Light green - OK
    
    for j in range(len(columns)):
        table[(row_idx, j)].set_facecolor(bg_color)

# Add title
ax.set_title('Variance Inflation Factor (VIF)\nMulticollinearity Check', 
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('artifacts/vif_table.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: artifacts/vif_table.png")

plt.show()

