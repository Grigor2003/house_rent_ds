import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Load coefficients
coefs = pd.read_csv('artifacts/coefficients.csv', index_col=0)

# Remove intercept
coefs = coefs[coefs.index != 'const']

# Sort by absolute value
coefs = coefs.sort_values('Coefficient', key=abs, ascending=True)

print("Coefficients:")
print(coefs[['Coefficient', 'Significant']].to_string())
print()

# Create horizontal bar chart (like heatmap rotated)
# Increase height based on number of coefficients to avoid overlap
n_coefs = len(coefs)
fig_height = max(10, n_coefs * 0.5)
fig, ax = plt.subplots(figsize=(14, fig_height))

colors = ['#d73027' if c > 0 else '#4575b4' for c in coefs['Coefficient']]

bars = ax.barh(range(len(coefs)), coefs['Coefficient'], color=colors, edgecolor='black', linewidth=0.5, height=0.7)

# Calculate x-axis limits with padding for text
x_min = coefs['Coefficient'].min()
x_max = coefs['Coefficient'].max()
x_range = x_max - x_min
x_padding = x_range * 0.15  # 15% padding for text labels
ax.set_xlim(x_min - x_padding, x_max + x_padding)

# Add coefficient values and p-values on bars (outside the bars to avoid overlap)
for i, (idx, row) in enumerate(coefs.iterrows()):
    x_pos = row['Coefficient']
    p_val = row['p-value']
    
    # Format p-value
    if p_val < 0.001:
        p_str = 'p<0.001'
    elif p_val < 0.01:
        p_str = f'p={p_val:.3f}'
    elif p_val < 0.05:
        p_str = f'p={p_val:.3f}'
    else:
        p_str = f'p={p_val:.2f}'
    
    # Place text outside bars with dynamic offset based on range
    offset = x_range * 0.02 if x_pos >= 0 else -x_range * 0.02
    label = f'{x_pos:.3f} ({p_str})'
    ax.text(x_pos + offset, i, label, va='center', ha='left' if x_pos >= 0 else 'right', 
            fontsize=9, fontweight='bold', color='black')

# Labels
ax.set_yticks(range(len(coefs)))
ax.set_yticklabels(coefs.index, fontsize=12)
ax.set_xlabel('Coefficient (effect on log-rent)', fontsize=14)
ax.set_title('Regression Coefficients\n(Red = increases rent, Blue = decreases rent)', fontsize=16, fontweight='bold')

# Add vertical line at 0
ax.axvline(x=0, color='black', linewidth=1)

# Calculate and plot mean for positive and negative coefficients
pos_coefs = coefs[coefs['Coefficient'] > 0]['Coefficient']
neg_coefs = coefs[coefs['Coefficient'] < 0]['Coefficient']

pos_mean = pos_coefs.mean()
neg_mean = neg_coefs.mean()

ax.axvline(x=pos_mean, color='#d73027', linewidth=2, linestyle='--', 
           label=f'Positive mean: {pos_mean:.3f}')
ax.axvline(x=neg_mean, color='#4575b4', linewidth=2, linestyle='--',
           label=f'Negative mean: {neg_mean:.3f}')

ax.legend(loc='lower right', fontsize=12)

print(f"Positive coefficients mean: {pos_mean:.4f}")
print(f"Negative coefficients mean: {neg_mean:.4f}")
print()

# Add grid
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('artifacts/coef_barplot.png', dpi=150, bbox_inches='tight')
print("Saved: artifacts/coef_barplot.png")

plt.show()

# Also create a 2D heatmap (single row) with TwoSlopeNorm
fig2, ax2 = plt.subplots(figsize=(20, 4))

# Sort by coefficient value for heatmap
coefs_sorted = coefs.sort_values('Coefficient', ascending=False)
values = coefs_sorted['Coefficient'].values.reshape(1, -1)

vmax = max(abs(values.min()), abs(values.max()))
im = ax2.imshow(values, cmap='seismic', norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), aspect='auto')

# Colorbar with mean lines
cbar = fig2.colorbar(im, ax=ax2, orientation='horizontal', pad=0.4, shrink=0.6)
cbar.set_label('Coefficient', fontsize=14)
cbar.ax.tick_params(labelsize=12)

# Add mean markers on colorbar
cbar.ax.axvline(x=(pos_mean + vmax) / (2 * vmax), color='darkred', linewidth=2, linestyle='--')
cbar.ax.axvline(x=(neg_mean + vmax) / (2 * vmax), color='darkblue', linewidth=2, linestyle='--')

# Custom ticks including means
ticks = sorted(set([-vmax, neg_mean, 0, pos_mean, vmax]))
cbar.set_ticks(ticks)
cbar.set_ticklabels([f'{t:.2f}' for t in ticks])

# Labels
ax2.set_xticks(range(len(coefs_sorted)))
ax2.set_xticklabels(coefs_sorted.index, rotation=60, ha='right', fontsize=12)
ax2.set_yticks([])
ax2.set_title(f'Coefficient Heatmap (Pos mean: {pos_mean:.3f}, Neg mean: {neg_mean:.3f})', fontsize=16, fontweight='bold')

# Add values
for j, val in enumerate(coefs_sorted['Coefficient']):
    color = 'white' if abs(val) > 0.3 else 'black'
    ax2.text(j, 0, f'{val:.2f}', ha='center', va='center', fontsize=11, fontweight='bold', color=color)

plt.tight_layout()
plt.savefig('artifacts/coef_heatmap.png', dpi=150, bbox_inches='tight')
print("Saved: artifacts/coef_heatmap.png")

plt.show()

print("\nInterpretation:")
print("  Red/Positive = Increases rent")
print("  Blue/Negative = Decreases rent")
