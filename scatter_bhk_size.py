import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Data/rent.csv')

plt.figure(figsize=(10, 8))
plt.scatter(df['BHK'], df['Size'], alpha=0.5, c='#4361EE', edgecolors='none', s=30)

z = np.polyfit(df['BHK'], df['Size'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['BHK'].min(), df['BHK'].max(), 100)
plt.plot(x_line, p(x_line), color='#F72585', linewidth=2, linestyle='--', label=f'y = {z[0]:.2f}x + {z[1]:.2f}')

correlation = np.corrcoef(df['BHK'], df['Size'])[0, 1]
r_squared = correlation ** 2

plt.xlabel('BHK', fontsize=12, fontweight='bold')
plt.ylabel('Size (sq ft)', fontsize=12, fontweight='bold')
plt.title(f'BHK vs Size Scatter Plot\nR² = {r_squared:.4f}', fontsize=14, fontweight='bold')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig('bhk_vs_size_scatter.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Correlation coefficient: {correlation:.4f}")
print(f"R² score: {r_squared:.4f}")
print(f"Regression equation: Size = {z[0]:.2f} * BHK + {z[1]:.2f}")



