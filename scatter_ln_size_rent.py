import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Data/rent.csv')

ln_size = np.log(df['Size'])
ln_rent = np.log(df['Rent'])

plt.figure(figsize=(10, 8))
plt.scatter(ln_size, ln_rent, alpha=0.3, c='#E84855', edgecolors='none', s=30)

z = np.polyfit(ln_size, ln_rent, 1)
p = np.poly1d(z)
x_line = np.linspace(ln_size.min(), ln_size.max(), 100)
plt.plot(x_line, p(x_line), color='#2D3142', linewidth=2, linestyle='--', label=f'y = {z[0]:.3f}x + {z[1]:.3f}')

correlation = np.corrcoef(ln_size, ln_rent)[0, 1]
r_squared = correlation ** 2

plt.xlabel('ln(Size)', fontsize=12, fontweight='bold')
plt.ylabel('ln(Rent)', fontsize=12, fontweight='bold')
plt.title(f'ln(Size) vs ln(Rent) Scatter Plot\nR² = {r_squared:.4f}', fontsize=14, fontweight='bold')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig('ln_size_vs_ln_rent_scatter.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Correlation coefficient: {correlation:.4f}")
print(f"R² score: {r_squared:.4f}")
print(f"Regression equation: ln(Rent) = {z[0]:.4f} * ln(Size) + {z[1]:.4f}")

