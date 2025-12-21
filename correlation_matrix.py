import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

X = pd.read_csv('artifacts/X_train_arr.csv')
corr = X.corr()

# Replace NaN with 0 (NaN occurs for constant columns)
corr = corr.fillna(0)

vmin, vmax = corr.values.min(), corr.values.max()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, cmap="seismic", norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))

cbar = fig.colorbar(im, ax=ax)
# Sort unique ticks: -1, vmin, 0, vmax, 1
ticks = sorted(set([-1, round(vmin, 2), 0, round(vmax, 2), 1]))
cbar.set_ticks(ticks)
# Label vmin/vmax specially
labels = []
for t in ticks:
    if t == round(vmin, 2) and t not in [-1, 0, 1]:
        labels.append(f'{t:.2f} (min)')
    elif t == round(vmax, 2) and t not in [-1, 0, 1]:
        labels.append(f'{t:.2f} (max)')
    else:
        labels.append(f'{t:.2f}')
cbar.set_ticklabels(labels)

ax.set_xticks(range(len(corr)))
ax.set_yticks(range(len(corr)))
# ax.set_xticklabels(corr.columns, rotation=90)
# ax.set_yticklabels(corr.columns)
plt.tight_layout()
plt.show()
