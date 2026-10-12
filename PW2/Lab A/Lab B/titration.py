import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TODO 1: Read titration.csv
data = pd.read_csv('titration.csv')
V = data['volume_base'].values
pH = data['pH'].values

# TODO 2: Compute slope and find equivalence point
dpH_dV = np.gradient(pH, V)
idx_equiv = np.argmax(dpH_dV)
v_equiv = V[idx_equiv]
pH_equiv = pH[idx_equiv]

print(f"Equivalence point is at Volume = {v_equiv:.2f} mL (pH = {pH_equiv:.2f})")

# TODO 3: Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left plot: Titration curve
ax1.plot(V, pH, 'b-', label='pH curve')
ax1.axvline(v_equiv, color='r', linestyle='--', label=f'Equiv Point ({v_equiv:.1f} mL)')
ax1.set_xlabel('Volume of Base (mL)')
ax1.set_ylabel('pH')
ax1.set_title('Titration Curve')
ax1.legend()
ax1.grid(True)

# Right plot: Derivative (slope)
ax2.plot(V, dpH_dV, 'g-', label='dpH/dV')
ax2.axvline(v_equiv, color='r', linestyle='--', label=f'Peak at {v_equiv:.1f} mL')
ax2.set_xlabel('Volume of Base (mL)')
ax2.set_ylabel('dpH / dV')
ax2.set_title('First Derivative (Slope)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('titration.png')
plt.show()