import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# TODO 1: Read kinetics.csv
data = pd.read_csv('kinetics.csv')
t = data['time'].values
C = data['concentration'].values
C0 = C[0]

# TODO 2: Define total_error(k)
def total_error(k):
    k_val = k[0] if isinstance(k, (list, np.ndarray)) else k
    model = C0 * np.exp(-k_val * t)
    return np.sum((C - model)**2)

# TODO 3: Minimize total_error
res = minimize(total_error, x0=[0.5], method="SLSQP", bounds=[(0, 5)])
k_fitted = res.x[0]

print(f"Fitted rate constant k = {k_fitted:.4f}")

# TODO 4: Plot data and fitted curve
plt.figure(figsize=(8, 5))
plt.scatter(t, C, color='red', label='Measured Data', zorder=5)

t_dense = np.linspace(min(t), max(t), 200)
C_dense = C0 * np.exp(-k_fitted * t_dense)
plt.plot(t_dense, C_dense, 'b-', label=f'Fitted Curve: C0*exp(-{k_fitted:.3f}*t)')

plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('First-Order Reaction Kinetics')
plt.legend()
plt.grid(True)
plt.savefig('kinetics.png')
plt.show()