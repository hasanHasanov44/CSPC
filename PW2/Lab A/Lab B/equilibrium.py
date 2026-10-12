import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton, minimize

K = 15.6
a = b = 1.0

# TODO 1: k_imbalance
def k_imbalance(x):
    return ((2*x)**2) / ((a - x) * (b - x)) - K

# TODO 2: Method 1 (Newton root-finding)
x_newton = newton(k_imbalance, 0.5)
print(f"Method 1 (Newton):   x = {x_newton:.6f}")

# TODO 3: Method 2 (SLSQP minimize square error)
def objective(x):
    return k_imbalance(x[0])**2

res = minimize(objective, x0=[0.5], method="SLSQP", bounds=[(0, 0.999)])
x_slsqp = res.x[0]
print(f"Method 2 (SLSQP):    x = {x_slsqp:.6f}")

# TODO 4: Report amounts and plot
x_eq = x_newton
h2_eq = a - x_eq
i2_eq = b - x_eq
hi_eq = 2 * x_eq

print(f"\nEquilibrium Amounts (x = {x_eq:.4f}):")
print(f"H2 : {h2_eq:.4f} mol")
print(f"I2 : {i2_eq:.4f} mol")
print(f"HI : {hi_eq:.4f} mol")

# Plot composition vs x
x_vals = np.linspace(0, 0.99, 200)
h2_vals = a - x_vals
i2_vals = b - x_vals
hi_vals = 2 * x_vals

plt.figure(figsize=(8, 5))
plt.plot(x_vals, h2_vals, label='H2 = 1 - x')
plt.plot(x_vals, i2_vals, label='I2 = 1 - x', linestyle='--')
plt.plot(x_vals, hi_vals, label='HI = 2x')

plt.axvline(x_eq, color='red', linestyle=':', label=f'Equilibrium (x = {x_eq:.3f})')
plt.xlabel('Reaction Extent (x)')
plt.ylabel('Amount (moles)')
plt.title('Chemical Equilibrium: H2 + I2 <=> 2 HI')
plt.legend()
plt.grid(True)
plt.savefig('equilibrium.png')
plt.show()