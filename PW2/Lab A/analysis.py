"""
PW2 Lab A -- Motion from tracking data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

data = np.loadtxt('freefall.csv', delimiter=',', skiprows=1)
t = data[:, 0]
y = data[:, 1]


v = np.gradient(y, t)
a = np.gradient(v, t)

mean_a = np.mean(a)
std_a = a.std()
print(f"Mean acceleration: {mean_a:.4f} m/s^2")
print(f"Standard deviation of acceleration: {std_a:.4f} m/s^2")


v_rec = cumulative_trapezoid(a, t, initial=0) + v[0]
y_rec = cumulative_trapezoid(v_rec, t, initial=0) + y[0]

max_diff = np.max(np.abs(y - y_rec))
print(f"Max position difference after integration: {max_diff:.4f} m")


fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)


axs[0].plot(t, y, label='Measured Position (y)', color='blue')
axs[0].set_ylabel('Position (m)')
axs[0].legend()
axs[0].grid(True)


axs[1].plot(t, v, label='Calculated Velocity (v)', color='orange')
axs[1].set_ylabel('Velocity (m/s)')
axs[1].legend()
axs[1].grid(True)


axs[2].plot(t, a, label='Calculated Acceleration (a)', color='red')
axs[2].axhline(-9.81, color='black', linestyle='--', label='g = -9.81 m/s²')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Acceleration (m/s²)')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.savefig('motion.png')
print("Saved motion.png successfully!")
plt.show()