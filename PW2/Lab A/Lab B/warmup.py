import numpy as np
from scipy.optimize import newton, minimize

# ---------- 2A: easy convex function ----------
def f(x):   return (x-3)**2 + 1
def df(x):  return 2*(x-3)
def d2f(x): return 2.0

print("=== PART 2A ===")

# (1) Gradient descent manual
x_gd = 0.0
lr = 0.1
tolerance = 1e-6
max_iter = 1000

for _ in range(max_iter):
    grad = df(x_gd)
    step = lr * grad
    x_gd -= step
    if abs(step) < tolerance:
        break

print(f"(1) Gradient Descent: x = {x_gd:.6f}, f(x) = {f(x_gd):.6f}")

# (2) Newton's method
x_newton = newton(df, 0.0, fprime=d2f)
print(f"(2) Newton's method:  x = {x_newton:.6f}, f(x) = {f(x_newton):.6f}")

# (3) SLSQP
res_slsqp = minimize(f, 0.0, method="SLSQP")
print(f"(3) SLSQP minimize:   x = {res_slsqp.x[0]:.6f}, f(x) = {res_slsqp.fun:.6f}")


# ---------- 2B: harder landscape ----------
def g(x):   return x**4 - 3*x**2 + x + 5
def dg(x):  return 4*x**3 - 6*x + 1
def d2g(x): return 12*x**2 - 6

def run_2b(x0):
    print(f"\n--- Starting from x0 = {x0} ---")
    
    # (1) Gradient descent
    x_gd = float(x0)
    lr = 0.01
    for _ in range(2000):
        step = lr * dg(x_gd)
        x_gd -= step
        if abs(step) < 1e-6:
            break
    print(f"Gradient Descent: x = {x_gd:.6f}, g(x) = {g(x_gd):.6f}")

    # (2) Newton
    try:
        x_newt = newton(dg, x0, fprime=d2g)
        second_deriv = d2g(x_newt)
        point_type = "Minimum" if second_deriv > 0 else ("Maximum" if second_deriv < 0 else "Saddle point")
        print(f"Newton's method:  x = {x_newt:.6f}, g(x) = {g(x_newt):.6f} (d2g = {second_deriv:.2f} -> {point_type})")
    except Exception as e:
        print(f"Newton's method failed: {e}")

    # (3) SLSQP
    res = minimize(g, x0, method="SLSQP")
    print(f"SLSQP minimize:   x = {res.x[0]:.6f}, g(x) = {res.fun:.6f}")

print("\n=== PART 2B ===")
run_2b(0.0)
run_2b(2.0)