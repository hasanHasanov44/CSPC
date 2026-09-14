import time
from decay import simulate, simulate_loop

# 200,000 atom üçün göstəricilər
N0 = 200000
rate = 0.4

# 1. Pure-Python loop versiyasının vaxtını ölçürük
start_loop = time.perf_counter()
res_loop = simulate_loop(N0, rate)
end_loop = time.perf_counter()
time_loop = end_loop - start_loop

# 2. NumPy (vectorized) simulate versiyasının vaxtını ölçürük
start_numpy = time.perf_counter()
res_numpy = simulate(N0, rate)
end_numpy = time.perf_counter()
time_numpy = end_numpy - start_numpy

# Sürətlənmə əmsalını hesablayırıq
speedup = time_loop / time_numpy

# Nəticələri çap edirik
print(f"loop:     {time_loop:.4f} s")
print(f"numpy:    {time_numpy:.4f} s")
print(f"speed-up: {speedup:.2f}x faster")