import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt('trajectory.csv', delimiter=',', skiprows=1)
t = data[:, 0]
x = data[:, 1]
y = data[:, 2]


vx = np.gradient(x, t)
vy = np.gradient(y, t)


speed = np.sqrt(vx**2 + vy**2)


plt.figure(figsize=(10, 4))


plt.subplot(1, 2, 1)
plt.plot(x, y, 'b-', label='2D Path')
plt.title('2D Trajectory (x vs y)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.grid(True)
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(t, speed, 'r-', label='Speed')
plt.title('Speed over Time')
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('trajectory.png')
print("Bonus completed! Saved trajectory.png successfully.")
plt.show()