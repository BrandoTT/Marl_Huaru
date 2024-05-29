import numpy as np

def decay_coefficient(step, total_steps):
    decay_rate = -np.log(0.001) / total_steps  # 衰减速率，这里假设最终值为0.01
    return np.exp(-decay_rate * step)

total_steps = 600
coefficients = [decay_coefficient(step, total_steps) for step in range(total_steps)]

# 绘制衰减曲线
import matplotlib.pyplot as plt
plt.plot(coefficients)
plt.xlabel('Steps')
plt.ylabel('Coefficient')
plt.title('Decay Curve')
plt.show()
