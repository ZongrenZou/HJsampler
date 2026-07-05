import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from matplotlib import colormaps


x = np.linspace(0, 1, 100).reshape([1, -1])
N = 100000
A1 = 1 + 2 * np.random.uniform(size=[N, 1])
A2 = 1 + 2 * np.random.uniform(size=[N, 1])
A3 = 1 + 2 * np.random.uniform(size=[N, 1])
A4 = 1 + 2 * np.random.uniform(size=[N, 1])
A5 = 1 + 2 * np.random.uniform(size=[N, 1])
A6 = 1 + 2 * np.random.uniform(size=[N, 1])
A7 = 1 + 2 * np.random.uniform(size=[N, 1])
A8 = 1 + 2 * np.random.uniform(size=[N, 1])
# b = 0.1 * np.pi * np.random.uniform(size=[100, 1])
y = A1 * np.sin(1 * np.pi * x) + \
    A2 * np.sin(2 * np.pi * x) + \
    A3 * np.sin(3 * np.pi * x) + \
    A4 * np.sin(4 * np.pi * x) + \
    A5 * np.sin(5 * np.pi * x) + \
    A6 * np.sin(6 * np.pi * x) + \
    A7 * np.sin(7 * np.pi * x) + \
    A8 * np.sin(8 * np.pi * x)
y0 = y / 16

x = x.T
y0 = y0.T


Nt = 51
T = 1
eps = 0.01
t = np.linspace(0, T, Nt)
ys = [y0]
dt = t[1] - t[0]
print(dt)
for i in range(int(T/dt)):
    ys += [ys[-1] + np.sqrt(dt * eps) * np.random.normal(size=y0.shape)]
ys = np.stack(ys)
ys = np.transpose(ys, [2, 0, 1])

sio.savemat(
    "./train.mat",
    {
        "t": t,
        "ys": ys,
        "eps": eps,
    }
)
