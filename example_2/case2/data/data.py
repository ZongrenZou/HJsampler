import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


def solve(y0, t, eps=0.01):
    ys = [y0]
    dt = t[1] - t[0]
    for i in range(t.shape[0] - 1):
        y = ys[-1]
        y1 = y[0:1] + y[1:2] * dt
        y2 = y[1:2] + (-1 * y[0:1] - 1 * y[1:2]) * dt + np.sqrt(eps * dt) * np.random.normal(size=y[0:1].shape)
        ys += [np.concatenate([y1, y2], axis=0)]
    return ys


T = 5
t = np.linspace(0, T, 501).reshape([-1, 1])
y0 = np.random.lognormal(mean=-2, sigma=0.5, size=[2, 100000])

# case 1: eps = 0.0001
# case 2: eps = 0.0005
# case 3: eps = 0.001
# case 4: eps = 0.00001
eps = 0.00001
ys = solve(y0, t, eps=eps)
ys = np.stack(ys, axis=0)


sio.savemat(
    "./train4.mat",
    {
        "t": t,
        "ys": ys,
        "eps": eps,
        "T": T
    }
)