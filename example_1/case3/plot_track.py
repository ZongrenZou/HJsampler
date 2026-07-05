import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import scipy.io as sio


x = np.linspace(-1., 1., 101)
y = np.linspace(-1., 1., 101)
xx, yy = np.meshgrid(x, y)
xx = xx.reshape([-1, 1])
yy = yy.reshape([-1, 1])
inputs = np.concatenate([xx, yy], axis=-1)

# cov1 = 1/3 ** 2 * np.eye(2)
# cov2 = 1/5 ** 2 * np.eye(2)
cov1 = np.array(
    [
        [1/2**2, 0.05],
        [0.05, 1/3**2],
    ]
)
cov2 = np.array(
    [
        [1/4**2, -0.05],
        [-0.05, 1/2**2],
    ]
)
T = 1
eps = 0.5
mu1 = np.array([0.5, 0.5]).reshape([1, -1])
mu2 = np.array([-0.5, -0.5]).reshape([1, -1])


def fn(x):
    T1 = (x - mu1) @ np.linalg.inv(cov1)
    T1 = np.einsum("Nk,kN->N", T1, (x - mu1).T)
    f1 = 1 / np.sqrt(2*np.pi*np.linalg.det(cov1)) * np.exp(-T1/2)
    T2 = (x - mu2) @ np.linalg.inv(cov2)
    T2 = np.einsum("Nk,kN->N", T2, (x - mu2).T)
    f2 = 1 / np.sqrt(2*np.pi*np.linalg.det(cov2)) * np.exp(-T2/2)
    return 0.5 * f1 + 0.5 * f2


# data = sio.loadmat("./outputs/analytic.mat")
data = sio.loadmat("./outputs/sgm.mat")
zts = data["zts"]
out = fn(inputs)
f = out.reshape([101, 101])
levels = np.linspace(np.min(f), np.max(f), 20)


fig = plt.figure(figsize=[25, 4])
# ind = [4, 1, 2, 3, 6] # for analytical
ind = [9, 4, 2, 3, 8] # for sgm
for i in range(5):
    ax = fig.add_subplot(1, 5, i+1)
    c = ax.contourf(
        xx.reshape([101, 101]), 
        yy.reshape([101, 101]), 
        f, 
        cmap=colormaps["jet"],
        levels=levels,
    )

    idx = ind[i]
    ax.plot(zts[::5, idx, 0], zts[::5, idx, 1], "y-", alpha=1)
    ax.plot(zts[0, idx, 0], zts[0, idx, 1], "wo", label="$Y_T$", markersize=10)
    ax.plot(zts[-1, idx, 0], zts[-1, idx, 1], "w*", label="$Y_0$", markersize=10)
    out = fn(inputs)

    ax.set_aspect('equal')
    ax.set_xlim([-1., 1.])
    ax.set_ylim([-1., 1.])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

# fig.savefig("./figs/analytical_path.png")
fig.savefig("./figs/sgm_path.png")
plt.show()