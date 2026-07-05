import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import scipy.io as sio

from tqdm import trange


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


def C(t, cov):
    return np.sqrt((2*np.pi) ** 2 * np.linalg.det(cov + eps * t * np.eye(2)))


def s(Y, t):
    T1 = (Y - mu1) @ np.linalg.inv(cov1 + eps * t * np.eye(2))
    T1 = np.einsum("Nk,kN->N", T1, (Y - mu1).T)
    T2 = (Y - mu2) @ np.linalg.inv(cov2 + eps * t * np.eye(2))
    T2 = np.einsum("Nk,kN->N", T2, (Y - mu2).T)
    A = 1/2 * 1/C(t, cov1) * np.exp(-T1/2) + 1/2 * 1/C(t, cov2) * np.exp(-T2/2)
    
    B1 = 1/C(t, cov1) * np.linalg.inv(cov1 + eps*t*np.eye(2)) @ (Y - mu1).T * np.exp(-T1/2)
    B2 = 1/C(t, cov2) * np.linalg.inv(cov2 + eps*t*np.eye(2)) @ (Y - mu2).T * np.exp(-T2/2)
    
    return - (0.5 * B1.T + 0.5 * B2.T) / A[:, None]


## sampling
yt = np.array([-0.9, 0.9]).reshape([1, -1])
# yt = np.array([0.9, 0.5]).reshape([1, -1])

dt = 0.001
t = 0
zts = [np.tile(yt, [100000, 1])]
for i in trange(int(T/dt)):
    zt = zts[-1] + eps * s(zts[-1], T-t) * dt + \
        np.sqrt(eps) * np.random.normal(size=zts[-1].shape) * np.sqrt(dt)
    zts += [zt]
    t = t + dt
zts = np.stack(zts, axis=0)
sio.savemat(
    "./outputs/analytic.mat",
    {"zts": zts[:, ::1000, :]},
)


## plot posterior histogram
plt.figure()
plt.hist2d(
    zts[-1, :, 0], 
    zts[-1, :, 1], 
    bins=(50, 50), 
    density=True, 
    cmap=colormaps["jet"], 
    # vmin=0, 
    # vmax=1,
)
plt.clim(0, 1)
plt.colorbar()
plt.xlim([-1., 1.])
plt.ylim([-1., 1.])
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.savefig("./figs/analytical.png")
plt.show()


## plot the exact density function
x = np.linspace(-1., 1., 501)
y = np.linspace(-1., 1., 501)
xx, yy = np.meshgrid(x, y)
xx = xx.reshape([-1, 1])
yy = yy.reshape([-1, 1])
inputs = np.concatenate([xx, yy], axis=-1)

T1 = (inputs - mu1) @ np.linalg.inv(cov1)
T1 = np.einsum("Nk,kN->N", T1, (inputs - mu1).T)
T2 = (inputs - mu2) @ np.linalg.inv(cov2)
T2 = np.einsum("Nk,kN->N", T2, (inputs - mu2).T)
A = 1/2 * 1/C(0, cov1) * np.exp(-T1/2) + 1/2 * 1/C(0, cov2) * np.exp(-T2/2)

D = 1/np.sqrt((2*np.pi*eps*T) ** 2) * np.exp(-1/2/eps/T*np.sum((inputs-yt) ** 2, axis=-1))

T1 = (yt - mu1) @ np.linalg.inv(cov1 + eps * T * np.eye(2))
T1 = np.einsum("Nk,kN->N", T1, (yt - mu1).T)
T2 = (yt - mu2) @ np.linalg.inv(cov2 + eps * T * np.eye(2))
T2 = np.einsum("Nk,kN->N", T2, (yt - mu2).T)
B = 1/2 * 1/C(T, cov1) * np.exp(-T1/2) + 1/2 * 1/C(T, cov2) * np.exp(-T2/2)

f = A * D / B
f = f.reshape([501, 501])

levels = np.linspace(np.min(f), np.max(f), 20)

plt.figure()
# c = plt.contourf(
#     xx.reshape([101, 101]), 
#     yy.reshape([101, 101]), 
#     f, 
#     cmap=colormaps["jet"],
#     levels=levels,
#     vmin=0,
#     vmax=2,
# )
plt.pcolormesh(
    xx.reshape(f.shape), 
    yy.reshape(f.shape), 
    f, 
    cmap=colormaps["jet"],
    # vmin=0,
    # vmax=1,  
    # shading="gouraud",
)
plt.clim(0, 1)
plt.colorbar()
plt.xlim([-1., 1.])
plt.ylim([-1., 1.])
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.savefig("./figs/exact.png")
plt.show()
