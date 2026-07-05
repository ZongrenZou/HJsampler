import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import scipy.io as sio
from scipy.linalg import expm
from tqdm import trange
import tensorflow as tf


import models


def check_riccati(cov1, cov2, mu1, mu2, yt, eps):
    B = np.array(
        [
            [0, -1],
            [1, 1],
        ]
    )
    # first solve for P, q
    dt = 0.00001
    M = 100

    P1s = [1/eps * cov1]
    # q1s = [mu1]
    for i in range(int(np.ceil(T/dt))):
        P_update = np.eye(n) - P1s[-1] @ B.T - B @ P1s[-1]
        # q_update = (-B @ q1s[-1].T).T
        P1s += [P1s[-1] + dt * P_update]
        # q1s += [q1s[-1] + dt * q_update]
    P1s = P1s[::M]
    # q1s = q1s[::M]
    P1s = P1s[1:]
    # q1s = q1s[1:]
    P1s.reverse()
    # q1s.reverse()

    P2s = [1/eps * cov2]
    # q2s = [mu2]
    for i in range(int(np.ceil(T/dt))):
        P_update = np.eye(n) - P2s[-1] @ B.T - B @ P2s[-1]
        # q_update = (-B @ q2s[-1].T).T
        P2s += [P2s[-1] + dt * P_update]
        # q2s += [q2s[-1] + dt * q_update]
    P2s = P2s[::M]
    # q2s = q2s[::M]
    P2s = P2s[1:]
    # q2s = q2s[1:]
    P2s.reverse()
    # q2s.reverse()

    r_fn = lambda P: eps / 2 * np.log((2*np.pi*eps)**n * np.linalg.det(P))
    q1_fn = lambda t: (expm(-B*t) @ mu1.T).T
    q2_fn = lambda t: (expm(-B*t) @ mu2.T).T
    

    def s_fn(Y, t, i):
        T1 = (Y - q1_fn(t)) @ np.linalg.inv(P1s[i]) # shape of [N, 2]
        T1 = np.einsum("Ni,Ni->N", T1, Y - q1_fn(t))
        T2 = (Y - q2_fn(t)) @ np.linalg.inv(P2s[i]) # shape of [N, 2]
        T2 = np.einsum("Ni,Ni->N", T2, Y - q2_fn(t))
        # T1 = (Y - q1s[i]) @ np.linalg.inv(P1s[i]) # shape of [N, 2]
        # T1 = np.einsum("Ni,Ni->N", T1, Y - q1s[i])
        # T2 = (Y - q2s[i]) @ np.linalg.inv(P2s[i]) # shape of [N, 2]
        # T2 = np.einsum("Ni,Ni->N", T2, Y - q2s[i])
        B1 = np.exp(-1/2/eps * T1 - r_fn(P1s[i]) / eps)
        B2 = np.exp(-1/2/eps * T2 - r_fn(P2s[i]) / eps)

        A = 0.5 * np.linalg.inv(P1s[i]) @ (Y - q1_fn(t)).T * B1 + \
            0.5 * np.linalg.inv(P2s[i]) @ (Y - q2_fn(t)).T * B2
        # A = 0.5 * np.linalg.inv(P1s[i]) @ (Y - q1s[i]).T * B1 + \
        #     0.5 * np.linalg.inv(P2s[i]) @ (Y - q2s[i]).T * B2
        B = 0.5 * B1 + 0.5 * B2
        return - A.T / B[:, None]
    

    dt = 0.001
    zt = yt * np.ones(shape=[100000, 2])
    t = 0
    for i in trange(int(T/dt)):
        update = s_fn(zt, T-t, i) + (B @ zt.T).T
        zt = zt + update * dt + np.sqrt(eps*dt) * np.random.normal(size=zt.shape)
        t = t + dt
    
    return zt


def check_sgm(yt, eps):
    B = np.array(
        [
            [0, -1],
            [1, 1],
        ]
    )

    N = 100000
    model = models.NN(units=50, name="nn_{}".format(str(N)), eps=np.array(eps), activation=tf.tanh)
    model.restore()

    dt = 0.001
    t = 0
    M = 100000
    zts = [np.tile(yt, [M, 1])]
    for i in trange(int(T/dt)):
        update = eps * model.call(
            tf.constant(zts[-1][:, 0:1], tf.float32),
            tf.constant(zts[-1][:, 1:2], tf.float32),
            (T - t) * tf.ones(shape=[M, 1]),
        ).numpy()
        b = (-B @ zts[-1].T).T
        zt = zts[-1] + (update - b) * dt + \
            np.sqrt(eps * dt) * np.random.normal(size=zts[-1].shape)
        zts += [zt]
        t = t + dt
    zts = np.stack(zts, axis=0)
    return zts[-1, ...]


n = 2
cov1 = np.array(
    [
        [0.5 ** 2, 0.1],
        [0.1, 0.4 ** 2],
    ]
)
cov2 = np.array(
    [
        [0.5 ** 2, -0.1],
        [-0.1, 0.4 ** 2],
    ]
)
mu1 = np.array([-0.7, 0]).reshape([1, 2])
mu2 = np.array([0.7, 0]).reshape([1, 2])

alpha = 1
eps = 1
T = 1

## case 1: 
# yt = np.array([0.5, -0.5]).reshape([1, 2])
# yt = np.array([0., -0.5]).reshape([1, 2])
## case 2: 
# yt = np.array([-1, 0.5]).reshape([1, 2])
## case 3: 
# yt = np.array([-0.7, -0.9]).reshape([1, 2])
## case 4: 
yt = np.array([0.5, 1]).reshape([1, 2])


zt_riccati = check_riccati(
    cov1=cov1,
    cov2=cov2,
    mu1=mu1,
    mu2=mu2,
    yt=yt, 
    eps=eps,
)
zt_sgm = check_sgm(
    yt=yt, eps=eps
)

######################### Make plot #########################
fig = plt.figure(figsize=[10, 5], dpi=100)

ax = fig.add_subplot(1, 2, 1)
h = ax.hist2d(
    zt_riccati[:, 0], zt_riccati[:, 1], bins=(100, 100), density=True, 
    cmap=colormaps["jet"],
    vmin=0,
    vmax=0.8,
)
ax.plot(yt[0, 0], yt[0, 1], "w*", markersize=10)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_aspect("equal")
fig.colorbar(h[3], ax=ax)
ax.set_title("Posterior samples: Riccati-HJ sampler")

ax = fig.add_subplot(1, 2, 2)
h = ax.hist2d(
    zt_sgm[:, 0], zt_sgm[:, 1], bins=(100, 100), density=True, 
    cmap=colormaps["jet"],
    vmin=0,
    vmax=0.8,
)
ax.plot(yt[0, 0], yt[0, 1], "w*", markersize=10)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_aspect("equal")
fig.colorbar(h[3], ax=ax)
ax.set_title("Posterior samples: SGM-HJ sampler")

fig.savefig("./figs/case1_4.png")
fig.show()

