import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import scipy.io as sio
import tensorflow as tf
from tqdm import trange
from scipy.integrate import odeint


import models


B = np.array(
    [
        [0, -1],
        [1, 1],
    ]
)


def rhs(y, t):
    y1, y2 = y
    dydt = [y2, -1*y1 - 1*y2 + 1*y1**2]
    return dydt


T = 5

## case 1: 
y0 = np.array([0.2, 0.1])
t = np.linspace(0, T, 5001)
sols = odeint(rhs, y0, t)
yt = sols[-1, :].reshape([1, 2])


def rhs_rev(y, t):
    y1, y2 = y
    dydt = [-y2, +1*y1 + 1*y2]
    return dydt


z0 = sols[-1, :]
t = np.linspace(0, 5, 5001)
z_sols = odeint(rhs_rev, z0, t)


## sampling
# case 1: eps=0.0001, nn1
# case 2: eps=0.0005, nn2
# case 3: eps=0.001, nn3
# case 4: eps=0.00001, nn4
eps = 0.00001
model = models.NN(units=50, name="nn4", eps=np.array(eps), activation=tf.tanh)
model.restore()


dt = 0.001
t = 0
M = 1000
zts = [np.tile(yt, [M, 1])]
forward = tf.function(model.call)

for i in trange(int(T/dt)):
    update = eps * forward(
        tf.constant(zts[-1][:, 0:1], tf.float32),
        tf.constant(zts[-1][:, 1:2], tf.float32),
        (T - t) * tf.ones(shape=[M, 1]),
    ).numpy()
    b = (-B @ zts[-1].T).T
    zts += [
        zts[-1] + (update - b) * dt + \
        np.sqrt(eps * dt) * np.random.normal(size=zts[-1].shape) * np.array([0, 1]).reshape([1, 2])
    ]
    t = t + dt

zts = np.stack(zts, axis=0)
mu = np.mean(zts, axis=1)
sd = np.std(zts, axis=1)

sio.savemat(
    "./outputs/sgm4.mat",
    {
        "t": np.linspace(0, T, 5001),
        "mu": mu,
        "sd": sd,
        "sols": sols,
        "z_sols": z_sols,
    }
)

######################### Make plot #########################
# fig = plt.figure(dpi=100)

# t = np.linspace(0, 5, 5001)
# plt.plot(t, sols[:, 0], "k-", label="Reference of $u_1$")
# plt.plot(t, sols[:, 1], "b-", label="Reference of $u_2$")
# plt.plot(t, np.flip(z_sols[:, 0]), "r--")
# plt.plot(t, np.flip(z_sols[:, 1]), "r--")
# plt.plot(t, np.flip(mu[:, 0]), "r--", label="Mean of $u_1$")
# plt.plot(t, np.flip(mu[:, 1]), "g--", label="Mean of $u_2$")
# plt.fill_between(t.flatten(), np.flip(mu[:, 0] + 2*sd[:, 0]), np.flip(mu[:, 0] - 2*sd[:, 0]), alpha=0.3)
# plt.fill_between(t.flatten(), np.flip(mu[:, 1] + 2*sd[:, 1]), np.flip(mu[:, 1] - 2*sd[:, 1]), alpha=0.3)
# plt.ylim([-0.5, 1])
# plt.xlabel("T")
# plt.legend()
# plt.title("SGM HJ sampler: $\epsilon=0.0001$")
# plt.savefig("./figs/sgm_case1.png")
# plt.show()

print("End main.")