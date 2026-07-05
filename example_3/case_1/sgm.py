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


def rhs(y, t):
    dydt = np.sin(4*np.pi*t) + 3 * y * y
    return dydt


def rhs_wrong(y, t):
    dydt = np.sin(4*np.pi*t) + 1 * y * y
    return dydt


T = 1


# y0 = np.array([0.0])
# t = np.linspace(0, 1, 1001)
# sols = odeint(rhs_wrong, y0, t)
# yt = sols[-1, :].reshape([1, 1])


def rhs_rev(y, t):
    dydt = - np.sin(4*np.pi*(T - t)) - 3 * y * y
    return dydt


def rhs_wrong_rev(y, t):
    dydt = - np.sin(4*np.pi*(T - t)) - 1 * y * y
    return dydt


# z0 = sols[-1, :]
z0 = np.array([0.1])
yt = z0
t = np.linspace(0, 1, 1001)
z_sols = odeint(rhs_wrong_rev, z0, t)
sols = odeint(rhs_rev, z0, t)
sols = np.flip(sols, axis=0)

plt.plot(t, sols, "k-")
plt.plot(t, np.flip(z_sols, axis=0), "r--")
plt.show()


## sampling
## nn1: eps = 0.001
## nn2: eps = 0.005
## nn3: eps = 0.01
## nn4: eps = 0.0001
## nn5: eps = 0.0005
eps = 0.001
model = models.NN(units=50, name="nn1", eps=np.array(eps), sigma=0.05, activation=tf.tanh)
model.restore()


dt = 0.001
t = 0
M = 100000
forward = tf.function(model.call)
zts = [np.tile(yt, [M, 1])]
for i in trange(int(np.ceil(T/dt))):
    update = eps * forward(
        (T - t) * tf.ones(shape=[M, 1]),
        tf.constant(zts[-1], tf.float32),
    ).numpy()
    b = np.sin(4*np.pi*(T - t)) + 1 * zts[-1] ** 2
    zts += [
        zts[-1] + (update - b) * dt + \
        np.sqrt(eps * dt) * np.random.normal(size=zts[-1].shape)
    ]
    t = t + dt

t = np.linspace(0, 1, 1001)
zts = np.stack(zts, axis=0)
mu = np.mean(zts, axis=1)
sd = np.std(zts, axis=1)
mu = np.flip(mu, axis=0)
sd = np.flip(sd, axis=0)


sio.savemat(
    "./outputs/sgm1.mat",
    {
        "t": t,
        "z_sols": z_sols,
        "sols": sols,
        "mu": mu, "sd": sd,
    }
)

plt.plot(t, sols, "k-", label="forward")
plt.plot(t, np.flip(z_sols, axis=0), "b--", label="misspecified")
plt.plot(t, mu, "r--", label="mean")
plt.fill_between(t.flatten(), (mu+2*sd).flatten(), (mu-2*sd).flatten(), alpha=0.3, label="2 SD")
plt.legend()
# plt.ylim([-0.2, 0.3])
# plt.title("$\epsilon=0.0$")
# plt.savefig("./figs/case2.png")
plt.show()

print("End main.")