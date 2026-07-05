import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import scipy.io as sio
import tensorflow as tf
from tqdm import trange


import models


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


## sampling
N = 100000
model = models.NN(units=50, name="nn_{}".format(str(N)), eps=np.array(eps), activation=tf.tanh)
model.restore()

yt = np.array([-0.9, 0.9]).reshape([1, -1])

dt = 0.001
t = 0
M = 100000
zts = [np.tile(yt, [M, 1])]
for i in trange(int(T/dt)):
    update = model.call(
        tf.constant(zts[-1][:, 0:1], tf.float32),
        tf.constant(zts[-1][:, 1:2], tf.float32),
        (T - t) * tf.ones(shape=[M, 1]),
    ).numpy()
    zt = zts[-1] + eps * update * dt + \
        np.sqrt(eps) * np.random.normal(size=zts[-1].shape) * np.sqrt(dt)
    zts += [zt]
    t = t + dt
zts = np.stack(zts, axis=0)

sio.savemat(
    "./outputs/sgm.mat",
    {"zts": zts[:, ::1000, :]},
)


## plot posterior histogram
plt.figure()
plt.hist2d(zts[-1, :, 0], zts[-1, :, 1], bins=(50, 50), density=True, cmap=colormaps["jet"]) #, vmin=0, vmax=1)
plt.clim(0, 1)
plt.colorbar()
plt.xlim([-1., 1.])
plt.ylim([-1., 1.])
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.savefig("./figs/sgm.png")
plt.show()
