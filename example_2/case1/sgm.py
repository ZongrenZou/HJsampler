import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import scipy.io as sio
import tensorflow as tf
from tqdm import trange


import models


B = np.array(
    [
        [0, -1],
        [1, 1],
    ]
)
alpha = 1
eps = 1
T = 1

## case 1: 
yt = np.array([0.5, -0.5]).reshape([1, 2])
# yt = np.array([0., -0.5]).reshape([1, 2])
## case 2: 
# yt = np.array([-1, 0.5]).reshape([1, 2])
## case 3: 
# yt = np.array([-0.7, -0.9]).reshape([1, 2])
## case 4: 
# yt = np.array([0.5, 1]).reshape([1, 2])

## sampling
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

# sio.savemat(
#     "./outputs/sgm.mat",
#     {"zts": zts[:, ::1000, :]},
# )


## plot
fig = plt.figure(dpi=100)

ax = fig.add_subplot(1, 1, 1)
h = ax.hist2d(
    zts[-1, :, 0], zts[-1, :, 1], bins=(100, 100), density=True, 
    cmap=colormaps["jet"],
    # vmin=0, vmax=0.8
)
ax.plot(yt[0, 0], yt[0, 1], "wo", markersize=20)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_aspect("equal")
fig.colorbar(h[3], ax=ax)
ax.set_title("Posterior samples: SGM HJ sampler")

fig.savefig("./figs/sgm.png")
fig.show()