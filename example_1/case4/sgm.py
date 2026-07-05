import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import trange
import tensorflow as tf


import models


a1 = -0.75
b1 = -0.25
a2 = 0.25
b2 = 0.75
# case 1: eps = 0.01, load nn1
# case 5: eps = 0.05, load nn5
eps = 0.05


scenario = 4


if scenario == 1:
    # scenario 1: T=1, yt = -0.3
    T = 1
    yt = -0.3
    N = 1000
elif scenario == 2:
    # scenario 2: T=0.3, yt = -0.2
    T = 0.3
    yt = -0.2
    N = 300
elif scenario == 3:
    # scenario 3: T=0.5, yt = -0.1
    T = 0.5
    yt = -0.1
    N = 500
elif scenario == 4:
    # scenario 3: T=0.8, yt = 0
    T = 0.8
    yt = 0.0
    N = 800
elif scenario == 5:
    # scenario 5: T=0.7, yt = 0.2
    T = 0.7
    yt = 0.2
    N = 700


def fn(x):
    A = 0.5 / np.abs(b1 - a1) * (x<=b1) * (x>=a1) + 0.5 / np.abs(b2 - a2) * (x<=b2) * (x>=a2)
    B = 1 / np.sqrt(2*np.pi*eps*T) * np.exp(-1/2/eps/T*(yt-x)**2)
    C1 = 1 / np.abs(b1 - a1) * (norm.cdf((yt - a1) / np.sqrt(eps*T)) - norm.cdf((yt - b1) / np.sqrt(eps*T)))
    C2 = 1 / np.abs(b2 - a2) * (norm.cdf((yt - a2) / np.sqrt(eps*T)) - norm.cdf((yt - b2) / np.sqrt(eps*T)))
    
    return A * B / (0.5 * C1 + 0.5 * C2)


model = models.NN(units=50, name="nn5_1000000", eps=eps, activation=tf.tanh)
model.restore()
forward = tf.function(model.call)


dt = 0.001
t = 0
zt = yt * np.ones(shape=[int(1e6)])


t = 0
for i in trange(N):
    update = eps * forward(
            (T - t) * tf.ones(shape=[zt.shape[0], 1]),
            tf.constant(zt.reshape([-1, 1]), tf.float32),
        ).numpy().reshape([-1])
    zt = zt + update * dt + np.sqrt(eps*dt) * np.random.normal(size=zt.shape)
    t = t + dt


x = np.linspace(-1.5, 1.5, 1001)

plt.plot(x, fn(x), "k--", label="exact density")
plt.hist(zt.flatten(), bins=100, density=True, color="c", label="posterior samples")
plt.xlim([-1.5, 1.5])
plt.legend()
plt.show()


sio.savemat(
    "./outputs/sgm_{}.mat".format(str(scenario)),
    {
        "zt": zt,
        "x": x,
        "y": fn(x),
    }
)