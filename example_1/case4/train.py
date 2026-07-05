import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import scipy.io as sio
import tensorflow as tf


import models


######################## Load data ########################
data = sio.loadmat("./data/train1.mat")

# y_train = data["y_train"]
# y_test = data["y_test"]
y1_train = data["y0_train"]
y2_train = data["y1_train"]
t = data["t"].reshape([-1, 1])
eps = data["eps"]
T = data["T"]

N = 1000000
batch_size = 1000
y1_train = y1_train[:N, ::1]
y12_train = y2_train[:N, ::1]
t_train = t[::1]

y1_train = y1_train[:, 1:]
y2_train = y2_train[:, 1:]
t_train = np.tile(t_train[1:].T, [y1_train.shape[0], 1])

print(eps, T)
print(y1_train.shape)


######################## Build model and train ########################
model = models.NN(units=50, name="nn1_{}".format(str(N)), eps=eps, activation=tf.tanh)
loss = model.train_batch(t_train, y1_train, y2_train, batch_size=batch_size, nepoch=5000)
