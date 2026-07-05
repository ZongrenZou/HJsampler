import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import scipy.io as sio
import tensorflow as tf


import models


######################## Load data ########################
data = sio.loadmat("./data/train.mat")

t = data["t"]
ys = data["ys"]
eps = data["eps"]
T = 1
y1s, y2s = np.split(ys, 2, axis=0)
batch_size = 1000
N = 100000

y1_train = y1s[:N, 1:]
y2_train = y2s[:N, 1:]
t_train = np.tile(t[:, 1:], [N, 1])


######################## Build model and train ########################
model = models.NN(units=50, name="nn_{}".format(str(N)), eps=eps, activation=tf.tanh)
# loss = model.train(y_train, t_train, niter=2001)
loss = model.train_batch(y1_train, y2_train, t_train, batch_size=batch_size, nepoch=3000)
# model.opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
# loss = model.train_batch(y_train, t_train, batch_size=1000, nepoch=5000)
