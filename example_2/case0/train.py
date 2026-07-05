import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import scipy.io as sio
import tensorflow as tf


import models


######################## Load data ########################
data = sio.loadmat("./data/train.mat")

y_train = data["ys"]
t = data["t"].reshape([-1, 1])
T = 1
print(y_train.shape)

N = 10000
batch_size = 100
y_train = y_train[:N, ::1]
t_train = t[::1]

y_train = y_train[:, 1:]
t_train = np.tile(t_train[1:].T, [y_train.shape[0], 1])


print("##################")
print(y_train.shape)

######################## Build model and train ########################
model = models.NN(units=50, name="nn", activation=tf.tanh)
loss = model.train_batch(t_train, y_train, batch_size=batch_size, nepoch=3000)

