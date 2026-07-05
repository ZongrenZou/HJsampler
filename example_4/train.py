import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import scipy.io as sio
import tensorflow as tf
import time


import models


######################## Load data ########################
data = sio.loadmat("./data/train.mat")

y_train = data["ys"]
t = data["t"].reshape([-1, 1])
eps = data["eps"]
C = data["C"]
T = 0.1
print(y_train.shape)

N = 100000
batch_size = 1000

y_train = y_train[:, 1:, :]
t_train = np.tile(t[1:].T[..., None], [y_train.shape[0], 1, 1])


print("##################")
print(y_train.shape, t_train.shape)

######################## Build model and train ########################
model = models.NN(units=200, name="nn3", activation=tf.tanh)
model.loss_function = model._loss_function_3

t0 = time.time()
loss = model.train_batch(t_train, y_train, batch_size=batch_size, nepoch=3000)
t1 = time.time()

print(t0 - t1)
