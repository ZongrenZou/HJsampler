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

ys = data["ys"]
y1, y2, y3 = np.split(ys, 3, axis=0)
t = data["t"].reshape([-1, 1])
eps = data["eps"]
T = data["T"]

N = 500000
batch_size = 1000
y1_train = y1[:N, ::1]
y2_train = y2[:N, ::1]
y3_train = y3[:N, ::1]
t_train = t[::1]
print(y1_train.shape, y2_train.shape, y3_train.shape, t_train.shape)

y1_train = y1_train[:, 1:]
y2_train = y2_train[:, 1:]
y3_train = y3_train[:, 1:]
t_train = np.tile(t_train[1:].T, [y1_train.shape[0], 1])


######################## Build model and train ########################
model = models.NN(units=50, name="nn_{}".format(str(N)), eps=eps, activation=tf.tanh)
loss = model.train_batch(
    t_train,
    y1_train, 
    y2_train, 
    y3_train,  
    batch_size=batch_size, 
    nepoch=5000,
)
