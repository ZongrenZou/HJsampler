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

t = data["t"].reshape([-1, 1])
y = data["y"]
eps = data["eps"]
T = 1

N = 1000000
batch_size = 1000
y_train = y[:N, ::1]
t_train = t[::1]

## for sliced score matching loss
print(y_train.shape)
print(eps)
y_train = y_train[:, 1:]
t_train = np.tile(t_train[1:].T, [y_train.shape[0], 1])
print(y_train.shape, t_train.shape)


######################## Build model and train ########################

model = models.NN(
    units=100, 
    name="nn1",
    eps=eps, 
    activation=tf.tanh,
    sigma=None,
)
loss = model.train_batch(
    t_train, 
    y_train,  
    batch_size=batch_size, 
    nepoch=5000,
)
