import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import scipy.io as sio
import tensorflow as tf


import models


######################## Load data ########################
# train3.mat: eps = 0.001
# train2.mat: eps = 0.0005
# train.mat: eps = 0.0001
data = sio.loadmat("./data/train3.mat")

t = data["t"].T
ys = data["ys"]
eps = data["eps"]
T = 5
ys = np.transpose(ys, [2, 0, 1])
batch_size = 1000
N = 100000

y_train = ys[:N, 1:]
t_train = np.tile(t[:, 1:], [N, 1])


######################## Build model and train ########################
model = models.NN(units=50, name="nn3_{}".format(str(N)), eps=eps, activation=tf.tanh)
loss = model.train_batch(y_train, t_train, batch_size=batch_size, nepoch=3000)
