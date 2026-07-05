import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import scipy.io as sio
import tensorflow as tf
from tqdm import trange
from scipy.stats import wasserstein_distance


import models as models


sigma1 = 0.5
sigma2 = 0.8
sigma3 = 0.6
mu1 = 0
mu2 = 2
mu3 = -2

eps = 5
T = 0.5
t0 = 0.02


def prior_fn(x):
    T1 = (x - mu1) ** 2 / sigma1 ** 2
    f1 = 1 / np.sqrt(2*np.pi*sigma1**2) * np.exp(-T1/2)
    T2 = (x - mu2) ** 2 / sigma2 ** 2
    f2 = 1 / np.sqrt(2*np.pi*sigma2**2) * np.exp(-T2/2)
    T3 = (x - mu3) ** 2 / sigma3 ** 2
    f3 = 1 / np.sqrt(2*np.pi*sigma3**2) * np.exp(-T3/2)
    return f1 / 3 + f2 / 3 + f3 / 3


def posterior_fn(x, yt, t):
    T1 = (x - mu1) ** 2 / (sigma1 ** 2 + eps * t * 1)
    # f1 = 1 / np.sqrt(2*np.pi*sigma1**2) * np.exp(-T1/2)
    f1 = 1 / C(t, sigma1) * np.exp(-T1/2)
    T2 = (x - mu2) ** 2 / (sigma2 ** 2 + eps * t * 1)
    # f2 = 1 / np.sqrt(2*np.pi*sigma2**2) * np.exp(-T2/2)
    f2 = 1 / C(t, sigma2) * np.exp(-T2/2)
    T3 = (x - mu3) ** 2 / (sigma3 ** 2 + eps * t * 1)
    # f3 = 1 / np.sqrt(2*np.pi*sigma3**2) * np.exp(-T3/2)
    f3 = 1 / C(t, sigma3) * np.exp(-T3/2)
    A = f1 / 3 + f2 / 3 + f3 / 3
    
    B = 1 / np.sqrt(2*np.pi*eps*(T-t)) * np.exp(-(x-yt) ** 2 / 2 / eps / (T-t))
    
    T1 = (yt - mu1) ** 2 / (sigma1 ** 2 + eps * T)
    T2 = (yt - mu2) ** 2 / (sigma2 ** 2 + eps * T)
    T3 = (yt - mu3) ** 2 / (sigma3 ** 2 + eps * T)
    
    C1 = 1/C(T, sigma1) * np.exp(-T1/2)
    C2 = 1/C(T, sigma2) * np.exp(-T2/2)
    C3 = 1/C(T, sigma3) * np.exp(-T3/2)
    
    return A * B / (C1/3 + C2/3 + C3/3)


def C(t, sigma):
    return np.sqrt(2 * np.pi * (sigma ** 2 + eps * t))


################################################################
####################### SGM HJ sampler #########################
################################################################
x = np.linspace(-5, 5, 1001)
model = models.NN(units=50, name="nn_1000000", eps=5, activation=tf.tanh)
model.restore()

forward_fn = tf.function(model.call)


dt = 0.001
M = int(1e6)
yt = -2

zts = yt * np.ones([M, 1], dtype=np.float32)
t = 0
tt = tf.ones([M, 1])
for i in trange(int((T-t0)/dt)):
    update = forward_fn(
        (T - t) * tt,
        tf.constant(zts, tf.float32),            
    ).numpy()
    zts = zts + eps * update * dt + \
        np.sqrt(eps) * np.random.normal(size=zts.shape).astype(np.float32) * np.sqrt(dt)
    t = t + dt
u_samples = zts.flatten()


################################################################
####################### Exact sampler ##########################
################################################################
s1 = np.sqrt(1/(1/(sigma1**2 + eps * t0) + 1/eps/(T-t0)))
m1 = s1 * s1 * (mu1 / (sigma1**2 + eps * t0) + yt / eps / (T-t0))
s2 = np.sqrt(1/(1/(sigma2**2 + eps * t0) + 1/eps/(T-t0)))
m2 = s2 * s2 * (mu2 / (sigma2**2 + eps * t0) + yt / eps / (T-t0))
s3 = np.sqrt(1/(1/(sigma3**2 + eps * t0) + 1/eps/(T-t0)))
m3 = s3 * s3 * (mu3 / (sigma3**2 + eps * t0) + yt / eps / (T-t0))

z1 = np.sqrt(2*np.pi*s1*s1) * (
    1/3/C(T, sigma1) * np.exp(-0.5*(yt-mu1)**2 / (sigma1**2 + eps * T)) + \
    1/3/C(T, sigma2) * np.exp(-0.5*(yt-mu2)**2 / (sigma2**2 + eps * T)) + \
    1/3/C(T, sigma3) * np.exp(-0.5*(yt-mu3)**2 / (sigma3**2 + eps * T))
) / (1 / C(T, sigma1) * np.exp(-0.5*(yt-mu1)**2 / (sigma1**2 + eps * T)))
z2 = np.sqrt(2*np.pi*s2*s2) * (
    1/3/C(T, sigma1) * np.exp(-0.5*(yt-mu1)**2 / (sigma1**2 + eps * T)) + \
    1/3/C(T, sigma2) * np.exp(-0.5*(yt-mu2)**2 / (sigma2**2 + eps * T)) + \
    1/3/C(T, sigma3) * np.exp(-0.5*(yt-mu3)**2 / (sigma3**2 + eps * T))
) / (1 / C(T, sigma2) * np.exp(-0.5*(yt-mu2)**2 / (sigma2**2 + eps * T)))
z3 = np.sqrt(2*np.pi*s3*s3) * (
    1/3/C(T, sigma1) * np.exp(-0.5*(yt-mu1)**2 / (sigma1**2 + eps * T)) + \
    1/3/C(T, sigma2) * np.exp(-0.5*(yt-mu2)**2 / (sigma2**2 + eps * T)) + \
    1/3/C(T, sigma3) * np.exp(-0.5*(yt-mu3)**2 / (sigma3**2 + eps * T))
) / (1 / C(T, sigma3) * np.exp(-0.5*(yt-mu3)**2 / (sigma3**2 + eps * T)))

z1 = np.sqrt(2*np.pi * s1 * s1) / z1 / 3
z2 = np.sqrt(2*np.pi * s2 * s2) / z2 / 3
z3 = np.sqrt(2*np.pi * s3 * s3) / z3 / 3

a = np.random.choice(3, p=[z1, z2, z3], size=M)
a1 = a == 0
a2 = a == 1
a3 = a == 2

samples1 = m1 + s1 * np.random.normal(size=[M])
samples2 = m2 + s2 * np.random.normal(size=[M])
samples3 = m3 + s3 * np.random.normal(size=[M])

v_samples = a1 * samples1 + a2 * samples2 + a3 * samples3


sio.savemat(
    "./outputs/sgm2.mat", 
    {
        "x": x,
        "y": posterior_fn(x, yt, t0),
        "samples": u_samples,
        "v_samples": v_samples,
    },
)


print("End")
