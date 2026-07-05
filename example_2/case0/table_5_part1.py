import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import scipy.io as sio
import tensorflow as tf
from scipy.stats import wasserstein_distance


import models


################################################################
######################### Parameters ###########################
################################################################
eps = 1.5
alpha = 3
sigma = 1
mu = 0
T = 1
M = int(1e6)


def sampler(T, M):
    # the exact sampler of Y_T
    m = np.exp(-alpha * T) * mu
    sd = np.sqrt(np.exp(-2*alpha*T)*sigma**2 + eps*(1-np.exp(-2*alpha*T))/alpha/2)
    return m + sd * np.random.normal(size=M)


################################################################
######################## Exact sampler #########################
################################################################
def exact(yt, t0, T, M):
    v = eps * (sigma ** 2 * np.exp(-2*alpha*t0) + eps * (1 - np.exp(-2*alpha*t0)) / 2 / alpha) * \
        (1 - np.exp(-2*alpha*(T-t0))) / \
        (eps * (1 - np.exp(-2*alpha*T)) + 2 * alpha * sigma**2 * np.exp(-2*alpha*T))
    m = (
        eps * (np.exp(-alpha*t0) - np.exp(alpha*(t0-2*T))) * mu + \
        2 * alpha * sigma**2 * np.exp(-alpha * (T+t0)) * yt + \
        eps * (np.exp(-alpha*(T-t0)) - np.exp(-alpha*(T+t0))) * yt
    ) / (eps * (1 - np.exp(-2*alpha*T)) + 2 * alpha * sigma**2 * np.exp(-2*alpha*T))
    
    samples = m + np.sqrt(v) * np.random.normal(size=[M])
    
    return samples


################################################################
####################### Analytic sampler #######################
################################################################
def analytic(yt, M):
    cov = sigma ** 2
    P_fn = lambda t: np.exp(-2*alpha*t) * cov / eps + (1 - np.exp(-2*alpha*t)) / 2 / alpha
    q_fn = lambda t: np.exp(-alpha*t) * mu
    
    dt = 0.01
    N = 100
    t = 0
    zt = yt * np.ones(shape=[M])
    t = 0
    for i in range(N):
        update = - 1/P_fn(T - t) * (zt - q_fn(T-t)) + alpha * zt
        zt = zt + update * dt + np.sqrt(eps*dt) * np.random.normal(size=zt.shape)
        t = t + dt
    return zt


################################################################
########## Sampling using the analytical HJ sampler ############
################################################################
"""
y_test = sampler(T=T, M=1000)
sio.savemat("./outputs/data_test.mat", {"y_test": y_test})
print("saved!")
"""
y_test = sio.loadmat("./outputs/data_test.mat")["y_test"]
y_test = y_test.flatten()
print("loaded!")
errs = []


for i in range(1000):
    # samples from the analytic HJ-sampler
    u_samples = analytic(y_test[i], M=M)
    # samples from the exact sampler
    v_samples = exact(y_test[i], 0, T, M=M)

    # zts += [zt]
    # yts += [y_test[i]]
    errs += [wasserstein_distance(u_samples, v_samples)]
    print(np.mean(errs), flush=True)

print("Analytic HJ sampler: ")
print("Mean:", np.mean(errs))
print("SD:", np.std(errs), flush=True)
