import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import scipy.io as sio
from scipy.stats import wasserstein_distance


data1 = sio.loadmat("./outputs/analytic1.mat")
x = data1["x"].flatten()
y = data1["y"].flatten()
samples = data1["samples"].flatten()
v_samples = data1["v_samples"].flatten()
print(samples.shape, v_samples.shape)
print("Case 1:")
print(wasserstein_distance(samples, v_samples))


data1 = sio.loadmat("./outputs/analytic2.mat")
x = data1["x"].flatten()
y = data1["y"].flatten()
samples = data1["samples"].flatten()
v_samples = data1["v_samples"].flatten()
print("Case 2:")
print(wasserstein_distance(samples, v_samples))


data1 = sio.loadmat("./outputs/analytic3.mat")
x = data1["x"].flatten()
y = data1["y"].flatten()
samples = data1["samples"].flatten()
v_samples = data1["v_samples"].flatten()
print("Case 3:")
print(wasserstein_distance(samples, v_samples))


data1 = sio.loadmat("./outputs/analytic4.mat")
x = data1["x"].flatten()
y = data1["y"].flatten()
samples = data1["samples"].flatten()
v_samples = data1["v_samples"].flatten()
print("Case 4:")
print(wasserstein_distance(samples, v_samples))


data1 = sio.loadmat("./outputs/analytic5.mat")
x = data1["x"].flatten()
y = data1["y"].flatten()
samples = data1["samples"].flatten()
v_samples = data1["v_samples"].flatten()
print("Case 5:")
print(wasserstein_distance(samples, v_samples))
