import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import scipy.io as sio


fig, axes = plt.subplots(1, 5, figsize=[15, 3], dpi=200)

data1 = sio.loadmat("./outputs/sgm1.mat")
x = data1["x"].flatten()
y = data1["y"].flatten()
samples = data1["samples"].flatten()

axes[0].plot(x, y, "k--", label="The exact posterior", linewidth=2)
axes[0].hist(samples, density=True, color="c", bins=100)
axes[0].set_title("$P(Y_{0.01}|Y_{0.8}=-4)$")


data2 = sio.loadmat("./outputs/sgm2.mat")
y = data2["y"].flatten()
samples = data2["samples"].flatten()

axes[1].plot(x, y, "k--", label="The exact posterior", linewidth=2)
axes[1].hist(samples, density=True, color="c", bins=100)
axes[1].set_title("$P(Y_{0.02}|Y_{0.5}=-2)$")


data3 = sio.loadmat("./outputs/sgm3.mat")
y = data3["y"].flatten()
samples = data3["samples"].flatten()

axes[2].plot(x, y, "k--", label="The exact posterior", linewidth=2)
axes[2].hist(samples, density=True, color="c", bins=100)
axes[2].set_title("$P(Y_{0.05}|Y_{0.6}=0.5)$")


data4 = sio.loadmat("./outputs/sgm4.mat")
y = data4["y"].flatten()
samples = data4["samples"].flatten()

axes[3].plot(x, y, "k--", label="The exact posterior", linewidth=2)
axes[3].hist(samples, density=True, color="c", bins=100)
axes[3].set_title("$P(Y_{0.45}|Y_{0.95}=1)$")


data5 = sio.loadmat("./outputs/sgm5.mat")
y = data5["y"].flatten()
samples = data5["samples"].flatten()

axes[4].plot(x, y, "k--", label="The exact posterior", linewidth=2)
axes[4].hist(samples, density=True, color="c", bins=100)
axes[4].set_title("$P(Y_{0.03}|Y_{0.4}=3)$")


for i in range(5):
    axes[i].set_ylim([-0.05, 0.55])
    axes[i].set_xlim([-4, 5])

fig.savefig("./outputs/sgm_2.png")
plt.show()
