import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import scipy.io as sio
from scipy.linalg import expm
from tqdm import trange
from scipy.integrate import odeint


def check_riccati(cov, mu, yt, eps):
    B = np.array(
        [
            [0, -1],
            [1, 1],
        ]
    )
    D = np.array(
        [
            [0, 0],
            [0, 1],
        ]
    )
    # first solve for P, q
    dt = 0.00005
    M = 100

    Ps = [1/eps * cov]
    for i in range(int(np.ceil(T/dt))):
        P_update = D - Ps[-1] @ B.T - B @ Ps[-1]
        Ps += [Ps[-1] + dt * P_update]
    Ps = Ps[::M]
    Ps = Ps[1:]
    Ps.reverse()

    q_fn = lambda t: (expm(-B*t) @ mu.T).T
    

    def s_fn(Y, t, i):
        return (-np.linalg.inv(Ps[i]) @ (Y - q_fn(t)).T).T
    

    dt = 0.005
    zts = [yt * np.ones(shape=[100000, 2])]
    t = 0
    for i in trange(int(T/dt)):
        update = s_fn(zts[-1], T-t, i) 
        b = (-B @ zts[-1].T).T
        zts += [
            zts[-1] + (update - b) * dt + \
            np.sqrt(eps*dt) * np.random.normal(size=zts[-1].shape) * np.array([0, 1]).reshape([1, 2])
        ]
        t = t + dt
    zts = np.stack(zts, axis=0)
    return zts


n = 2
cov = np.array(
    [
        [1, 0],
        [0, 1],
    ]
)
mu = np.array([0, 0]).reshape([1, 2])

alpha = 1
eps = 0.0001
T = 5

B = np.array(
    [
        [0, -1],
        [1, 1],
    ]
)


def rhs(y, t):
    y1, y2 = y
    dydt = [y2, -1.1*y1 - 1*y2]
    return dydt


## case 1: 
y0 = np.array([0, 0.5])
t = np.linspace(0, 5, 1001)
sols = odeint(rhs, y0, t)
yt = sols[-1, :].reshape([1, 2])


def rhs_rev(y, t):
    y1, y2 = y
    dydt = [-y2, +1*y1 + 1*y2]
    return dydt


z0 = sols[-1, :]
t = np.linspace(0, 5, 1001)
z_sols = odeint(rhs_rev, z0, t)


zts = check_riccati(
    cov=cov,
    mu=mu,
    yt=yt, 
    eps=eps,
)
mu = np.mean(zts, axis=1)
sd = np.std(zts, axis=1)

sio.savemat(
    "./outputs/riccati.mat",
    {
        "samples": zts[-1, ...]
    }
)

# sio.savemat(
#     "./outputs/riccati.mat",
#     {
#         "t": t,
#         "mu": mu,
#         "sd": sd,
#     }
# )

######################### Make plot #########################
fig = plt.figure(dpi=100)

t = np.linspace(0, 5, 1001)
plt.plot(t, sols[:, 0], "k-", label="Reference of $u_1$")
plt.plot(t, sols[:, 1], "b-", label="Reference of $u_2$")
# plt.plot(t, np.flip(z_sols[:, 0]), "r--")
# plt.plot(t, np.flip(z_sols[:, 1]), "r--")
plt.plot(t, np.flip(mu[:, 0]), "r--", label="Mean of $u_1$")
plt.plot(t, np.flip(mu[:, 1]), "g--", label="Mean of $u_2$")
plt.fill_between(t.flatten(), np.flip(mu[:, 0] + 2*sd[:, 0]), np.flip(mu[:, 0] - 2*sd[:, 0]), alpha=0.3)
plt.fill_between(t.flatten(), np.flip(mu[:, 1] + 2*sd[:, 1]), np.flip(mu[:, 1] - 2*sd[:, 1]), alpha=0.3)
plt.ylim([-0.5, 1])
plt.xlabel("T")
plt.legend()
plt.title("Riccati HJ sampler: $\epsilon=0.0001$")
plt.savefig("./figs/riccati_case1.png")
plt.show()


print("End main.")