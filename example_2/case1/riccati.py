import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import scipy.io as sio
from scipy.linalg import expm
from tqdm import trange


def check_riccati(cov1, cov2, mu1, mu2, yt, eps, s, t0):
    B = np.array(
        [
            [0, -1],
            [1, 1],
        ]
    )
    # first solve for P, q
    dt = 0.00001
    M = 100

    P1s = [1/eps * cov1]
    # q1s = [mu1]
    for i in range(int(np.ceil(T/dt))):
        P_update = np.eye(n) - P1s[-1] @ B.T - B @ P1s[-1]
        # q_update = (-B @ q1s[-1].T).T
        P1s += [P1s[-1] + dt * P_update]
        # q1s += [q1s[-1] + dt * q_update]
    P1s = P1s[::M]
    # q1s = q1s[::M]
    P1s = P1s[1:]
    # q1s = q1s[1:]
    P1s.reverse()
    # q1s.reverse()

    P2s = [1/eps * cov2]
    # q2s = [mu2]
    for i in range(int(np.ceil(T/dt))):
        P_update = np.eye(n) - P2s[-1] @ B.T - B @ P2s[-1]
        # q_update = (-B @ q2s[-1].T).T
        P2s += [P2s[-1] + dt * P_update]
        # q2s += [q2s[-1] + dt * q_update]
    P2s = P2s[::M]
    # q2s = q2s[::M]
    P2s = P2s[1:]
    # q2s = q2s[1:]
    P2s.reverse()
    # q2s.reverse()

    r_fn = lambda P: eps / 2 * np.log((2*np.pi*eps)**n * np.linalg.det(P))
    q1_fn = lambda t: (expm(-B*t) @ mu1.T).T
    q2_fn = lambda t: (expm(-B*t) @ mu2.T).T
    

    def s_fn(Y, t, i):
        T1 = (Y - q1_fn(t)) @ np.linalg.inv(P1s[i]) # shape of [N, 2]
        T1 = np.einsum("Ni,Ni->N", T1, Y - q1_fn(t))
        T2 = (Y - q2_fn(t)) @ np.linalg.inv(P2s[i]) # shape of [N, 2]
        T2 = np.einsum("Ni,Ni->N", T2, Y - q2_fn(t))
        # T1 = (Y - q1s[i]) @ np.linalg.inv(P1s[i]) # shape of [N, 2]
        # T1 = np.einsum("Ni,Ni->N", T1, Y - q1s[i])
        # T2 = (Y - q2s[i]) @ np.linalg.inv(P2s[i]) # shape of [N, 2]
        # T2 = np.einsum("Ni,Ni->N", T2, Y - q2s[i])
        B1 = np.exp(-1/2/eps * T1 - r_fn(P1s[i]) / eps)
        B2 = np.exp(-1/2/eps * T2 - r_fn(P2s[i]) / eps)

        A = 0.5 * np.linalg.inv(P1s[i]) @ (Y - q1_fn(t)).T * B1 + \
            0.5 * np.linalg.inv(P2s[i]) @ (Y - q2_fn(t)).T * B2
        # A = 0.5 * np.linalg.inv(P1s[i]) @ (Y - q1s[i]).T * B1 + \
        #     0.5 * np.linalg.inv(P2s[i]) @ (Y - q2s[i]).T * B2
        B = 0.5 * B1 + 0.5 * B2
        return - A.T / B[:, None]
    

    dt = 0.001
    M = int(1e6)
    zt = yt * np.ones(shape=[M, 2])
    t = t0
    start = int(np.round(t0/dt, 0))
    end = int(np.round(s/dt, 0))
    for i in trange(start, end):
    # for i in trange(int(T/dt)):
        update = s_fn(zt, T-t, i) + (B @ zt.T).T
        zt = zt + update * dt + np.sqrt(eps*dt) * np.random.normal(size=zt.shape)
        t = t + dt
    
    return zt


n = 2
cov1 = np.array(
    [
        [0.5 ** 2, 0.1],
        [0.1, 0.4 ** 2],
    ]
)
cov2 = np.array(
    [
        [0.5 ** 2, -0.1],
        [-0.1, 0.4 ** 2],
    ]
)
mu1 = np.array([-0.7, 0]).reshape([1, 2])
mu2 = np.array([0.7, 0]).reshape([1, 2])

alpha = 1
eps = 1
T = 1

## case 1: 
# Ts = 1
# t0 = 0
# yt = np.array([0.5, -0.5]).reshape([1, 2])
## case 2: 
s = 0.9
t0 = 0.1
yt = np.array([-1, 0.5]).reshape([1, 2])
## case 3: 
# yt = np.array([-0.7, -0.9]).reshape([1, 2])
## case 4: 
# yt = np.array([0.5, 1]).reshape([1, 2])


zt = check_riccati(
    cov1=cov1,
    cov2=cov2,
    mu1=mu1,
    mu2=mu2,
    yt=yt, 
    eps=eps,
    s=s,
    t0=t0,
)

######################### Make plot #########################
fig = plt.figure(dpi=100)

ax = fig.add_subplot(1, 1, 1)
h = ax.hist2d(
    zt[:, 0], zt[:, 1], bins=(100, 100), density=True, 
    cmap=colormaps["jet"],
    vmin=0,
    vmax=0.8,
)
ax.plot(yt[0, 0], yt[0, 1], "wo", markersize=20)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_aspect("equal")
fig.colorbar(h[3], ax=ax)
ax.set_title("Posterior samples: Riccati sampler")

fig.savefig("./figs/riccati.png")
fig.show()

