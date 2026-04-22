import numpy as np
from scipy.optimize import brentq
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ===============================
# Parameters
# ===============================
beta = 3 / 14
gamma = 1 / 14
R0 = beta / gamma
tau = 60

ti_grid = np.linspace(1, 80, 80)
f_grid = np.linspace(0, 1, 85)
sigma_grid = np.linspace(0, 0.9, 10)

y0 = [0.999999, 0.000001, 0]
t = np.arange(0, 400, 1)
options = {'rtol': 1e-12, 'atol': 1e-12}

# ===============================
# ODE SYSTEMS
# ===============================
def sir_fixed(y, t, beta, gamma, ti, sigma):
    if ti <= t <= ti + tau:
        b = beta * sigma
    else:
        b = beta

    S, I, R = y
    return [-b * S * I, b * S * I - gamma * I, gamma * I]


def sir_ms(y, t, beta, gamma, ti, f, sigma):
    if ti <= t <= ti + f * tau:
        b = beta * sigma
    elif ti + f * tau < t <= ti + tau:
        b = 0
    else:
        b = beta

    S, I, R = y
    return [-b * S * I, b * S * I - gamma * I, gamma * I]

# Final size equation
def final_size_eq(x, S_end, I_end, R0):
    return (1 - x) + (1/R0)*np.log(1/(1-x)) - (S_end + I_end + (1/R0)*np.log(1/S_end))

# ===============================
# STORAGE
# ===============================
IPP_ms = np.zeros((len(ti_grid), len(f_grid), len(sigma_grid)))
EFS_ms = np.zeros_like(IPP_ms)

IPP_fixed = np.zeros((len(ti_grid), len(sigma_grid)))
EFS_fixed = np.zeros_like(IPP_fixed)

# ===============================
# COMPUTE MS STRATEGY
# ===============================
for k, ti in enumerate(ti_grid):
    for i, f in enumerate(f_grid):
        for j, sigma in enumerate(sigma_grid):
            sol = odeint(sir_ms, y0, t, args=(beta, gamma, ti, f, sigma), **options)

            IPP_ms[k, i, j] = np.max(sol[:, 1])

            S_end = sol[int(round(ti)+tau), 0]
            I_end = sol[int(round(ti)+tau), 1]
            
            EFS_ms[k, i, j] = brentq(
                lambda x: final_size_eq(x, S_end, I_end, R0),
                0.66, 0.99999999
            )

# ===============================
# COMPUTE FIXED STRATEGY
# ===============================
for k, ti in enumerate(ti_grid):
    for j, sigma in enumerate(sigma_grid):
        sol = odeint(sir_fixed, y0, t, args=(beta, gamma, ti, sigma), **options)

        IPP_fixed[k, j] = np.max(sol[:, 1])

        S_end = sol[int(round(ti)+tau), 0]
        I_end = sol[int(round(ti)+tau), 1]
        
        EFS_fixed[k, j] = brentq(
            lambda x: final_size_eq(x, S_end, I_end, R0),
            0.66, 0.99999999
        )

# ===============================
# OPTIMAL FIXED CONTROL
# ===============================
opt_IPP_fixed = np.min(IPP_fixed, axis=1)
opt_EFS_fixed = np.min(EFS_fixed, axis=1)

# ===============================
# FIND MS STRATEGIES BEATING FIXED
# ===============================
Tuple_of_control = [[] for _ in ti_grid]
better_IPP = [[] for _ in ti_grid]
better_EFS = [[] for _ in ti_grid]

for k, ti in enumerate(ti_grid):
    for i, f in enumerate(f_grid):
        for j, sigma in enumerate(sigma_grid):
            if (IPP_ms[k, i, j] <= opt_IPP_fixed[k] and
                EFS_ms[k, i, j] <= opt_EFS_fixed[k]):

                better_IPP[k].append(IPP_ms[k, i, j])
                better_EFS[k].append(EFS_ms[k, i, j])
                Tuple_of_control[k].append((tau * f, sigma))

# ===============================
# PLOTTING
# ===============================
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10
})

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
(ax1, ax2), (ax3, ax4) = axes

# ---- (A) IPP ----
ax1.plot(ti_grid, opt_IPP_fixed, color='red', lw=2, label='Optimal IPP (fixed control)')

for k in range(len(ti_grid)):
    ax1.plot([ti_grid[k]] * len(better_IPP[k]),
             better_IPP[k], '.', color='tab:blue', markersize=2)
ax1.plot([], [], '.', color='tab:blue', label='IPP (MS strategies)')
ax1.set_xlabel(r'$t_p$')
ax1.set_ylabel(r'$I_{\max}$')
ax1.set_title('(A)')
ax1.set_xlim(5, 85)
ax1.grid(True)
ax1.legend()

# ---- (B) EFS ----
ax2.plot(ti_grid, opt_EFS_fixed, color='red', lw=2, label='Optimal EFS (fixed control)')

for k in range(len(ti_grid)):
    ax2.plot([ti_grid[k]] * len(better_EFS[k]),
             better_EFS[k], '.', color='tab:blue', markersize=2)
ax2.plot([], [], '.', color='tab:blue', label='EFS (MS strategies)')
ax2.set_xlabel(r'$t_p$')
ax2.set_ylabel(r'$R_{\infty}$')
ax2.set_title('(B)')
ax2.set_xlim(5, 85)
ax2.grid(True)
ax2.legend()

# ===============================
# Beneficial MS strategies
# ===============================
x_points = ti_grid
W = Tuple_of_control
all_z_values = [z for dataset in W for _, z in dataset]
unique_z = sorted(set(all_z_values))
colors = ["#000000", "#1f77b4", "#d62728", "#2ca02c", "#ffdf00"]
markers = ['o', '<', '>', '^','v', 'X','*', 'P', 'D', 's']  
z_to_color_marker = {z: (colors[i % len(colors)], markers[i % len(markers)]) for i, z in enumerate(unique_z)}

# ---- (C) f*tau ----
for x, dataset in zip(x_points, W):
    for y, z in dataset:
        color, marker = z_to_color_marker[z]
        label = f'$\sigma={z:.1f}$'

        if label not in ax3.get_legend_handles_labels()[1]:
            ax3.scatter(x, y, color=color, marker=marker, label=label)
        else:
            ax3.scatter(x, y, color=color, marker=marker)

ax3.set_xlabel(r'$t_p$')
ax3.set_ylabel(r'$f\tau$')
ax3.set_title('(C)')
ax3.set_xlim(5, 85)
ax3.grid(True)
ax3.legend()

# ---- (D) sigma ----
for x, dataset in zip(x_points, W):
    for y, z in dataset:
        color, marker = z_to_color_marker[z]
        ax4.scatter(x, z, color=color, marker=marker)

ax4.set_xlabel(r'$t_p$')
ax4.set_ylabel(r'$\sigma$')
ax4.set_title('(D)')
ax4.set_xlim(5, 85)
ax4.grid(True)

plt.tight_layout()
plt.show()
