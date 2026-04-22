
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ----------------------------
# Model parameters with fixed control
# ----------------------------
beta = 3/14
gamma = 1/14
tau = 60
R0 = beta / gamma
ti_grid_full = np.linspace(1, 110, 55)
sigma_grid = np.linspace(0, 1, 101)

# ----------------------------
# SIR model
# ----------------------------
def sir_model(y, t, beta, gamma, ti, sigma):
    if ti <= t <= ti + tau:
        b = beta * sigma
    else:
        b = beta
    S, I, R = y
    return [-b*S*I, b*S*I - gamma*I, gamma*I]

# Final size equation
def final_size_eq(x, S_end, I_end, R0):
    return (1 - x) + (1/R0)*np.log(1/(1-x)) - (S_end + I_end + (1/R0)*np.log(1/S_end))

# ----------------------------
# Compute t_peak (no control)
# ----------------------------
t = np.arange(0, 401, 1)
y0 = [0.999999, 1e-6, 0]

sol_nc = odeint(sir_model, y0, t, args=(beta, gamma, -1, 1))  
# ti=-1 ensures no intervention

I_nc = sol_nc[:,1]
t_peak = t[np.argmax(I_nc)]

ti_grid = ti_grid_full[ti_grid_full <= t_peak]

# ----------------------------
# Simulation
# ----------------------------
IPP = np.zeros((len(ti_grid), len(sigma_grid)))
EFS = np.zeros((len(ti_grid), len(sigma_grid)))

for i, ti in enumerate(ti_grid):
    for j, sigma in enumerate(sigma_grid):
        
        sol = odeint(sir_model, y0, t, args=(beta, gamma, ti, sigma))
        
        # Peak
        IPP[i, j] = np.max(sol[:,1])
        
        # Final size
        idx = int(round(ti + tau))
        S_end = sol[idx, 0]
        I_end = sol[idx, 1]
        
        EFS[i, j] = brentq(
            lambda x: final_size_eq(x, S_end, I_end, R0),
            0.66, 0.99999999
        )

# ----------------------------
# Optimization
# ----------------------------
IPP_opt = np.min(IPP, axis=1)
EFS_opt = np.min(EFS, axis=1)

sigma_opt_IPP = sigma_grid[np.argmin(IPP, axis=1)]
sigma_opt_EFS = sigma_grid[np.argmin(EFS, axis=1)]


mask = ti_grid >= 40

tp_plot = ti_grid[mask]
IPP_plot = IPP_opt[mask]
EFS_plot = EFS_opt[mask]

sigma_IPP_plot = sigma_opt_IPP[mask]
sigma_EFS_plot = sigma_opt_EFS[mask]

# ----------------------------
# Plot
# ----------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# ---- Left: V-shape ----
ax1 = axes[0]
ax2 = ax1.twinx()

ax1.plot(tp_plot, IPP_plot, linewidth=2, label='IPP')
ax2.plot(tp_plot, EFS_plot, 'r--', linewidth=2, label='EFS')

ax1.set_xlabel(r'$t_p$')
ax1.set_ylabel(r'$I_{\max}$')
ax2.set_ylabel(r'$R_\infty$')

ax1.set_title('Optimal outcomes vs triggering time')
ax1.grid(True)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

# ---- Right: Controls ----
axes[1].plot(tp_plot, sigma_IPP_plot, linewidth=2, label='IPP-opt')
axes[1].plot(tp_plot, sigma_EFS_plot, 'r--', linewidth=2, label='EFS-opt')

axes[1].set_xlabel(r'$t_p$')
axes[1].set_ylabel(r'$\sigma^{opt}$')
axes[1].set_title('Optimal control intensity')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
