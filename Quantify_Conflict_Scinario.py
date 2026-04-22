import numpy as np
from scipy.optimize import brentq
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ----------------------------
# Parameters
# ----------------------------
beta = 3/14
gamma = 1/14
R0 = beta / gamma

ti_grid = np.linspace(1, 90, 90)
sigma_grid = np.linspace(0, 0.9, 101)
tau_grid = np.linspace(10, 120, 56)

# ----------------------------
# SIR model
# ----------------------------
def sir_model(y, t, beta, gamma, ti, sigma, tau):
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
# Simulation grid
# ----------------------------
t = np.arange(0, 400, 1)
y0 = [0.999999, 1e-6, 0]
options = {'rtol': 1e-12, 'atol': 1e-12}

IPP = np.zeros((len(tau_grid), len(ti_grid), len(sigma_grid)))
EFS = np.zeros((len(tau_grid), len(ti_grid), len(sigma_grid)))

for k, tau in enumerate(tau_grid):
    for i, ti in enumerate(ti_grid):
        for j, sigma in enumerate(sigma_grid):
            
            sol = odeint(sir_model, y0, t, args=(beta, gamma, ti, sigma, tau),**options)
            
            # Peak
            IPP[k, i, j] = np.max(sol[:,1])
            
            # Final size
            S_end = sol[int(round(ti)+tau), 0]
            I_end = sol[int(round(ti)+tau), 1]
            
            EFS[k, i, j] = brentq(
                lambda x: final_size_eq(x, S_end, I_end, R0),
                0.66, 0.99999999
            )

# ----------------------------
# Deviation calculation
# ----------------------------
Delta_EFS = np.zeros(len(tau_grid))
Delta_IPP = np.zeros(len(tau_grid))

for k in range(len(tau_grid)):
    
    # Optimal values
    PP = np.min(IPP[k])
    EE = np.min(EFS[k])
    
    # Corresponding cross values
    idx_IPP = np.unravel_index(np.argmin(IPP[k]), IPP[k].shape)
    idx_EFS = np.unravel_index(np.argmin(EFS[k]), EFS[k].shape)
    
    EP = EFS[k][idx_IPP]  # EFS under IPP-opt control
    PE = IPP[k][idx_EFS]  # IPP under EFS-opt control
    
    # Deviations (relative to optimal values)
    Delta_EFS[k] = (EP - EE) / EE * 100
    Delta_IPP[k] = (PE - PP) / PP * 100

# ----------------------------
# Plot
# ----------------------------
fig, ax1 = plt.subplots(figsize=(8,5))

ax2 = ax1.twinx()

ax1.plot(tau_grid, Delta_IPP, color='red', linewidth=2, label=r'$\Delta_{\mathrm{IPP}}$')
ax2.plot(tau_grid, Delta_EFS, color='blue', linewidth=2, linestyle='--', label=r'$\Delta_{\mathrm{EFS}}$')

ax1.set_xlabel(r'Intervention duration $\tau$')
ax1.set_ylabel(r'$\Delta_{\mathrm{IPP}}$ (%)', color='red')
ax2.set_ylabel(r'$\Delta_{\mathrm{EFS}}$ (%)', color='blue')

ax1.tick_params(axis='y', colors='red')
ax2.tick_params(axis='y', colors='blue')

ax1.grid(True)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

plt.title('Trade-off intensity vs intervention duration')

plt.tight_layout()
plt.show()
