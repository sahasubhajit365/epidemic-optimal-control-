import numpy as np
from scipy.optimize import brentq
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ===============================
# PARAMETERS
# ===============================
beta = 3/14
gamma = 1/14
R0 = beta/gamma
tau = 60

ti_grid = np.linspace(1, 102, 60)
f_grid = np.linspace(0, 1, 30)
sigma_grid = np.linspace(0.01, 0.9, 10)

q_values = np.linspace(0.01, 5, 40)

# threshold
I_max_th = 0.15


y0 = [0.999999, 0.000001, 0]
t = np.arange(0, 400, 1)

# ===============================
# ODE SYSTEMS
# ===============================

def sir_ms(y, t, beta, gamma, ti, f, sigma):
    if ti <= t <= ti + f * tau:
        b = beta * sigma
    elif ti + f * tau < t <= ti + tau:
        b = 0
    else:
        b = beta
    S, I, R = y
    return [-b*S*I, b*S*I - gamma*I, gamma*I]

# Final size equation
def final_size_eq(x, S_end, I_end, R0):
    return (1 - x) + (1/R0)*np.log(1/(1-x)) - (S_end + I_end + (1/R0)*np.log(1/S_end))

# ===============================
# STEP 1: Compute I_max^opt (IPP optimal)
# ===============================
I_ms = np.zeros((len(ti_grid), len(f_grid), len(sigma_grid)))

for k, ti in enumerate(ti_grid):
    for i, f in enumerate(f_grid):
        for j, sigma in enumerate(sigma_grid):
            sol = odeint(sir_ms, y0, t, args=(beta, gamma, ti, f, sigma))
            I_ms[k, i, j] = np.max(sol[:, 1])

I_max_opt = np.min(I_ms) 

# ===============================
# STEP 2: Optimize for each q
# ===============================
Imax_q = []
opt_controls = []

for q in q_values:

    best_J = np.inf
    best_Imax = None
    best_control = None

    for ti in ti_grid:
        for f in f_grid:
            for sigma in sigma_grid:

                sol = odeint(sir_ms, y0, t, args=(beta, gamma, ti, f, sigma))

                I_max = np.max(sol[:, 1])
                
                S_end = sol[int(round(ti)+tau), 0]
                I_end = sol[int(round(ti)+tau), 1]
                
                R_inf = brentq(
                    lambda x: final_size_eq(x, S_end, I_end, R0),
                    0.66, 0.99999999
                )
                S_inf = 1 - R_inf

                # normalized objective (paper version)
                J = q * ((I_max - I_max_opt) / I_max_opt) + (1 - R0 * S_inf)

                if J < best_J:
                    best_J = J
                    best_Imax = I_max
                    best_control = (ti, f, sigma)

    Imax_q.append(best_Imax)
    opt_controls.append(best_control)

Imax_q = np.array(Imax_q)

# ===============================
# STEP 3: FEASIBLE REGION
# ===============================
feasible = Imax_q <= I_max_th
infeasible = Imax_q > I_max_th

# ===============================
# STEP 4: PLOT 
# ===============================
plt.figure(figsize=(7,5))

plt.plot(q_values, Imax_q, 'o-', label=r'$I_{\max}^*(q)$')

plt.axhline(I_max_th, color='red', linestyle='--',
            label=r'$I_{\max}^{th}$')

plt.xlabel(r'$q = W_1/W_2$')
plt.ylabel(r'$I_{\max}^*(q)$')
plt.title(f'Feasible Weight Ratios for $I_{{\\max}}^{{\\mathrm{{th}}}} = {I_max_th:.2f}$')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
