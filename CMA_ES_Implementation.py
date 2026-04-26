"""
CMA-ES Optimization of Time-Dependent Control in SIR Model

This script performs derivative-free optimization of a time-dependent control
function using CMA-ES. The objective balances infection peak prevalence (IPP)
and final epidemic size (EFS).

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import solve_ivp, odeint
from scipy.interpolate import interp1d
import cma

# ===============================
# MODEL PARAMETERS
# ===============================
beta = 3 / 14
gamma = 1 / 14
R0 = beta / gamma

tau = 60
T = 400 

y0 = [0.999999, 0.000001, 0.0]

"""
---------------------------------------------------------
Optional: Compute Imax_opt (reference optimal IPP)
---------------------------------------------------------

This value is obtained via grid search over (t_p, f, sigma).
Uncomment the block below to recompute.
"""

##ti_grid = np.linspace(50, 102, 53)
##f_grid = np.linspace(0, 1, 61)
##sigma_grid = np.linspace(0.1, 0.9, 91)
##
##def sir_ms(y, t, beta, gamma, ti, f, sigma):
##    if ti <= t <= ti + f*tau:
##        b = beta * sigma
##    elif ti + f*tau < t <= ti + tau:
##        b = 0
##    else:
##        b = beta
##    S, I, R = y
##    return [-b*S*I, b*S*I - gamma*I, gamma*I]
##
##Imax_all = []
##t = np.linspace(0, 400, 401)
##
##for ti in ti_grid:
##    for f in f_grid:
##        for sigma in sigma_grid:
##            sol = odeint(sir_ms, y0, t, args=(beta, gamma, ti, f, sigma))
##            Imax_all.append(np.max(sol[:,1]))
##
##Imax_opt = np.min(Imax_all)
##print('Imax_opt:',Imax_opt)

Imax_opt = 0.0944957329112984 # precomputed reference opt IPP

# ===============================
# SIR MODEL WITH TIME DEPENDENT CONTROL
# ===============================
def sir_model(t, y, beta, gamma, sigma_func, ti, tau):
    if ti <= t <= ti + tau:
        b = beta * sigma_func(t)
    else:
        b = beta
    S, I, R = y
    return [-b * S * I, b * S * I - gamma * I, gamma * I]

def final_size_eq(x, S_end, I_end, R0):
    return (1 - x) + (1/R0)*np.log(1/(1-x)) - (S_end + I_end + (1/R0)*np.log(1/S_end))

# Objective weight ratio
q = 0.25 # 0.109

# Control discretization
n_controls = 10

# ===============================
# OBJECTIVE FUNCTION
# ===============================
eval_counter = [0]

def objective_cma(x):
    eval_counter[0] += 1

    ti = x[0]
    sigma_values = np.clip(x[1:], 0, 1)

    # Interpolated control
    t_control = np.linspace(ti, ti + tau, n_controls)
    sigma_func = interp1d(
        t_control, sigma_values,
        kind='linear',
        bounds_error=False,
        fill_value="extrapolate"
    )

    # Solve system (moderate resolution)
    t_eval = np.linspace(0, T, 501)
    sol = solve_ivp(
        sir_model, [0, T], y0, t_eval=t_eval,
        args=(beta, gamma, sigma_func, ti, tau),
        rtol=1e-6, atol=1e-8
    )

    I_max = np.max(sol.y[1])
    
    t_end = ti + tau
    idx = np.argmin(np.abs(sol.t - t_end))
    S_end = sol.y[0, idx]
    I_end = sol.y[1, idx]

    R_inf = brentq(
        lambda x: final_size_eq(x, S_end, I_end, R0),
        0.66, 0.99999999
    )
    S_inf = 1 - R_inf

    # Total burden objective (paper-consistent)
    obj = q * ((I_max - Imax_opt) / Imax_opt) + (1 - R0 * S_inf)

    # Optional progress print
    if eval_counter[0] % 20 == 0:
        print(f"[Eval {eval_counter[0]}] ti={ti:.2f}, Obj={obj:.6f}")

    return obj

# ===============================
# INITIALIZATION
# ===============================
ti_guess = 76
sigma_init = np.full(n_controls, 0.5)

x0 = np.concatenate(([ti_guess], sigma_init))

# ===============================
# CMA-ES SETUP
# ===============================
es = cma.CMAEvolutionStrategy(
    x0,
    0.5,
    {
        'bounds': [[60] + [0]*n_controls, [T - tau] + [1]*n_controls],
        'maxiter': 300,
        'popsize': 16,
        'tolfun': 1e-6,
        'verb_disp': 1,
        'verb_log': 0
    }
)

# ===============================
# RUN OPTIMIZATION
# ===============================
print("Starting CMA-ES optimization...")
res = es.optimize(objective_cma)

# ===============================
# EXTRACT OPTIMAL SOLUTION
# ===============================
x_best = res.result.xbest

ti_opt = x_best[0]
sigma_opt = np.clip(x_best[1:], 0, 1)


# ===============================
# RECONSTRUCT CONTROL
# ===============================
t_control = np.linspace(ti_opt, ti_opt + tau, n_controls)

sigma_func_opt = interp1d(
    t_control, sigma_opt,
    kind='linear',
    bounds_error=False,
    fill_value="extrapolate"
)

# ===============================
# FINAL SIMULATION
# ===============================
t_plot = np.linspace(0, T, 2000)

sol = solve_ivp(
    sir_model, [0, T], y0, t_eval=t_plot,
    args=(beta, gamma, sigma_func_opt, ti_opt, tau)
)

S, I, R = sol.y

# ===============================
# SAVE RESULTS
# ===============================
np.savetxt("infected_trajectory.txt", np.column_stack((t_plot, I)))
np.savetxt("recovered_trajectory.txt", np.column_stack((t_plot, R)))
np.savetxt("optimal_control.txt", np.column_stack((t_control, sigma_opt)))


# ===============================
# PLOTTING
# ===============================
plt.figure(figsize=(10,4))
plt.plot(t_plot, I, 'r-', label='Infected I(t)')
plt.axvline(ti_opt, color='gray', linestyle='--')
plt.axvline(ti_opt + tau, color='gray', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Infected')
plt.title('Optimal Control: Infection Dynamics')
plt.legend()
plt.grid()

plt.figure(figsize=(10,4))
plt.plot(t_plot, R, 'g-', label='Recovered R(t)')
plt.axvline(ti_opt, color='gray', linestyle='--')
plt.axvline(ti_opt + tau, color='gray', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Recovered')
plt.title('Optimal Control: Final Size')
plt.legend()
plt.grid()

plt.figure(figsize=(8,3))
plt.plot(t_control, sigma_opt, 'b-o')
plt.xlabel('Time')
plt.ylabel(r'$\sigma(t)$')
plt.title('Optimal Control Profile')
plt.grid()

plt.tight_layout()
plt.show()
