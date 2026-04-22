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

ti_grid = np.linspace(50, 102, 53)
f_grid = np.linspace(0, 1, 61)
sigma_grid = np.linspace(0.1, 0.9, 91)

q_values = [0.25, 0.109]

y0 = [0.999999, 0.000001, 0]
t = np.linspace(0, 400, 401)

# ===============================
# ODE
# ===============================
def sir_ms(y, t, beta, gamma, ti, f, sigma):
    if ti <= t <= ti + f*tau:
        b = beta * sigma
    elif ti + f*tau < t <= ti + tau:
        b = 0
    else:
        b = beta
    S, I, R = y
    return [-b*S*I, b*S*I - gamma*I, gamma*I]

def final_size_eq(x, S_end, I_end, R0):
    return (1 - x) + (1/R0)*np.log(1/(1-x)) - (S_end + I_end + (1/R0)*np.log(1/S_end))


# ===============================
# STEP 1: compute Imax_opt
# ===============================
Imax_all = []

for ti in ti_grid:
    for f in f_grid:
        for sigma in sigma_grid:
            sol = odeint(sir_ms, y0, t, args=(beta, gamma, ti, f, sigma))
            Imax_all.append(np.max(sol[:,1]))

Imax_opt = np.min(Imax_all)

# ===============================
# STEP 2: compute Total Burden for each q
# ===============================
J_results = {}
control_results = {}

for q in q_values:

    J_tp = []
    controls_tp = []

    for ti in ti_grid:

        best_J = np.inf
        best_control = None

        for f in f_grid:
            for sigma in sigma_grid:

                sol = odeint(sir_ms, y0, t,
                             args=(beta, gamma, ti, f, sigma))

                I_max = np.max(sol[:,1])

                S_end = sol[int(round(ti)+tau), 0]
                I_end = sol[int(round(ti)+tau), 1]
                
                R_inf = brentq(
                    lambda x: final_size_eq(x, S_end, I_end, R0),
                    0.66, 0.99999999
                )
                S_inf = 1 - R_inf

                J = q*((I_max - Imax_opt)/Imax_opt) + (1 - R0*S_inf)

                if J < best_J:
                    best_J = J
                    best_control = (f, sigma)

        J_tp.append(best_J)
        controls_tp.append((ti, best_control))

    J_results[q] = np.array(J_tp)
    control_results[q] = controls_tp

# ===============================
# PLOTTING
# ===============================
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12
})

fig, axes = plt.subplots(1, 2, figsize=(10,4))
ax1, ax2 = axes

colors = ['#1f77b4', '#2ca02c']

# ---- (A) Total burden vs tp ----

for i, q in enumerate(q_values):
    ax1.plot(ti_grid, J_results[q],
             color=colors[i], lw=2,
             label=rf'$q={q}$')

ax1.set_xlabel(r'$t_p$')
ax1.set_ylabel(r'$\tilde{J}$')
ax1.set_title('(A)')
ax1.grid(True)
ax1.legend()

# ---- (B) Optimal controls ----
opt_control_q = {}

for q in q_values:
    idx = np.argmin(J_results[q])
    tp_opt = ti_grid[idx]
    f_opt, sigma_opt = control_results[q][idx][1]

    opt_control_q[q] = (tp_opt, f_opt, sigma_opt)
    
for i, q in enumerate(q_values):

    ti, f, sigma = opt_control_q[q]

    t_control = [0, ti, ti, ti+f*tau, ti+f*tau, ti+tau, ti+tau, 200]
    b_control = [1, 1, sigma, sigma, 0, 0, 1, 1]

    ax2.plot(t_control, b_control,
             color=colors[i], alpha=0.6)

ax2.set_xlabel('time')
ax2.set_ylabel(r'$b(t)$')
ax2.set_title('(B)')
ax2.set_ylim(-0.05,1.05)
ax2.grid(True)

plt.tight_layout()
plt.show()
